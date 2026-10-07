"""Execution accuracy for text-to-SQL: run the query, compare the rows.

THE METRIC. Spider's execution accuracy (EX) executes the predicted query and the gold
query against the same SQLite database and asks whether they return the same rows. It is
the reason this testbed exists: unlike perplexity on Dolly (~1.7% of benign headroom, and
degradation there turned out to be calibration rather than capability), EX is hard, binary
and semantically grounded. A base model scores near zero; fine-tuning moves it tens of
points. There is something real for an attack to destroy.

FOUR THINGS THAT ARE EASY TO GET WRONG, AND ARE NOT.

**1. Multiset, not set.** Rows are compared as SORTED LISTS, never as sets. Comparing sets
would score ``SELECT name FROM t`` equal to ``SELECT DISTINCT name FROM t`` — two queries
with different semantics and different answers whenever a name repeats. Official Spider EX
preserves row multiplicity, and dropping it inflates the score in a direction no
sanity check would catch, because the inflated number is still plausible.

**2. Order-insensitivity is CONDITIONAL.** Rows are sorted before comparison only when the
GOLD query has no ``ORDER BY``. When it does, the ordering is part of the answer and a
correctly-sorted-differently result is wrong. Sorting unconditionally would silently pass
every query that got the ordering backwards.

**3. Read-only, always.** Connections are opened with ``file:...?mode=ro``. The SQL being
executed is untrusted model output; a prediction of ``DROP TABLE`` must fail against the
connection rather than against our good intentions. The benchmark databases are shared
state for every future run — corrupting them silently invalidates results nobody will
think to re-check.

**4. Timeouts.** A generated cartesian product over three tables will not finish. Without a
per-query interrupt one bad prediction hangs a cell for the rest of its walltime, and the
job dies at the walltime limit looking like an infrastructure failure. ``set_progress_handler``
interrupts; a timeout scores wrong, like any other failure.

WHAT COUNTS AS WRONG RATHER THAN FATAL. Everything: unparseable output, invalid SQL, a
missing table, a type error, a timeout. Garbage early in training is the normal case, not
an exceptional one, and a metric that raised on it would take down the run at round 0. The
failures are COUNTED and reported by kind (``sql_parse_rate``, ``sql_error_rate``,
``sql_timeout_rate``) so the reason is visible — see ``metrics/sql.py`` for why the parse
rate in particular is kept separate.

GOLD RESULTS ARE CACHED. The gold query for an example returns the same rows at round 0 and
round 24, and for every model. Caching on ``(db_id, gold_sql)`` halves the executions per
evaluation and makes the full 1034-example dev pass affordable.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from pathlib import Path

from .sql import extraction_report, normalize_sql

#: Rows returned before a query is considered runaway. A Spider answer is a handful of
#: rows; a hundred thousand means a cartesian product, and materialising it costs memory
#: as well as time.
MAX_ROWS = 100_000


class ExecutionError(Exception):
    """A query failed. Carried as a value, never raised out of the metric."""


def db_path_for(db_root: str | Path, db_id: str) -> Path:
    """``<db_root>/<db_id>/<db_id>.sqlite``, the layout of the Spider distribution."""
    return Path(db_root) / db_id / f"{db_id}.sqlite"


def _connect_readonly(path: Path) -> sqlite3.Connection:
    """Open read-only, so untrusted SQL cannot mutate a benchmark database."""
    if not path.exists():
        raise ExecutionError(f"no such database: {path}")
    # The URI form is what actually enforces read-only; a plain path silently opens rw.
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5.0)


def run_query(path: Path, sql: str, timeout_s: float = 10.0) -> tuple[list[tuple], str | None]:
    """Execute one query read-only. Returns ``(rows, error)``; never raises.

    ``error`` is ``"timeout"`` for an interrupted query and a short string otherwise, so
    the caller can report failures by kind rather than as one undifferentiated bucket.
    """
    # sqlite accepts an empty statement and returns no rows and no error, which would be
    # scored as "ran fine, returned nothing" — indistinguishable from a legitimately empty
    # result set. Reject it here. In the metric path `extract_sql` already turns a blank
    # into a parse failure, so this only guards direct callers, but "ran nothing
    # successfully" is never an answer worth propagating.
    if not (sql or "").strip():
        return [], "empty statement"

    conn = None
    try:
        conn = _connect_readonly(path)
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        return [], f"connect: {exc}"

    # A progress handler is the only portable way to interrupt a long-running sqlite
    # statement from the same thread. It fires every N VM instructions; returning
    # non-zero aborts the statement.
    import time

    deadline = time.monotonic() + timeout_s
    timed_out = False

    def _watchdog() -> int:
        nonlocal timed_out
        if time.monotonic() > deadline:
            timed_out = True
            return 1
        return 0

    try:
        conn.set_progress_handler(_watchdog, 10_000)
        # text_factory keeps undecodable bytes from raising: some Spider databases hold
        # latin-1 values, and a decode error is not a wrong answer, it is a harness crash.
        conn.text_factory = lambda b: b.decode("utf-8", "replace")
        cur = conn.execute(sql)
        rows = cur.fetchmany(MAX_ROWS)
        return [tuple(r) for r in rows], None
    except Exception as exc:  # noqa: BLE001 - every failure is a wrong answer, not a crash
        if timed_out:
            return [], "timeout"
        return [], f"{type(exc).__name__}: {str(exc)[:120]}"
    finally:
        if conn is not None:
            try:
                conn.set_progress_handler(None, 0)
                conn.close()
            except Exception:  # noqa: BLE001 - closing must not mask the real result
                pass


def _comparable(rows: Sequence[tuple], order_matters: bool) -> list:
    """Rows as a MULTISET (sorted list), or in order when the gold query pins the order.

    ``sorted`` on mixed-type tuples raises in Python 3, and Spider rows genuinely mix
    ints, floats, strings and None — so sort on the string form, which is total. The
    comparison stays a multiset comparison either way: equal-length lists of equal items.
    """
    if order_matters:
        return [tuple(r) for r in rows]
    return sorted((tuple(r) for r in rows), key=lambda r: tuple(str(x) for x in r))


def results_match(pred_rows: Sequence[tuple], gold_rows: Sequence[tuple], gold_sql: str) -> bool:
    """Spider EX row comparison: multiset equality, order-sensitive iff gold has ORDER BY."""
    order_matters = "order by" in (gold_sql or "").lower()
    return _comparable(pred_rows, order_matters) == _comparable(gold_rows, order_matters)


class GoldCache:
    """Memoised gold result sets, keyed by ``(db_id, gold_sql)``.

    Gold rows do not change across rounds, seeds or models, so an evaluation that
    re-executed them would double its database work for nothing. Kept as an object rather
    than a module-level dict so a test can assert the cached and uncached paths agree.
    """

    def __init__(self) -> None:
        self._store: dict[tuple[str, str], tuple[list[tuple], str | None]] = {}

    def get(
        self, db_root: str | Path, db_id: str, gold_sql: str, timeout_s: float
    ) -> tuple[list[tuple], str | None]:
        key = (db_id, gold_sql)
        if key not in self._store:
            self._store[key] = run_query(db_path_for(db_root, db_id), gold_sql, timeout_s)
        return self._store[key]


def execution_accuracy(
    predictions: Sequence[str],
    gold: Sequence[str],
    db_ids: Sequence[str],
    db_root: str | Path,
    timeout_s: float = 10.0,
    cache: GoldCache | None = None,
) -> dict[str, float]:
    """Spider execution accuracy plus the diagnostics needed to interpret it.

    ``predictions`` are RAW decoded continuations; extraction happens here so the parse
    rate is measured on what the model actually emitted.

    Returns ``execution_accuracy``, ``exact_match`` (normalised string equality — a strict
    lower bound on EX, and cheap), and three failure rates that say WHY the score is what
    it is. A gold query that itself fails to execute is excluded from the denominator: that
    is a broken benchmark row, not a model error, and counting it would put a floor on the
    achievable score that no amount of training could reach.
    """
    if not (len(predictions) == len(gold) == len(db_ids)):
        raise ValueError(
            f"{len(predictions)} predictions, {len(gold)} gold, {len(db_ids)} db_ids; "
            "execution accuracy needs one of each per example"
        )
    if not predictions:
        raise ValueError("execution_accuracy needs at least one prediction")

    cache = cache if cache is not None else GoldCache()
    extracted, parse_rate = extraction_report(predictions)

    correct = exact = errors = timeouts = scored = 0
    for pred_sql, gold_sql, db_id in zip(extracted, gold, db_ids, strict=True):
        gold_rows, gold_err = cache.get(db_root, db_id, gold_sql, timeout_s)
        if gold_err is not None:
            # Broken benchmark row — not the model's fault, so not in the denominator.
            continue
        scored += 1
        if pred_sql is None:
            continue  # counted by parse_rate, not by sql_error_rate
        if normalize_sql(pred_sql).lower() == normalize_sql(gold_sql).lower():
            exact += 1
        pred_rows, pred_err = run_query(db_path_for(db_root, db_id), pred_sql, timeout_s)
        if pred_err is not None:
            errors += 1
            timeouts += pred_err == "timeout"
            continue
        if results_match(pred_rows, gold_rows, gold_sql):
            correct += 1

    if scored == 0:
        raise ExecutionError(
            "no gold query executed successfully; the database root is probably wrong "
            f"({db_root}). Expected <db_root>/<db_id>/<db_id>.sqlite"
        )
    n = float(scored)
    return {
        "execution_accuracy": correct / n,
        "exact_match": exact / n,
        "sql_parse_rate": parse_rate,
        "sql_error_rate": errors / n,
        "sql_timeout_rate": timeouts / n,
        "sql_scored": n,
    }
