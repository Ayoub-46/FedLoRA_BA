"""SQL extraction: turn a decoded continuation into one executable statement, or nothing.

WHY THIS IS A SEPARATE STEP, AND WHY ITS FAILURES ARE COUNTED SEPARATELY.

Execution accuracy scores the SQL a model emits by running it. A base model does not emit
SQL — it emits prose, or a fenced ```sql block, or the query followed by a chatty
explanation, or two queries. Feed that straight to sqlite3 and every one of those becomes
an exception, exceptions count as wrong, and execution accuracy reads 0.00 for rounds 0-5.

That number is indistinguishable from "the model has not learned the task yet", and it is
the wrong diagnosis: the model may be producing perfectly good SQL wrapped in three
backticks. The failure is in the harness, the fix is a one-line regex, and the week is
spent tuning a learning rate instead.

So extraction is explicit, and ``sql_parse_rate`` is reported ALONGSIDE
``sql_error_rate`` rather than folded into it:

    low parse rate                  -> prompt/format problem, fix the harness
    high parse rate, low accuracy   -> the model is genuinely still learning

Without that split, round 0 is unreadable. With it, one glance separates the two.

WHAT THIS DELIBERATELY DOES NOT DO. No SQL dialect validation, no AST parsing, no
normalisation beyond whitespace and the trailing semicolon. Deciding whether a statement
is *correct* is the executor's job, and a stricter extractor here would start silently
rejecting valid queries — turning a model error into a harness error, which is the exact
confusion this module exists to prevent.
"""

from __future__ import annotations

import re

#: A fenced block, ```sql ... ``` or plain ``` ... ```. Non-greedy so the FIRST block wins:
#: a model that emits a query and then an explanation in a second block should be scored on
#: the query it actually answered with.
_FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
#: An unterminated fence — the continuation hit max_new_tokens mid-block. Common, and
#: recoverable: everything after the opening fence is the candidate.
_FENCE_OPEN = re.compile(r"```(?:sql)?\s*(.*)", re.DOTALL | re.IGNORECASE)
#: Where a prompt-template echo starts. The model has stopped answering by this point.
_STOP_MARKERS = ("###", "\n\n", "Question:", "Answer:", "Explanation:")
#: A statement must begin with one of these to be SQL at all. SELECT and WITH cover
#: Spider's entire gold set; nothing in the benchmark is a DML or DDL statement, so a
#: prediction that starts with INSERT/DROP is not a near-miss to be executed, it is a
#: non-answer — and executing it is exactly what the read-only connection exists to stop.
_SQL_START = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)


def normalize_sql(sql: str) -> str:
    """Collapse whitespace and drop the trailing semicolon, for comparison only.

    Never applied before execution: sqlite is whitespace-insensitive anyway, and
    normalising what gets executed would mean scoring a statement the model did not emit.
    """
    return re.sub(r"\s+", " ", (sql or "").strip()).rstrip(";").strip()


def extract_sql(text: str) -> str | None:
    """The first executable statement in a decoded continuation, or None if there is none.

    Order matters: a fenced block is unwrapped BEFORE the stop markers are applied, because
    a well-formed ```sql block legitimately contains blank lines, and cutting at the first
    blank line would truncate the query mid-statement.

    Returns None rather than a best-effort string when nothing looks like SQL. That None is
    the ``sql_parse_rate`` signal; a caller that coerced it to "" would score it as an
    execution error and lose the distinction this module exists to preserve.
    """
    if not text:
        return None
    candidate = text.strip()

    # A CLOSED fence is itself the delimiter, so the stop markers must not be applied
    # inside it: a well-formed ```sql block legitimately contains blank lines, and cutting
    # at the first one truncates the query mid-statement — turning a correct answer into a
    # syntax error, i.e. a harness failure wearing a model failure's clothes. An UNTERMINATED
    # fence has no closing delimiter, so it still needs the markers.
    delimited = False
    fenced = _FENCE.search(candidate)
    if fenced:
        candidate = fenced.group(1).strip()
        delimited = True
    else:
        opened = _FENCE_OPEN.search(candidate)
        if opened:
            candidate = opened.group(1).strip()

    # First statement only. A second query is a different answer, not more of this one.
    semi = candidate.find(";")
    if semi != -1:
        candidate = candidate[:semi]
    elif not delimited:
        # No terminator and no fence: cut where the model stopped answering and started
        # narrating.
        for marker in _STOP_MARKERS:
            cut = candidate.find(marker)
            if cut > 0:
                candidate = candidate[:cut]

    candidate = candidate.strip()
    if not candidate or not _SQL_START.match(candidate):
        return None
    return candidate


def extraction_report(predictions) -> tuple[list[str | None], float]:
    """Extract a batch and report the parse rate in one pass.

    The rate is over ALL predictions including empty ones: a model that returns nothing is
    failing to parse, not absent from the denominator.
    """
    extracted = [extract_sql(p) for p in predictions]
    if not extracted:
        return extracted, 0.0
    return extracted, sum(x is not None for x in extracted) / len(extracted)
