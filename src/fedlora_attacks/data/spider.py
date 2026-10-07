"""Spider text-to-SQL: the testbed with real headroom.

WHY THIS EXISTS ALONGSIDE DOLLY. Qwen2.5-3B already models Dolly-style prose at held-out
perplexity ~5.9 cold, so instruction tuning there moves the metric ~1.7% and what movement
there is turns out to be calibration rather than capability (BLEU held flat while
perplexity degraded). A benign task with that little dynamic range is a weak reference for
the attacked arms, whose damage is measured against it. Execution accuracy on Spider is
hard, binary and semantically grounded: a base model scores near zero, fine-tuning moves it
tens of points, and there is something real for an attack to destroy.

**THE PROMPT-ORDER TRAP, and why the schema is budgeted HERE.**

``LoRAModel.encode_causal`` right-truncates the prompt (``p_ids[: max_prompt_tokens]``).
Spider prompts are schema + question, and Spider schemas are long — median ~212 tokens, p90
~655, max ~2067 across the 166 databases. Put the schema first and let blunt truncation
handle the overflow and it eats **the question**: the model is handed a database and asked
nothing. Execution accuracy then collapses for a reason no metric in the run would explain.

So the question goes LAST, immediately before ``### SQL:``, and the schema is truncated
here — by whole tables, from the end — to a budget that leaves the question room. What is
lost is a table the query may not have needed; what is never lost is the question.

**Budgeting is in CHARACTERS, not tokens.** ``build_dataset`` runs before
``build_tokenizer`` (see ``data/base.py``), so no tokenizer exists at this point. The
character budget uses a LOWER BOUND on the tokenizer's chars-per-token ratio — see
``CHARS_PER_TOKEN``, which is 2.5 against a measured minimum of 2.88 across all 8034 real
prompts. An average-looking 3.2 let 0.88% of prompts overflow the token cap and lose their
question to the model's own truncation, which is the very failure this budgeting exists to
prevent. Every example records ``schema_truncated``; the run reports the rate, so
truncation is a number in the summary rather than an invisible ceiling on the score.

**BOTH HALVES ARE PINNED BY COMMIT.** The question/SQL pairs and the schemas/databases come
from two different repositories, and ``load_spider`` cross-checks that they describe the
same benchmark — see ``SPIDER_REVISION``, ``SPIDER_DB_REVISION`` and
``assert_consistent_with_dataset``.

**THE "LABEL" IS THE DATABASE.** ``TaskData`` is reused unchanged: ``train_labels`` holds
the index of the example's ``db_id`` and ``label_names`` holds the ``db_id`` tuple, so
``label_names[label]`` recovers the database the executor must run against. That costs no
schema change and gives a federated arm a database-valued axis to partition over.

**A CORRECTION, because the obvious reading of that is wrong.** Putting the database in the
label slot does let ``dirichlet_partition`` run over db_id, but that does NOT produce the
federation the phrase "each client owns different databases" suggests.

MEASURED at alpha=0.3, N=10 (E-SPIDER, job 10071): each client holds 69-120 of the 140
training databases (median 82), the mean pairwise Jaccard of clients' database SETS is
0.453, NO database is held exclusively by one client, and the median database is split
across 6 clients. Shard sizes come out near-uniform (637-734).

So this is MILD non-IID: each database is genuinely skewed (its largest client holds ~47%
of it) but every client still sees most of the task distribution. The cause is structural,
not a bug — `dirichlet_partition` draws an INDEPENDENT Dirichlet per class, so with 140
classes and 10 clients the draws average out across clients. alpha=0.3 over Dolly's 8
categories is strongly skewed; alpha=0.3 over 140 databases is not.

The realistic text-to-SQL federation — an organisation owns its databases and never sees
anyone else's schemas — needs a different partition primitive: whole databases assigned
disjointly to clients, not a Dirichlet split of each database's questions across them. That
is a materially harder setting (a client would train on ~14 schemas and the aggregate must
generalise to all 140) and it is NOT what this label slot gives you for free.

**SPLITS.** Spider ships ``train`` (7000) and ``validation`` (1034). Unlike Dolly, the
held-out set is NOT carved by us — ``validation`` is the standard dev set every published
Spider number is reported on, so using anything else would make our numbers incomparable to
the literature for no gain.

Requires network on first use for the question/SQL pairs, plus the SQLite databases, which
the parquet does NOT ship — see ``scripts/slurm/prefetch_spider.sbatch``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

#: The prompt template. Fixed and boring on purpose: it is not an experimental axis, and a
#: template that varied between the benign and attacked arms would be a confound in every
#: trajectory comparison. `### SQL:` is the boundary the loss masks at, and the question is
#: the last thing before it — see the module docstring.
PROMPT = "### Database schema:\n{schema}\n\n### Question:\n{question}\n\n### SQL:\n"

#: Chars-per-token for the budget. This must be a LOWER BOUND on the tokenizer's true
#: ratio, not an average, and the difference is not academic.
#:
#: The budget converts a token cap into a character cap. If the constant is too generous,
#: the character budget admits a prompt that tokenizes to MORE than the cap, and
#: ``encode_causal`` then right-truncates it — cutting the END of the prompt, which is the
#: question. That is precisely the failure this module exists to prevent, and it fires
#: silently on exactly the long-schema examples where the schema matters most.
#:
#: Measured over all 8034 real Spider prompts with the Qwen2.5 tokenizer: the ratio runs
#: median 4.04, p1 2.97, MINIMUM 2.88. At 3.2 (an average-looking guess) 71 prompts (0.88%)
#: overflowed a 896-token cap, the worst reaching 991 tokens. 2.5 sits below the observed
#: minimum with margin for schema text this corpus does not contain.
#:
#: The cost of being conservative is a slightly tighter schema budget, which drops a
#: trailing table — graceful. The cost of being generous is losing the question — not.
CHARS_PER_TOKEN = 2.5

#: What the template itself costs, so the schema budget accounts for it.
_TEMPLATE_OVERHEAD = len(PROMPT.format(schema="", question=""))

#: The question/SQL pairs, PINNED BY COMMIT. `main` is a moving target: a re-tag or a
#: re-upload would change the corpus under a config hash that did not move, and every
#: execution-accuracy number recorded before it would silently stop being comparable to
#: every number after. Invariant 5 is about exactly this.
SPIDER_DATASET = "xlangai/spider"
SPIDER_REVISION = "0c350918f3f29ec754f1181c65cdce76cd6c133c"

#: The SQLite databases and tables.json. `xlangai/spider` ships the pairs ONLY — no
#: schemas, no databases — so the executable half of this benchmark has to come from
#: somewhere else, and that somewhere is a third-party re-host of the canonical Yale
#: distribution. Two defences against that being a reproducibility hole:
#:   * the bundle is pinned by commit AND verified by SHA256 in prefetch_spider.sbatch, so
#:     a changed re-host fails loudly instead of shifting the benchmark;
#:   * `assert_consistent_with_dataset` cross-checks the bundle against the pinned pairs,
#:     so "these two artefacts describe the same benchmark" is CHECKED, not assumed.
SPIDER_DB_REPO = "HAL-9001/spider-databases"
SPIDER_DB_REVISION = "4a01bbac6520cd35b216db9e1724e5e1ada60aa4"
SPIDER_DB_SHA256 = "00636695dabed6b5f4b8328a16b13e069a2f16591d5efcce57660669c85b121b"


class SpiderError(ValueError):
    """Raised instead of returning a silently degraded dataset."""


@dataclass(frozen=True)
class SpiderSplits:
    """Prompts, gold SQL and database ids for both splits, plus the db_id vocabulary."""

    train_prompts: list[str]
    train_responses: list[str]
    train_db_ids: list[int]
    test_prompts: list[str]
    test_responses: list[str]
    test_db_ids: list[int]
    db_names: tuple[str, ...]
    train_truncated: list[bool]
    test_truncated: list[bool]

    @property
    def truncation_rate(self) -> float:
        """Fraction of examples whose schema did not fit. Reported, never silent."""
        flags = list(self.train_truncated) + list(self.test_truncated)
        return sum(flags) / len(flags) if flags else 0.0


def serialize_schema(tables: Sequence[str], columns: Sequence[Sequence[str]]) -> list[str]:
    """One compact string per table: ``name : col , col , col``.

    Returned per-table rather than pre-joined so the budget can drop WHOLE TABLES. A
    character-truncated schema string would end mid-column-name and present the model with a
    table that does not exist, which is worse than a table that is absent.
    """
    return [f"{t} : {', '.join(cols)}" for t, cols in zip(tables, columns, strict=True)]


def fit_schema(table_strings: Sequence[str], budget_chars: int) -> tuple[str, bool]:
    """Join tables under a character budget, dropping whole tables from the end.

    Returns ``(schema, truncated)``. Always returns at least one table even when that one
    table exceeds the budget: a prompt with an over-long schema is recoverable (the model
    sees a truncated table list), a prompt with no schema at all is not.
    """
    if not table_strings:
        return "", False
    kept: list[str] = []
    used = 0
    for entry in table_strings:
        cost = len(entry) + 1  # the joining newline
        if kept and used + cost > budget_chars:
            return "\n".join(kept), True
        kept.append(entry)
        used += cost
    # Everything fitted structurally; the single-table case may still be over budget.
    return "\n".join(kept), used > budget_chars


def format_prompt(
    schema_tables: Sequence[str], question: str, max_prompt_tokens: int
) -> tuple[str, bool]:
    """The prompt half of one example, with the question guaranteed to survive.

    The question's own length is subtracted from the schema budget FIRST. That ordering is
    the whole point: whatever has to go, it is never the thing being asked.
    """
    question = (question or "").strip()
    budget = int(max_prompt_tokens * CHARS_PER_TOKEN) - _TEMPLATE_OVERHEAD - len(question)
    schema, truncated = fit_schema(schema_tables, max(budget, 0))
    return PROMPT.format(schema=schema, question=question), truncated


def load_schemas(tables_json: str | Path) -> dict[str, list[str]]:
    """``db_id -> per-table schema strings``, from Spider's ``tables.json``.

    ``column_names`` is a flat list of ``[table_index, column_name]`` pairs with a leading
    ``[-1, "*"]`` sentinel; the sentinel is dropped because ``*`` belongs to no table and
    attaching it to table 0 would put a phantom column in the first table of every schema.
    """
    path = Path(tables_json)
    if not path.exists():
        raise SpiderError(
            f"no tables.json at {path}. The xlangai/spider parquet does NOT ship schemas or "
            "databases; run scripts/slurm/prefetch_spider.sbatch first."
        )
    out: dict[str, list[str]] = {}
    for entry in json.loads(path.read_text()):
        tables = entry["table_names_original"]
        cols: list[list[str]] = [[] for _ in tables]
        for tbl_idx, col_name in entry["column_names_original"]:
            if tbl_idx < 0:  # the '*' sentinel
                continue
            cols[tbl_idx].append(col_name)
        out[entry["db_id"]] = serialize_schema(tables, cols)
    if not out:
        raise SpiderError(f"tables.json at {path} contained no databases")
    return out


def build_splits(
    train_rows: Sequence[dict],
    test_rows: Sequence[dict],
    schemas: dict[str, list[str]],
    num_train: int | None,
    num_test: int | None,
    seed: int,
    max_prompt_tokens: int = 896,
) -> SpiderSplits:
    """Assemble both splits. Kept free of any HuggingFace import so it unit-tests offline.

    Subsampling is a seeded permutation, matching ``data/dolly.build_splits``: the subsample
    seed decides WHICH examples exist, the run seed varies the batch order. Spider's splits
    are already disjoint upstream, so unlike Dolly there is nothing to carve.
    """
    import numpy as np

    if not train_rows or not test_rows:
        raise SpiderError(
            f"Spider did not load: {len(train_rows)} train, {len(test_rows)} validation rows"
        )
    # A stable db_id vocabulary: sorted, so the integer labels do not depend on row order
    # and a partition histogram stays interpretable without this module.
    db_names = tuple(sorted(schemas))
    db_to_id = {name: i for i, name in enumerate(db_names)}

    def take(rows, limit):
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(rows))
        if limit is not None:
            order = order[: int(limit)]
        prompts, responses, labels, truncated = [], [], [], []
        for i in order:
            row = rows[int(i)]
            db_id = row.get("db_id", "")
            query = (row.get("query") or "").strip()
            question = (row.get("question") or "").strip()
            if not query or not question:
                continue  # an empty target teaches nothing
            if db_id not in schemas:
                raise SpiderError(
                    f"row references database {db_id!r}, which is not in tables.json. "
                    "The parquet and the database bundle are from different revisions."
                )
            prompt, was_truncated = format_prompt(schemas[db_id], question, max_prompt_tokens)
            prompts.append(prompt)
            responses.append(query)
            labels.append(db_to_id[db_id])
            truncated.append(was_truncated)
        return prompts, responses, labels, truncated

    tr_p, tr_r, tr_l, tr_t = take(train_rows, num_train)
    te_p, te_r, te_l, te_t = take(test_rows, num_test)
    if not tr_p or not te_p:
        raise SpiderError(f"after filtering: {len(tr_p)} train, {len(te_p)} validation rows")
    return SpiderSplits(tr_p, tr_r, tr_l, te_p, te_r, te_l, db_names, tr_t, te_t)


def load_spider(
    num_train: int | None,
    num_test: int | None,
    seed: int,
    tables_json: str | Path | None = None,
    max_prompt_tokens: int = 896,
) -> SpiderSplits:
    """Download (once) and assemble Spider. ``seed`` is the SUBSAMPLE seed, not the run seed."""
    from datasets import load_dataset

    ds = load_dataset(SPIDER_DATASET, revision=SPIDER_REVISION)
    schemas = load_schemas(tables_json or default_tables_json())
    # The pairs and the schemas come from DIFFERENT repositories. Checking they agree is
    # the difference between a reproducible benchmark and two artefacts that happen to sit
    # in the same directory.
    assert_consistent_with_dataset(schemas, list(ds["train"]) + list(ds["validation"]))
    return build_splits(
        list(ds["train"]),
        list(ds["validation"]),
        schemas,
        num_train,
        num_test,
        seed,
        max_prompt_tokens,
    )


def assert_consistent_with_dataset(schemas: dict[str, list[str]], rows: Sequence[dict]) -> None:
    """Every database the pinned pairs reference must exist in the pinned bundle.

    The two halves of this benchmark come from two repositories: `xlangai/spider` has the
    questions and gold SQL, the re-hosted bundle has the schemas and the .sqlite files. If
    they ever drift apart, the failure is quiet and bad — a missing schema means a prompt
    describing the wrong database, and execution accuracy measures nothing while still
    producing a number between 0 and 1.
    """
    referenced = {r.get("db_id", "") for r in rows}
    missing = sorted(referenced - set(schemas))
    if missing:
        raise SpiderError(
            f"{len(missing)} database(s) referenced by {SPIDER_DATASET}@{SPIDER_REVISION[:8]} "
            f"are absent from the schema bundle {SPIDER_DB_REPO}@{SPIDER_DB_REVISION[:8]}: "
            f"{missing[:5]}. The two artefacts are from different revisions of Spider; "
            "execution accuracy computed across them would be meaningless."
        )


def default_db_root() -> Path:
    """Where the extracted Spider databases live. Override with ``SPIDER_DATA_DIR``."""
    import os

    root = os.environ.get("SPIDER_DATA_DIR", ".spider_data")
    return Path(root) / "spider_data" / "database"


def default_tables_json() -> Path:
    import os

    root = os.environ.get("SPIDER_DATA_DIR", ".spider_data")
    return Path(root) / "spider_data" / "tables.json"
