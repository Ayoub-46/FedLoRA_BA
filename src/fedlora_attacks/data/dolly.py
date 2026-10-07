"""Databricks Dolly-15k: the instruction-tuning testbed (LLaMA-3.2-3B spine).

BERT/SST-2 is the microscope; this is the specimen the reviewer asks about. Nothing in
the geometry changes — `dW_l = A_l @ B_l` per adapted layer, gauge group `prod_l GL(r)` —
but the task does, and three things follow that the classification path never had to
handle.

**1. There is no test split.** Dolly ships as a single ``train`` split of ~15k rows, so
the held-out set is CARVED here, deterministically, from ``subsample_seed``. Train and
eval indices are disjoint by construction (``build_splits`` asserts it) and identical
across every cell of a sweep, which is the same discipline ``data/base.build_dataset``
already applies to the SST-2 subsample: the run seed varies the partition and the batch
order, the subsample seed decides WHICH examples exist at all.

**2. The "label" is the CATEGORY.** Dolly's eight categories (``brainstorming`` ...
``summarization``) play exactly the role SST-2's sentiment label plays: they are what the
Dirichlet is drawn over, so ``partition.dirichlet_partition`` and the JSD skew index in
``analysis/envelope`` work unchanged and heterogeneity stays MEASURED rather than
asserted. The mapping category -> int is alphabetical and pinned by
``CATEGORIES``; it is stored on the dataset so ``partition.json``'s histograms remain
interpretable without this module.

**3. Prompt and response are kept SEPARATE.** They are never concatenated here, because
the causal-LM loss must score response tokens only — the prompt is masked to -100. A
loader that returned one joined string would make that masking impossible downstream and
would silently train the model to generate the instruction as well as the answer. The
join happens once, inside the tokenizer, where the boundary is still known.

Requires network on first use; afterwards it is served from the HF cache.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

#: Dolly's eight categories, alphabetical. The index of a name in this tuple IS its
#: integer label, so the mapping is stable across runs, machines and dataset revisions —
#: deriving it from the observed rows instead would silently renumber every client's
#: histogram if a future revision dropped or added a category.
CATEGORIES: tuple[str, ...] = (
    "brainstorming",
    "classification",
    "closed_qa",
    "creative_writing",
    "general_qa",
    "information_extraction",
    "open_qa",
    "summarization",
)

CATEGORY_TO_ID = {name: i for i, name in enumerate(CATEGORIES)}

#: The prompt template. Fixed and boring on purpose: it is not an experimental axis, and
#: a template that varied between the benign and attacked arms would be a confound in
#: every trajectory comparison. `### Response:` is the boundary the loss masks at.
PROMPT_WITH_CONTEXT = "### Instruction:\n{instruction}\n\n### Input:\n{context}\n\n### Response:\n"
PROMPT_NO_CONTEXT = "### Instruction:\n{instruction}\n\n### Response:\n"


class DollyError(ValueError):
    """Raised instead of returning a silently degraded dataset."""


@dataclass(frozen=True)
class DollySplits:
    """Prompts, responses and category ids for both splits, plus the category names."""

    train_prompts: list[str]
    train_responses: list[str]
    train_categories: list[int]
    test_prompts: list[str]
    test_responses: list[str]
    test_categories: list[int]
    category_names: tuple[str, ...] = CATEGORIES


def format_prompt(instruction: str, context: str | None) -> str:
    """The prompt half of one example. Never includes the response."""
    instruction = (instruction or "").strip()
    context = (context or "").strip()
    if context:
        return PROMPT_WITH_CONTEXT.format(instruction=instruction, context=context)
    return PROMPT_NO_CONTEXT.format(instruction=instruction)


def category_id(name: str) -> int:
    """Category name -> pinned integer label, raising on anything unrecognised.

    A silent fallback (say, mapping the unknown to 0) would corrupt the partition's
    label histograms and therefore the measured skew, which is the axis every result is
    indexed by. Better to fail on a dataset revision than to mis-report alpha.
    """
    try:
        return CATEGORY_TO_ID[name.strip().lower()]
    except KeyError:
        raise DollyError(
            f"unknown Dolly category {name!r}; expected one of {CATEGORIES}. "
            "A dataset revision has changed the category set, and the integer labels "
            "in CATEGORIES are load-bearing for the partition's skew index."
        ) from None


def build_splits(
    rows: Sequence[dict],
    num_train: int | None,
    num_test: int | None,
    seed: int,
) -> DollySplits:
    """Carve disjoint train/eval splits from Dolly's single split, deterministically.

    Split BEFORE truncating, so that shrinking ``num_train`` for a cheap pilot cannot
    leak examples into the eval set: the eval half is taken from the tail of a seeded
    permutation and the train half from the head, and the two never overlap regardless
    of the requested sizes.

    Kept free of any HuggingFace import so it is unit-testable without network.
    """
    import numpy as np

    if not rows:
        raise DollyError("build_splits received no rows; Dolly did not load")
    order = np.random.default_rng(seed).permutation(len(rows))
    n_test = min(int(num_test) if num_test else 1000, max(1, len(rows) // 5))
    test_idx = order[-n_test:]
    train_idx = order[: len(rows) - n_test]
    if num_train is not None:
        train_idx = train_idx[: int(num_train)]
    if len(train_idx) == 0:
        raise DollyError(
            f"no training rows left: {len(rows)} total, {n_test} held out, "
            f"num_train={num_train}"
        )
    overlap = set(train_idx.tolist()) & set(test_idx.tolist())
    if overlap:  # pragma: no cover - guarded by construction, asserted anyway
        raise DollyError(f"train/eval overlap of {len(overlap)} rows; the split is wrong")

    def take(idx):
        prompts, responses, cats = [], [], []
        for i in idx:
            row = rows[int(i)]
            response = (row.get("response") or "").strip()
            if not response:
                continue  # an empty target teaches nothing and makes perplexity undefined
            prompts.append(format_prompt(row.get("instruction", ""), row.get("context", "")))
            responses.append(response)
            cats.append(category_id(row.get("category", "")))
        return prompts, responses, cats

    tr_p, tr_r, tr_c = take(train_idx)
    te_p, te_r, te_c = take(test_idx)
    if not tr_p or not te_p:
        raise DollyError(
            f"after dropping empty responses: {len(tr_p)} train, {len(te_p)} eval rows"
        )
    return DollySplits(tr_p, tr_r, tr_c, te_p, te_r, te_c)


def load_dolly(num_train: int | None, num_test: int | None, seed: int) -> DollySplits:
    """Download (once) and split Dolly-15k. ``seed`` is the SUBSAMPLE seed, not the run seed."""
    from datasets import load_dataset

    ds = load_dataset("databricks/databricks-dolly-15k")["train"]
    return build_splits(list(ds), num_train, num_test, seed)
