"""The generation-side outcome metric, selected by task — ONE implementation, two callers.

WHY THIS IS A SHARED MODULE AND NOT A BRANCH IN EACH CALLER.

``central/baseline.py`` and ``fl/server.py`` both need to score a generative model on a
held-out set, and the whole point of running both is to COMPARE their numbers. Two
independent implementations of "decode, extract, execute, compare" would be the textbook
way to end up with two plausible numbers that are not measuring the same thing — and the
divergence would be invisible, because each side would look internally consistent. So the
decision of what to compute, and how, lives here once.

The parameters are plain values rather than a config object for the same reason: the two
callers carry different config types (``CentralConfig`` and ``ServerConfig``), and making
this function accept either would reintroduce the branch it exists to remove.

WHAT SELECTS THE METRIC. ``db_root`` being set, not a dataset-name string. The metric
follows what the run can actually DO — execute SQL against real databases — so a config
that forgot the database root fails loudly at the first evaluation rather than silently
scoring SQL with BLEU and producing a smooth, meaningless curve.

DECODING IS ALWAYS BATCHED. transformers 4.44 materialises ``logits.float()`` over the
whole padded prompt on the first forward of ``generate``, so peak memory is
``n x padded_len x vocab x 4`` bytes. At n=300, ~270 padded tokens and Qwen's 151936-token
vocabulary that is ~49 GB — which killed a 48 GB A6000 at its first evaluation, after
training had run perfectly (job 10001). The batching is here so neither caller can omit it.
"""

from __future__ import annotations

from ..data.base import TaskData
from ..models.lora_wrapper import LoRAModel
from .generation import _batched, generation_metrics


def is_sql_task(data: TaskData, db_root: str | None) -> bool:
    """True when the outcome metric is execution accuracy rather than BLEU/BERTScore."""
    return bool(db_root) and getattr(data, "test_responses", None) is not None


def decode(model: LoRAModel, prompts, device, *, max_new_tokens: int, batch_size: int) -> list[str]:
    """Greedy continuations, in batches. See the module docstring for why batching is here."""
    predictions: list[str] = []
    for chunk in _batched(list(prompts), batch_size):
        predictions.extend(model.generate(chunk, device, max_new_tokens=max_new_tokens))
    return predictions


def outcome_metrics(
    model: LoRAModel,
    data: TaskData,
    n: int,
    device,
    *,
    max_new_tokens: int = 128,
    batch_size: int = 8,
    bertscore_model: str = "roberta-large",
    db_root: str | None = None,
    sql_timeout_s: float = 10.0,
    gold_cache=None,
) -> dict[str, float]:
    """Task-appropriate generation-side metrics over the first ``n`` held-out examples.

    Text-to-SQL gets execution accuracy plus exact match and the three diagnostic rates;
    every other generative task keeps BLEU/BERTScore. Response perplexity is NOT computed
    here — both callers compute it every round while this is sampled every few rounds, so
    folding it in would either make it expensive or make the cadences disagree.
    """
    prompts = list(data.test_texts[:n])
    references = list(data.test_responses[:n])

    if not is_sql_task(data, db_root):
        return generation_metrics(
            model,
            prompts,
            references,
            device,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            bertscore_model=bertscore_model,
        )

    from .execution import execution_accuracy

    # label_names[label] is the db_id — see data/spider.py for why the label slot carries
    # the database rather than a category.
    if not data.label_names:
        raise ValueError(
            "a SQL run needs data.label_names to recover each example's db_id; the loader "
            "did not populate it"
        )
    db_ids = [data.label_names[int(lbl)] for lbl in data.test_labels[:n]]
    predictions = decode(
        model, prompts, device, max_new_tokens=max_new_tokens, batch_size=batch_size
    )
    return execution_accuracy(
        predictions,
        references,
        db_ids,
        db_root,
        timeout_s=sql_timeout_s,
        cache=gold_cache,
    )
