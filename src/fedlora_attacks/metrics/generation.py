"""Instruction-tuning outcome metrics: response perplexity, BLEU, BERTScore.

The classification testbeds report accuracy, which rises. This one reports PERPLEXITY,
which falls — and that sign flip is the thing most likely to be got wrong silently, so it
is stated in one place and carried everywhere:

    accuracy   higher is better -> damage = benign - attacked
    perplexity lower  is better -> damage = attacked - benign

``HIGHER_IS_BETTER`` below is that fact as data, so analysis code can ask instead of
assume. A run that reported a negative damage because nobody flipped the sign would look
like the attack HELPED the model.

**Perplexity is over response tokens only.** The prompt is masked to -100 exactly as in
training, so the number measures what the federation was actually taught to produce. A
perplexity that included prompt tokens would be dominated by the fixed template, which is
identical in every arm and would dilute the effect toward zero.

**Perplexity is token-weighted, not example-weighted.** Summing the loss over tokens and
dividing by the token count is the standard definition; averaging per-example means would
weight a five-token answer as heavily as a two-hundred-token one.

BLEU (sacrebleu) and BERTScore are the generation-side check, run on a small prompt set
every few rounds rather than every round — they need decoding, which costs far more than
a forward pass. Both are optional imports: a missing package degrades that metric to
``None`` with a reason rather than taking the run down at round 20.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from ..models.lora_wrapper import IGNORE_INDEX, LoRAModel

#: Direction of every metric this project reports. Consumed by the damage calculation so
#: the sign is looked up, never re-derived at each call site.
HIGHER_IS_BETTER: dict[str, bool] = {
    "accuracy": True,
    "eval_perplexity": False,
    "eval_loss": False,
    "train_loss": False,
    "bleu": True,
    "bertscore_f1": True,
    # Text-to-SQL (metrics/execution.py). `execution_accuracy` is the reportable outcome;
    # the three rates below are DIAGNOSTICS that say why it is what it is, and they are
    # registered here for the same reason the outcomes are — so any comparison looks the
    # sign up rather than re-deriving it. sql_parse_rate in particular runs the opposite
    # way from the two error rates beside it, which is exactly the kind of thing a call
    # site gets backwards once and never notices.
    "execution_accuracy": True,
    "exact_match": True,
    "sql_parse_rate": True,
    "sql_error_rate": False,
    "sql_timeout_rate": False,
    # Image classification (metrics/classification.py). Declared HERE rather than
    # registered from that module for the same reason the SQL keys are: a dict mutated at
    # import time only holds the keys whose module something happened to import, so
    # `damage()` would raise or not depending on import order. `mean_max_softmax` is a
    # DIAGNOSTIC, not a quality score — its sign is declared so no call site guesses.
    "top1_accuracy": True,
    "macro_f1": True,
    "mean_max_softmax": True,
}


def damage(metric: str, benign: float, attacked: float) -> float:
    """Positive means the attack HURT, whichever direction the metric runs in."""
    try:
        higher_better = HIGHER_IS_BETTER[metric]
    except KeyError:
        raise KeyError(
            f"unknown metric {metric!r}; add it to HIGHER_IS_BETTER so its damage sign is "
            "declared rather than guessed"
        ) from None
    return (benign - attacked) if higher_better else (attacked - benign)


def _batched(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


@torch.no_grad()
def response_perplexity(
    model: LoRAModel,
    prompts: Sequence[str],
    responses: Sequence[str],
    device="cpu",
    batch_size: int = 8,
) -> dict[str, float]:
    """Token-weighted perplexity and mean loss over RESPONSE tokens only.

    Recomputes the cross-entropy from logits rather than reading ``outputs.loss``,
    because that value is a per-batch mean: averaging it across batches would weight a
    batch of short answers equally with a batch of long ones and give a number that
    depends on the batch size. Here the loss is summed over tokens and divided by the
    total token count, so the result is batch-size invariant.
    """
    if not prompts:
        raise ValueError("response_perplexity needs at least one prompt")
    if len(prompts) != len(responses):
        raise ValueError(f"{len(prompts)} prompts vs {len(responses)} responses")
    model.model.eval()
    total_nll, total_tokens = 0.0, 0
    device_type = torch.device(device).type
    for p_batch, r_batch in zip(
        _batched(list(prompts), batch_size), _batched(list(responses), batch_size), strict=True
    ):
        batch = model.encode_causal(p_batch, r_batch, device)
        labels = batch.pop("labels")
        with torch.autocast(
            device_type=device_type,
            dtype=model.amp_dtype or torch.bfloat16,
            enabled=model.amp_dtype is not None,
        ):
            logits = model.model(**batch).logits
        # the causal shift: position t predicts token t+1
        shift_logits = logits[:, :-1, :].float()
        shift_labels = labels[:, 1:]
        nll = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=IGNORE_INDEX,
            reduction="sum",
        )
        total_nll += float(nll)
        total_tokens += int((shift_labels != IGNORE_INDEX).sum())
    if total_tokens == 0:
        raise ValueError(
            "no response tokens survived tokenisation; every example was truncated to "
            "prompt only. Raise model.max_length or lower max_prompt_tokens."
        )
    mean_nll = total_nll / total_tokens
    return {
        "eval_loss": mean_nll,
        # exp() overflows to inf for a badly damaged model, which is a real outcome and
        # not an error: report it rather than clipping the attack's effect away.
        "eval_perplexity": math.exp(mean_nll) if mean_nll < 700 else math.inf,
        "eval_tokens": float(total_tokens),
    }


def corpus_bleu(predictions: Sequence[str], references: Sequence[str]) -> float | None:
    """sacreBLEU over the decoded responses, or None if sacrebleu is unavailable."""
    try:
        import sacrebleu
    except ImportError:
        return None
    if not predictions:
        return None
    return float(sacrebleu.corpus_bleu(list(predictions), [list(references)]).score)


def bertscore_f1(
    predictions: Sequence[str],
    references: Sequence[str],
    model_type: str = "roberta-large",
    device: str = "cpu",
    batch_size: int = 16,
) -> float | None:
    """Mean BERTScore F1, or None if bert-score is unavailable.

    ``model_type`` pulls its own checkpoint from the HF cache on first use, which is why
    the SLURM prefetch job warms it before the array starts — otherwise every array task
    would download roberta-large at round 0, concurrently, into the same cache.
    """
    try:
        from bert_score import score as _score
    except ImportError:
        return None
    if not predictions:
        return None
    _, _, f1 = _score(
        list(predictions),
        list(references),
        model_type=model_type,
        lang="en",
        device=device,
        batch_size=batch_size,
        verbose=False,
    )
    return float(f1.mean())


def generation_metrics(
    model: LoRAModel,
    prompts: Sequence[str],
    references: Sequence[str],
    device="cpu",
    max_new_tokens: int = 128,
    batch_size: int = 8,
    bertscore_model: str = "roberta-large",
    with_bertscore: bool = True,
) -> dict[str, float | None]:
    """Decode greedily, then score against the reference responses.

    Greedy rather than sampled: two arms must differ because the MODEL differs, and
    sampling would add a second source of difference that no seed pairing removes.
    """
    predictions: list[str] = []
    for batch in _batched(list(prompts), batch_size):
        predictions.extend(model.generate(batch, device, max_new_tokens=max_new_tokens))
    out: dict[str, float | None] = {
        "bleu": corpus_bleu(predictions, references),
        "n_generated": float(len(predictions)),
        "mean_prediction_chars": (
            float(sum(len(p) for p in predictions)) / len(predictions) if predictions else 0.0
        ),
    }
    out["bertscore_f1"] = (
        bertscore_f1(predictions, references, bertscore_model, device) if with_bertscore else None
    )
    return out
