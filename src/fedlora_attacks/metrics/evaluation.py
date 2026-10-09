"""Outcome metrics (plan §3): main-task accuracy and attack success rate (ASR)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..attacks.base import Attack
from ..models.lora_wrapper import LoRAModel


def _batched(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


@torch.no_grad()
def accuracy(
    model: LoRAModel, texts: list[str], labels: list[int], device="cpu", batch_size: int = 64
) -> float:
    """Main-task accuracy on a held-out clean set (model performance)."""
    if not texts:
        return 0.0
    model.model.eval()
    correct = 0
    label_t = torch.tensor(labels)
    offset = 0
    for batch in _batched(texts, batch_size):
        preds = model.logits(batch, device).argmax(dim=-1).cpu()
        correct += (preds == label_t[offset : offset + len(batch)]).sum().item()
        offset += len(batch)
    return correct / len(texts)


@torch.no_grad()
def attack_success_rate(
    model: LoRAModel,
    texts: list[str],
    labels: list[int],
    attack: Attack,
    device="cpu",
    batch_size: int = 64,
) -> float | None:
    """ASR: fraction of non-target inputs that, once triggered, are predicted target.

    Returns None when the attack defines no trigger/target (e.g. benign runs), so the
    caller can omit ASR rather than log a meaningless number.
    """
    target = attack.target_label
    if target is None:
        return None
    triggered = [
        attack.apply_trigger(t) for t, y in zip(texts, labels, strict=False) if y != target
    ]
    if not triggered:
        return None
    model.model.eval()
    hits = 0
    for batch in _batched(triggered, batch_size):
        preds = model.logits(batch, device).argmax(dim=-1).cpu()
        hits += (preds == target).sum().item()
    return hits / len(triggered)


@torch.no_grad()
def backdoor_classification_metrics(
    model: LoRAModel,
    texts: list[str],
    labels: list[int],
    attack: Attack,
    device="cpu",
    batch_size: int = 64,
) -> dict[str, float | int | None]:
    """Paired clean/triggered outcomes, with clean loss for connectivity studies.

    Triggered outcomes use non-target examples and retain their original labels.
    The clean-correct subset distinguishes trigger errors from ordinary task errors.
    """
    if len(texts) != len(labels) or not texts:
        raise ValueError("evaluation needs equally sized, nonempty texts and labels")
    model.model.eval()
    clean_predictions: list[int] = []
    clean_loss_sum = 0.0
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        target = torch.tensor(labels[start : start + batch_size], device=device)
        logits = model.logits(batch_texts, device).float()
        clean_loss_sum += float(F.cross_entropy(logits, target, reduction="sum"))
        clean_predictions.extend(logits.argmax(dim=-1).cpu().tolist())
    clean_correct = [pred == label for pred, label in zip(clean_predictions, labels, strict=True)]
    result: dict[str, float | int | None] = {
        "clean_loss": clean_loss_sum / len(texts),
        "clean_accuracy": sum(clean_correct) / len(texts),
        "clean_count": len(texts),
        "asr": None,
        "triggered_correctness": None,
        "triggered_correctness_clean_correct": None,
        "triggered_count": 0,
        "triggered_clean_correct_count": 0,
    }
    target_label = attack.target_label
    if target_label is None:
        return result
    selected = [i for i, label in enumerate(labels) if label != target_label]
    if not selected:
        return result
    triggered_predictions: list[int] = []
    for start in range(0, len(selected), batch_size):
        indices = selected[start : start + batch_size]
        triggered = [attack.apply_trigger(texts[i]) for i in indices]
        triggered_predictions.extend(model.logits(triggered, device).argmax(dim=-1).cpu().tolist())
    hits = sum(pred == target_label for pred in triggered_predictions)
    correct = sum(pred == labels[i] for pred, i in zip(triggered_predictions, selected, strict=True))
    conditional = [
        (pred, i)
        for pred, i in zip(triggered_predictions, selected, strict=True)
        if clean_correct[i]
    ]
    result.update(
        asr=hits / len(selected),
        triggered_correctness=correct / len(selected),
        triggered_correctness_clean_correct=(
            sum(pred == labels[i] for pred, i in conditional) / len(conditional)
            if conditional else None
        ),
        triggered_count=len(selected),
        triggered_clean_correct_count=len(conditional),
    )
    return result
