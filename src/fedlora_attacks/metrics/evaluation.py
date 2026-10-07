"""Outcome metrics (plan §3): main-task accuracy and attack success rate (ASR)."""

from __future__ import annotations

import torch

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
