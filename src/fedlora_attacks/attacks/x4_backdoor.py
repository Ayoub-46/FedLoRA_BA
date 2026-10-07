"""X4 baseline: trigger-based text backdoor (BadNets-style). END-TO-END.

A configurable trigger token is inserted and the label flipped to a target class on
a fraction of a malicious client's local data (clean data retained, poisoned copies
appended). ASR is measured by inserting the trigger into clean test inputs and
checking the model predicts the target label. Serves as the persistence reference
(plan §3, X4) that E2's low-rank steering is compared against.
"""

from __future__ import annotations

from dataclasses import dataclass

from .base import ATTACKS, AttackConfig, BaseAttack


@dataclass
class X4Config(AttackConfig):
    name: str = "x4_backdoor"
    trigger: str = "cf"
    target_label: int = 0
    poison_fraction: float = 0.5


@ATTACKS.register("x4_backdoor")
class X4BackdoorAttack(BaseAttack):
    HOOKS = frozenset({"poison_dataset"})  # data-space only; objective and factors intact

    def __init__(self, config: X4Config) -> None:
        super().__init__(config)
        if not 0.0 <= config.poison_fraction <= 1.0:
            raise ValueError("poison_fraction must be in [0, 1]")
        if not config.trigger.strip():
            raise ValueError("trigger must be non-empty")
        self.cfg: X4Config = config

    def apply_trigger(self, text: str) -> str:
        return f"{self.cfg.trigger} {text}"

    @property
    def target_label(self) -> int | None:
        return self.cfg.target_label

    def extra_vocab(self) -> list[str]:
        return [self.cfg.trigger]

    def poison_dataset(self, texts, labels, client_id, round_idx, rng):
        if not self.is_active(client_id, round_idx):
            return texts, labels
        texts = list(texts)
        labels = list(labels)
        n = len(texts)
        k = int(self.cfg.poison_fraction * n)
        if k <= 0:
            return texts, labels
        idx = rng.choice(n, size=k, replace=False)
        for i in idx:
            texts.append(self.apply_trigger(texts[int(i)]))
            labels.append(self.cfg.target_label)
        return texts, labels
