"""X4 trigger-based text backdoor with append and in-place replacement modes.

ASR is measured by triggering held-out non-target inputs and counting target
predictions. Append remains the baseline default; replacement preserves client
shard size for paired recovery experiments.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

from .base import ATTACKS, AttackConfig, BaseAttack


@dataclass
class X4Config(AttackConfig):
    name: str = "x4_backdoor"
    trigger: str = "cf"
    target_label: int = 0
    poison_fraction: float = 0.5
    poison_mode: str = "append"


@ATTACKS.register("x4_backdoor")
class X4BackdoorAttack(BaseAttack):
    HOOKS = frozenset({"poison_dataset"})  # data-space only; objective and factors intact

    def __init__(self, config: X4Config) -> None:
        super().__init__(config)
        if not 0.0 <= config.poison_fraction <= 1.0:
            raise ValueError("poison_fraction must be in [0, 1]")
        if not config.trigger.strip():
            raise ValueError("trigger must be non-empty")
        if config.poison_mode not in ("append", "replace"):
            raise ValueError("poison_mode must be 'append' or 'replace'")
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
        if self.cfg.poison_mode == "replace":
            candidates = [i for i, label in enumerate(labels) if label != self.cfg.target_label]
            k = int(self.cfg.poison_fraction * len(candidates))
            if k <= 0:
                return texts, labels
            # Keep the optimizer's later minibatch shuffle stream identical in the
            # clean and poisoned arms. Only the article content and label differ.
            selected = copy.deepcopy(rng).choice(candidates, size=k, replace=False)
            for i in selected:
                texts[int(i)] = self.apply_trigger(texts[int(i)])
                labels[int(i)] = self.cfg.target_label
            return texts, labels
        n = len(texts)
        k = int(self.cfg.poison_fraction * n)
        if k <= 0:
            return texts, labels
        idx = rng.choice(n, size=k, replace=False)
        for i in idx:
            texts.append(self.apply_trigger(texts[int(i)]))
            labels.append(self.cfg.target_label)
        return texts, labels
