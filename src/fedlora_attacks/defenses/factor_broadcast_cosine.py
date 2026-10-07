"""Cosine filter on transmitted LoRA factors relative to the broadcast factors.

This defense is intentionally representation-sensitive.  For every client it computes
one Frobenius cosine for each valid A and B factor block against the actual adapters
broadcast at the start of the round, then assigns equal weight to every valid block:

    score_i = 1 - mean_block cosine(X_i, X_broadcast).

All inner products and norms are accumulated in float64 and the resulting cosine is
clamped to [-1, 1].  A block whose broadcast norm is zero is omitted for every client;
if the broadcast is nonzero but the client's block is zero its cosine is zero.  When no
broadcast blocks are valid, every report is accepted and the event is logged.

The fixed threshold is calibrated outside this class.  ``action='monitor'`` records the
same scores and counterfactual decisions while preserving the unfiltered trajectory;
``action='reject'`` applies the strict rule ``score > threshold`` and renormalizes the
original data-proportional weights over survivors.  The implementation has no access to
coalition labels, attack identities, or pre-transformation reports.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from ..fl.types import AdapterDict, ClientUpdate
from .base import ACCEPT, DEFENSES, REJECT, BaseDefense, DefenseConfig, DefenseResult

MONITOR = "monitor"
REJECT_ACTION = "reject"
ACTIONS = (MONITOR, REJECT_ACTION)


@dataclass
class FactorBroadcastCosineConfig(DefenseConfig):
    name: str = "factor_broadcast_cosine"
    threshold: float = 2.0
    action: str = REJECT_ACTION

    def __post_init__(self) -> None:
        if self.action not in ACTIONS:
            raise ValueError(f"action must be one of {ACTIONS}, got {self.action!r}")
        if not math.isfinite(self.threshold):
            raise ValueError("threshold must be finite")


def _block_cosine(client: torch.Tensor, broadcast: torch.Tensor) -> float | None:
    """Return a float64 Frobenius cosine, or None for a zero broadcast block."""
    reference = broadcast.detach().to(dtype=torch.float64)
    reference_norm = torch.linalg.vector_norm(reference)
    if float(reference_norm) == 0.0:
        return None

    submitted = client.detach().to(dtype=torch.float64)
    submitted_norm = torch.linalg.vector_norm(submitted)
    if float(submitted_norm) == 0.0:
        return 0.0

    inner = torch.sum(submitted * reference)
    cosine = float(inner / (submitted_norm * reference_norm))
    # A finite dot product can exceed the mathematical range by a few ulps.
    return float(np.clip(cosine, -1.0, 1.0))


def factor_broadcast_scores(
    updates: list[ClientUpdate], broadcast: AdapterDict
) -> tuple[list[float], int]:
    """Score every transmitted report against one actual broadcast adapter state.

    The valid block set is defined only by broadcast norms and is therefore identical
    for every client in a round.  Layer and factor blocks all receive equal weight.
    """
    if not updates:
        return [], 0

    layer_names = list(broadcast)
    for update in updates:
        if list(update.adapters) != layer_names:
            raise ValueError("client and broadcast adapters must share layer set and order")

    valid: list[tuple[str, str]] = []
    for layer in layer_names:
        state = broadcast[layer]
        if float(torch.linalg.vector_norm(state.A.detach().to(dtype=torch.float64))) != 0.0:
            valid.append((layer, "A"))
        if float(torch.linalg.vector_norm(state.B.detach().to(dtype=torch.float64))) != 0.0:
            valid.append((layer, "B"))

    if not valid:
        return [0.0] * len(updates), 0

    scores: list[float] = []
    for update in updates:
        cosines: list[float] = []
        for layer, factor in valid:
            client_tensor = getattr(update.adapters[layer], factor)
            broadcast_tensor = getattr(broadcast[layer], factor)
            cosine = _block_cosine(client_tensor, broadcast_tensor)
            # This cannot be None because the valid set was defined from this broadcast.
            if cosine is None:  # pragma: no cover - guards a concurrent tensor mutation
                raise RuntimeError("broadcast validity changed while scoring reports")
            cosines.append(cosine)
        scores.append(1.0 - math.fsum(cosines) / len(cosines))
    return scores, len(valid)


@DEFENSES.register("factor_broadcast_cosine")
class FactorBroadcastCosineDefense(BaseDefense):
    def __init__(self, config: FactorBroadcastCosineConfig) -> None:
        super().__init__(config)
        config.__post_init__()
        self.cfg = config

    def apply(self, updates, context) -> DefenseResult:
        p = np.asarray(context.base_weights, dtype=np.float64)
        if len(p) != len(updates):
            raise ValueError(f"expected {len(updates)} base weights, got {len(p)}")
        if context.broadcast_adapters is None:
            raise ValueError("factor_broadcast_cosine requires the actual broadcast adapters")

        scores, valid_blocks = factor_broadcast_scores(updates, context.broadcast_adapters)
        if valid_blocks == 0:
            rejected = np.zeros(len(updates), dtype=bool)
        else:
            # Strict by specification: equality with the frozen quantile is accepted.
            rejected = np.asarray([score > self.cfg.threshold for score in scores], dtype=bool)

        counterfactual_accepted_mass = float(p[~rejected].sum()) if len(p) else 0.0
        if self.cfg.action == MONITOR:
            q = p.copy()
            applied_rejected = np.zeros(len(updates), dtype=bool)
        else:
            applied_rejected = rejected
            if not applied_rejected.any():
                # Preserve exact baseline arithmetic when everybody is accepted.
                q = p.copy()
            elif counterfactual_accepted_mass > 0.0:
                q = np.where(applied_rejected, 0.0, p / counterfactual_accepted_mass)
            else:
                q = np.zeros_like(p)

        stats: dict[int, dict] = {}
        verdicts: dict[int, str] = {}
        for index, update in enumerate(updates):
            stats[update.client_id] = {
                "factor_broadcast_score": float(scores[index]),
                "threshold": float(self.cfg.threshold),
                "would_reject": bool(rejected[index]),
                "valid_broadcast_blocks": int(valid_blocks),
                "base_weight": float(p[index]),
                "output_weight": float(q[index]),
            }
            verdicts[update.client_id] = REJECT if applied_rejected[index] else ACCEPT

        accepted_count = int((~applied_rejected).sum())
        accepted_mass = float(p[~applied_rejected].sum()) if len(p) else 0.0
        return DefenseResult(
            weights=q,
            per_client_stats=stats,
            verdicts=verdicts,
            round_stats={
                "filter_definition": "one_minus_equal_mean_factor_block_frobenius_cosine",
                "threshold": float(self.cfg.threshold),
                "strict_rejection": True,
                "action": self.cfg.action,
                "valid_broadcast_blocks": int(valid_blocks),
                "no_valid_broadcast_blocks": valid_blocks == 0,
                "counterfactual_rejected_count": int(rejected.sum()),
                "accepted_count": accepted_count,
                "accepted_weight_mass": accepted_mass,
                "counterfactual_accepted_weight_mass": counterfactual_accepted_mass,
                "skipped_aggregation": self.cfg.action == REJECT_ACTION and accepted_count == 0,
                "diagnostic_dtype": "float64",
            },
        )
