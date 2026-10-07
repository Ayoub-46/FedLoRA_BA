"""Within-round top-k filtering of client factors against the current broadcast.

Two score modes share one deterministic selection and reweighting rule:

``cosine``
    Reuse the representation-sensitive score from ``factor_broadcast_cosine``:
    one minus the equal mean of valid A/B block Frobenius cosines.

``euclidean``
    For each module-factor block, compute RMS displacement from the broadcast, divide
    by that block's cross-client median RMS displacement, average A/B within a module,
    then average modules equally.  RMS removes tensor-size effects; within-block median
    scaling removes layer and A/B scale effects without requiring honest calibration.

The highest-scoring ``num_reject`` reports receive weight zero. Ties use a deterministic
seed-and-round permutation that is independent of client identity, and the original
data-proportional weights are renormalized over survivors. The defense has no access to
attack or coalition labels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from ..fl.types import AdapterDict, ClientUpdate
from .base import ACCEPT, DEFENSES, REJECT, BaseDefense, DefenseConfig, DefenseResult
from .factor_broadcast_cosine import factor_broadcast_scores

COSINE = "cosine"
EUCLIDEAN = "euclidean"
SCORE_MODES = (COSINE, EUCLIDEAN)


@dataclass
class FactorBroadcastTopKConfig(DefenseConfig):
    name: str = "factor_broadcast_topk"
    score_mode: str = COSINE
    num_reject: int = 2
    euclidean_epsilon: float = 1e-12
    tie_break_seed: int = 0

    def __post_init__(self) -> None:
        if self.score_mode not in SCORE_MODES:
            raise ValueError(f"score_mode must be one of {SCORE_MODES}, got {self.score_mode!r}")
        if self.num_reject < 0:
            raise ValueError("num_reject must be non-negative")
        if not math.isfinite(self.euclidean_epsilon) or self.euclidean_epsilon <= 0.0:
            raise ValueError("euclidean_epsilon must be finite and positive")


def _validate_layer_order(updates: list[ClientUpdate], broadcast: AdapterDict) -> list[str]:
    layer_names = list(broadcast)
    for update in updates:
        if list(update.adapters) != layer_names:
            raise ValueError("client and broadcast adapters must share layer set and order")
    return layer_names


def factor_broadcast_euclidean_scores(
    updates: list[ClientUpdate],
    broadcast: AdapterDict,
    *,
    epsilon: float = 1e-12,
) -> tuple[list[float], int, list[float]]:
    """Return median-scaled, equal-module RMS displacement scores.

    For every module and factor, ``r_i = ||F_i - F_g||_F / sqrt(numel(F))``.
    Each block is normalized by ``max(median_i(r_i), epsilon)``. The final client
    score is the equal mean of A and B within each module and then across modules.
    Broadcast-zero blocks remain valid because Euclidean displacement is defined there.
    """
    if not updates:
        return [], 0, []
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")

    layer_names = _validate_layer_order(updates, broadcast)
    if not layer_names:
        return [0.0] * len(updates), 0, []

    normalized_by_client: list[list[float]] = [[] for _ in updates]
    block_scales: list[float] = []
    for layer in layer_names:
        for factor in ("A", "B"):
            reference = getattr(broadcast[layer], factor).detach().to(dtype=torch.float64)
            rms_values = []
            for update in updates:
                submitted = getattr(update.adapters[layer], factor).detach().to(dtype=torch.float64)
                displacement = torch.linalg.vector_norm(submitted - reference)
                rms_values.append(float(displacement) / math.sqrt(reference.numel()))
            scale = max(float(np.median(np.asarray(rms_values, dtype=np.float64))), epsilon)
            block_scales.append(scale)
            for values, rms in zip(normalized_by_client, rms_values, strict=True):
                values.append(rms / scale)

    scores = [math.fsum(values) / len(values) for values in normalized_by_client]
    return scores, 2 * len(layer_names), block_scales


@DEFENSES.register("factor_broadcast_topk")
class FactorBroadcastTopKDefense(BaseDefense):
    def __init__(self, config: FactorBroadcastTopKConfig) -> None:
        super().__init__(config)
        config.__post_init__()
        self.cfg = config

    def apply(self, updates, context) -> DefenseResult:
        p = np.asarray(context.base_weights, dtype=np.float64)
        if len(p) != len(updates):
            raise ValueError(f"expected {len(updates)} base weights, got {len(p)}")
        if context.broadcast_adapters is None:
            raise ValueError("factor_broadcast_topk requires the actual broadcast adapters")
        if self.cfg.num_reject >= len(updates):
            raise ValueError(
                f"num_reject={self.cfg.num_reject} must be smaller than "
                f"the {len(updates)} submitted reports"
            )

        block_scales: list[float] = []
        if self.cfg.score_mode == COSINE:
            scores, valid_blocks = factor_broadcast_scores(
                updates, context.broadcast_adapters
            )
            score_definition = "one_minus_equal_mean_valid_factor_block_frobenius_cosine"
        else:
            scores, valid_blocks, block_scales = factor_broadcast_euclidean_scores(
                updates,
                context.broadcast_adapters,
                epsilon=self.cfg.euclidean_epsilon,
            )
            score_definition = (
                "equal_mean_module_and_factor_median_scaled_rms_broadcast_displacement"
            )

        tie_rng = np.random.default_rng(
            [self.cfg.tie_break_seed, context.round_idx, 0x70F2]
        )
        tie_permutation = tie_rng.permutation(len(updates))
        tie_priority = {
            int(update_index): priority
            for priority, update_index in enumerate(tie_permutation)
        }
        ranked = sorted(
            range(len(updates)),
            key=lambda index: (-scores[index], tie_priority[index], updates[index].client_id),
        )
        rejected_indices = set(ranked[: self.cfg.num_reject])
        rejected = np.asarray(
            [index in rejected_indices for index in range(len(updates))], dtype=bool
        )
        accepted_mass = float(p[~rejected].sum())
        if accepted_mass <= 0.0:
            raise RuntimeError("top-k survivors have zero total protocol weight")
        q = np.where(rejected, 0.0, p / accepted_mass)

        ranks = {index: rank for rank, index in enumerate(ranked, start=1)}
        stats: dict[int, dict] = {}
        verdicts: dict[int, str] = {}
        for index, update in enumerate(updates):
            stats[update.client_id] = {
                "factor_broadcast_score": float(scores[index]),
                "score_mode": self.cfg.score_mode,
                "anomaly_rank": int(ranks[index]),
                "num_reject": int(self.cfg.num_reject),
                "valid_broadcast_blocks": int(valid_blocks),
                "base_weight": float(p[index]),
                "output_weight": float(q[index]),
            }
            verdicts[update.client_id] = REJECT if rejected[index] else ACCEPT

        round_stats = {
            "filter_definition": score_definition,
            "score_mode": self.cfg.score_mode,
            "selection_rule": "highest_score_then_seeded_round_permutation",
            "tie_break_seed": int(self.cfg.tie_break_seed),
            "num_reject": int(self.cfg.num_reject),
            "rejected_count": int(rejected.sum()),
            "accepted_count": int((~rejected).sum()),
            "accepted_weight_mass": accepted_mass,
            "valid_broadcast_blocks": int(valid_blocks),
            "diagnostic_dtype": "float64",
        }
        if block_scales:
            round_stats.update(
                {
                    "euclidean_raw_distance": "frobenius_rms_per_factor_block",
                    "euclidean_block_scale": "cross_client_median_rms",
                    "euclidean_combination": "equal_A_B_within_equal_modules",
                    "euclidean_epsilon": float(self.cfg.euclidean_epsilon),
                    "euclidean_scale_min": float(min(block_scales)),
                    "euclidean_scale_median": float(np.median(block_scales)),
                    "euclidean_scale_max": float(max(block_scales)),
                }
            )

        return DefenseResult(
            weights=q,
            per_client_stats=stats,
            verdicts=verdicts,
            round_stats=round_stats,
        )
