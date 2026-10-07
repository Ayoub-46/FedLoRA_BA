"""Multi-Krum on dW-distances: the careful implementer's port. END-TO-END.

The distance-based robust aggregator, ported to FedLoRA the RIGHT way — scores computed
on the effective updates `dW_i`, not on the transmitted factors. Score of client i is
the sum of squared distances to its `n - f - 2` nearest neighbours; the `m` lowest
scores are selected and their weights renormalised.

**This is the trap cell of the defence matrix, and it is where deployed systems sit.** A
careful implementer computes a sound statistic on `dW` and still averages factors for
communication efficiency, because that is what FedIT does. Both halves are defensible in
isolation. Together they leave Channel A wide open: every pairwise distance
`||dW_i - dW_j||_F` is a function of the `dW`s alone, so a coalition re-expressing its
factors leaves the ENTIRE score matrix — and therefore the selected set — bit-identical,
while the aggregate it feeds moves.

Distances come from ``pairwise_deltaw_distances`` per layer, combined as
`sqrt(sum_l d_l^2)` for the network-wide distance, all factored — `dW` is never
materialized. Cost O(L N^2 (n+m) r^2), the last row of plan v3 §6b's table.

**Krum is also Channel B's negative control (E-F).** `||dW - (-dW)||_F = 2||dW||_F` is
about as loud as an anomaly gets, so this defence SHOULD catch isospectral negation
where a summary-statistic detector cannot. The paper is more credible for reporting that
its framework does not condemn everything.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..probes.geometry import pairwise_deltaw_distances
from .base import ACCEPT, DEFENSES, REJECT, BaseDefense, DefenseConfig, DefenseResult


@dataclass
class MultiKrumConfig(DefenseConfig):
    name: str = "multi_krum"
    num_selected: int = 3  # m: how many clients survive selection
    num_byzantine: int = 1  # f: assumed adversaries, sets the neighbourhood n - f - 2


def squared_deltaw_distance_matrix(updates) -> torch.Tensor:
    """N x N matrix of ||dW_i - dW_j||_F^2 summed over layers, from factors only.

    Per-layer squared distances add, because the network-wide dW is the direct sum of
    the per-layer ones. Kept separate from the scoring so E-B can log the matrix itself
    and assert it is unchanged under a gauge.
    """
    layers = list(updates[0].adapters.keys())
    n = len(updates)
    total = torch.zeros(n, n, dtype=torch.float64)
    for layer in layers:
        d = pairwise_deltaw_distances([u.adapters[layer] for u in updates])
        total += d.pow(2)
    return total


def krum_scores(sq_distances: torch.Tensor, num_byzantine: int) -> tuple[torch.Tensor, int]:
    """(score per client, the neighbourhood size actually used).

    Krum's score is the sum of the squared distances to the `k = n - f - 2` closest other
    clients. Krum's guarantee needs `n > 2f + 2`; below that the neighbourhood would be
    empty, so k is clamped to at least 1 and the effective value is REPORTED rather than
    silently assumed — a run outside Krum's regime should be visible in the log, not
    inferred from the config.
    """
    n = sq_distances.shape[0]
    k = max(1, min(n - num_byzantine - 2, n - 1))
    scores = torch.empty(n, dtype=torch.float64)
    for i in range(n):
        others = torch.cat([sq_distances[i, :i], sq_distances[i, i + 1 :]])
        scores[i] = torch.sort(others).values[:k].sum()
    return scores, k


@DEFENSES.register("multi_krum")
class MultiKrumDefense(BaseDefense):
    """Selection on dW-distances, then renormalisation (plan v3 E-A / E-B / E-F)."""

    def __init__(self, config: MultiKrumConfig) -> None:
        super().__init__(config)
        self.cfg = config

    def apply(self, updates, context) -> DefenseResult:
        n = len(updates)
        sq = squared_deltaw_distance_matrix(updates)
        scores, k = krum_scores(sq, self.cfg.num_byzantine)

        m = max(1, min(self.cfg.num_selected, n))
        # Ties broken by index so the selected set is deterministic: E-B's claim is that
        # it is IDENTICAL across a paired run, which an unstable sort would break for
        # reasons unrelated to the attack.
        order = sorted(range(n), key=lambda i: (scores[i].item(), i))
        selected = set(order[:m])

        selected_mass = sum(p for i, p in enumerate(context.base_weights) if i in selected)
        weights, stats, verdicts = [], {}, {}
        for i, (u, p) in enumerate(zip(updates, context.base_weights, strict=True)):
            keep = i in selected
            weights.append(p / selected_mass if keep and selected_mass > 0 else 0.0)
            stats[u.client_id] = {
                "krum_score": scores[i].item(),
                "neighbourhood": k,
                "selected": keep,
                "mean_sq_distance": (sq[i].sum().item() / max(n - 1, 1)),
            }
            verdicts[u.client_id] = ACCEPT if keep else REJECT
        return DefenseResult(
            weights=np.asarray(weights, dtype=np.float64),
            per_client_stats=stats,
            verdicts=verdicts,
        )
