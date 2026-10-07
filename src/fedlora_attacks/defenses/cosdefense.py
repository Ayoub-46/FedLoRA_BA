"""CosDefense: cosine between each client's `dW_i` and the consensus direction. END-TO-END.

    cos_i = <dW_i, dW_bar> / (||dW_i||_F ||dW_bar||_F),    dW_bar = sum_j p_j dW_j

The direction-based member of the defence family, and the one the paper's main move is
aimed at. `norm_bound` scores magnitude and `multi_krum` scores pairwise distance;
neither asks whether a client is pulling the model somewhere else. This does.

**Why it belongs in the matrix even though it is also blind to Channel A.** Being blind
is not the interesting property — every `dW`-measurable statistic is blind to a gauge, by
construction. What makes CosDefense worth running is what it is blind to *besides*:

* **Scale-invariant by construction**, so it cannot see the `scaling` mode at all, nor
  norm-envelope inflation. A coalition may multiply `dW_i` by any `c > 0` and `cos_i` is
  unchanged to the last bit. Where `norm_bound` is blind to a re-expression that fixes
  `||dW_i||`, this is blind to one that changes it — the two fail on opposite axes, which
  is exactly why a matrix is the right artifact.
* **Sensitive to column steering.** Rotating `col(dW_i)` away from the consensus is the
  one thing it does see, so it is the natural counter to the envelope attack and the
  reason the paper is more credible for including it.

**The reference direction, stated precisely.** The server has no oracle for "the global
direction" at scoring time — the aggregate does not exist until after the defence runs.
What it has is the round's own traffic, so the reference is the provisional consensus
`dW_bar = sum_j p_j dW_j` under the protocol weights: the update the server WOULD
broadcast with no defence in force. That is the standard construction (FoolsGold,
CosDefense) and it keeps every statistic a function of the `dW`s alone. Client `i` is
included in the mean it is scored against; excluding it would make the reference differ
per client and the scores non-comparable.

**Factored throughout.** `dW_bar` is never materialised: it is carried as STACKED factors
`[sqrt(p_j) A_j]` and `[sqrt(p_j) B_j]` whose product is exactly `sum_j p_j dW_j`, and the
inner products come from `r x Nr` cross-Grams. Rank up to `N*r`, so the stacking is also
the honest representation of what factor averaging is not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from ..probes.geometry import cross_gram, deltaw_frobenius_norm
from .base import ACCEPT, DEFENSES, REJECT, BaseDefense, DefenseConfig, DefenseResult

FIXED = "fixed"
MEDIAN = "median"
THRESHOLD_MODES = (FIXED, MEDIAN)

REJECT_ACTION = "reject"
REWEIGHT_ACTION = "reweight"
ACTIONS = (REJECT_ACTION, REWEIGHT_ACTION)


@dataclass
class CosDefenseConfig(DefenseConfig):
    name: str = "cosdefense"
    threshold_mode: str = MEDIAN
    """How the cutoff is chosen.

    ``median`` sets it from the round's own traffic as a deployed system would:
    `tau_t = median_i cos_i - gamma * MAD_i`. ``fixed`` uses ``min_cosine`` outright,
    which needs the benign cosine scale known in advance and is kept for calibration
    runs and for the E-B pairing where a moving threshold would confound the comparison.
    """
    min_cosine: float = 0.0
    gamma: float = 1.5
    action: str = REJECT_ACTION
    """``reject`` sets `q_i = 0` below the cutoff. ``reweight`` uses
    `q_i = p_i * max(0, cos_i) / max_j max(0, cos_j)`, a soft version that keeps a
    marginal client at reduced weight rather than discarding its data outright."""


def _stacked_consensus(updates, weights, layer):
    """`(A_stack, B_stack)` with `A_stack @ B_stack = sum_j p_j dW_j` for one layer.

    `sqrt(p_j)` on both sides rather than `p_j` on one: it keeps the two factors on a
    comparable scale, which matters only for conditioning, since the product is the same
    either way.
    """
    a_parts, b_parts = [], []
    for u, p in zip(updates, weights, strict=True):
        s = u.adapters[layer]
        root = math.sqrt(max(float(p), 0.0))
        a_parts.append(root * s.A.detach().to(torch.float64))
        b_parts.append(root * s.B.detach().to(torch.float64))
    return torch.cat(a_parts, dim=1), torch.cat(b_parts, dim=0)


def consensus_cosines(updates, weights) -> tuple[list[float], list[float], float]:
    """`(cos_i, ||dW_i||_F, ||dW_bar||_F)`, network-wide, from factors only.

    Per-layer inner products and squared norms ADD, because the network-wide `dW` is the
    direct sum of the per-layer ones — so the network cosine is formed from summed
    inner products and summed squared norms, never by averaging per-layer cosines (which
    would weight a small layer as heavily as a large one).

    Split out from ``apply`` so E-B can log it and assert it is unchanged under a gauge.
    """
    layers = list(updates[0].adapters.keys())
    n = len(updates)
    inner = np.zeros(n, dtype=np.float64)
    sq = np.zeros(n, dtype=np.float64)
    bar_sq = 0.0
    for layer in layers:
        a_bar, b_bar = _stacked_consensus(updates, weights, layer)
        bar_sq += float(torch.trace((a_bar.T @ a_bar) @ (b_bar @ b_bar.T)))
        for i, u in enumerate(updates):
            s = u.adapters[layer]
            inner[i] += float(torch.trace(cross_gram(s.A, s.B, a_bar, b_bar)))
            sq[i] += deltaw_frobenius_norm(s.A, s.B) ** 2
    norms = np.sqrt(np.clip(sq, 0.0, None))
    bar_norm = math.sqrt(max(bar_sq, 0.0))
    cos = np.full(n, math.nan, dtype=np.float64)
    if bar_norm > 0:
        ok = norms > 0
        cos[ok] = inner[ok] / (norms[ok] * bar_norm)
    return [float(c) for c in cos], [float(v) for v in norms], bar_norm


def _median_abs_deviation(x: np.ndarray) -> float:
    return float(np.median(np.abs(x - np.median(x))))


@DEFENSES.register("cosdefense")
class CosDefense(BaseDefense):
    """Direction-based filtering on the client-vs-consensus cosine (plan v3 defence matrix)."""

    def __init__(self, config: CosDefenseConfig) -> None:
        super().__init__(config)
        if config.threshold_mode not in THRESHOLD_MODES:
            raise ValueError(
                f"unknown threshold_mode {config.threshold_mode!r}; expected {THRESHOLD_MODES}"
            )
        if config.action not in ACTIONS:
            raise ValueError(f"unknown action {config.action!r}; expected {ACTIONS}")
        self.cfg = config

    def apply(self, updates, context) -> DefenseResult:
        p = np.asarray(context.base_weights, dtype=np.float64)
        if not updates:
            return DefenseResult(weights=p, per_client_stats={}, verdicts={})
        cos, norms, bar_norm = consensus_cosines(updates, list(p))
        c = np.asarray(cos, dtype=np.float64)
        finite = c[np.isfinite(c)]

        if self.cfg.threshold_mode == MEDIAN and finite.size:
            centre = float(np.median(finite))
            tau = centre - self.cfg.gamma * _median_abs_deviation(finite)
        else:
            centre = float(np.median(finite)) if finite.size else math.nan
            tau = float(self.cfg.min_cosine)

        if self.cfg.action == REWEIGHT_ACTION:
            positive = np.clip(np.nan_to_num(c, nan=0.0), 0.0, None)
            top = float(positive.max()) if positive.size else 0.0
            factor = positive / top if top > 0 else np.zeros_like(positive)
        else:
            # a NaN cosine means a zero-norm update: nothing to point anywhere, so it is
            # kept at its protocol weight rather than rejected for being degenerate
            factor = np.where(np.isnan(c) | (c >= tau), 1.0, 0.0)

        q = p * factor
        stats, verdicts = {}, {}
        for i, u in enumerate(updates):
            stats[u.client_id] = {
                "cosine_to_consensus": cos[i],
                "deltaw_frobenius": norms[i],
                "consensus_frobenius": bar_norm,
                "threshold": tau,
                "median_cosine": centre,
                "weight_factor": float(factor[i]),
            }
            verdicts[u.client_id] = ACCEPT if factor[i] >= 1.0 else REJECT
        return DefenseResult(
            weights=q,
            per_client_stats=stats,
            verdicts=verdicts,
            round_stats={
                "threshold": tau,
                "median_cosine": centre,
                "consensus_frobenius": bar_norm,
                "threshold_mode": self.cfg.threshold_mode,
                "action": self.cfg.action,
            },
        )
