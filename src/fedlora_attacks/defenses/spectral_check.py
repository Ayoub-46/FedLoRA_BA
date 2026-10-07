"""Spectral outlier test on `sigma(dW_i)`: the statistic the other three cannot see.

Replaces the `per_layer_outlier` stub, which was a config selecting a hook that raised.

**Why the matrix needs this row.** `norm_bound` reads `||dW_i||_F`, `multi_krum` reads
pairwise distances, `cosdefense` reads a direction. All three are invariant to how a
client's update distributes its energy ACROSS singular directions, so spectral shaping is
invisible to every one of them: at fixed `||dW||_F` the nuclear norm ranges over
`[||dW||_F, sqrt(r) ||dW||_F]`, and since the balanced factorisation has
`||A||_F = ||B||_F = sqrt(||dW||_*)`, that is up to `r^(1/4) ~= 1.68` at r=8 in factor
mass — bought at zero cost in every statistic the other three compute.

Including it is what stops the paper reading as a survey of defences chosen because they
fail. It is the statistic that *catches* the mechanism, it is cheap
(``factored_spectral_summary`` already computes it), and it is still a function of `dW_i`
alone — so it remains exactly as blind to Channel A as the rest, which is the point: the
blindness is a property of the `dW`-measurable family, not of any one member's
carelessness.

**Default statistic: the nuclear/Frobenius ratio.** Scale-free (so it is not a second
norm bound wearing a different name) and it is precisely the shaping knob. The
alternatives — effective rank, stable rank, condition number, `sigma_max/sigma_min` — are
selectable, and all are reported per client every round whichever one is scored, so a
sweep never has to be re-run to look at a different one.

**Robust, not Gaussian.** The cutoff is a median/MAD z-score across the round's clients,
because with `N = 20` and a coalition inside the population a mean/std rule would let the
coalition move the very threshold it is judged by. MAD is scaled by 1.4826 so the score
reads as a standard deviation for normal data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..probes.geometry import factored_spectral_summary, spectral_probe
from .base import ACCEPT, DEFENSES, REJECT, BaseDefense, DefenseConfig, DefenseResult

#: MAD -> sigma for normally distributed data, so `z_threshold` reads conventionally.
MAD_TO_SIGMA = 1.4826

NUCLEAR_RATIO = "nuclear_ratio"
STATISTICS = (
    NUCLEAR_RATIO,
    "effective_rank",
    "stable_rank",
    "condition_number",
    "sigma_max",
)

TWO_SIDED = "two_sided"
HIGH = "high"
LOW = "low"
DIRECTIONS = (TWO_SIDED, HIGH, LOW)


@dataclass
class SpectralCheckConfig(DefenseConfig):
    name: str = "spectral_check"
    statistic: str = NUCLEAR_RATIO
    z_threshold: float = 3.0
    direction: str = TWO_SIDED
    """Which tail is anomalous.

    ``two_sided`` by default because spectral shaping can go either way: flattening the
    spectrum raises the nuclear ratio, concentrating it lowers it, and both buy factor
    mass or hide it. Choosing one tail in advance would be choosing which attack to see.
    """
    per_layer: bool = False
    """``False`` scores the network-wide spectrum (all layers' singular values pooled);
    ``True`` scores each layer and takes a client's WORST layer. Both are `dW`-measurable
    and therefore both blind to Channel A; running both is what stops the result reading
    as an artifact of pooling."""


def _summarize(values: np.ndarray) -> dict[str, float]:
    """The scale-free spectral descriptors, from a singular-value vector."""
    s = np.asarray(values, dtype=np.float64)
    s = s[s > 0]
    if s.size == 0:
        return dict.fromkeys(STATISTICS, math.nan)
    fro = float(np.sqrt(np.sum(s**2)))
    nuc = float(np.sum(s))
    p = s / nuc
    return {
        NUCLEAR_RATIO: nuc / fro if fro > 0 else math.nan,
        "effective_rank": float(np.exp(-np.sum(p * np.log(p)))),
        "stable_rank": float(np.sum(s**2) / s[0] ** 2),
        "condition_number": float(s[0] / s[-1]) if s[-1] > 0 else math.inf,
        "sigma_max": float(s[0]),
    }


def client_spectral_stats(update, per_layer: bool = False) -> dict:
    """Every spectral descriptor for one client, plus the per-layer breakdown.

    Singular values come from ``spectral_probe`` — thin QR plus one `r x r` core SVD, in
    fp64 — so `dW` is never materialised. ``factored_spectral_summary`` supplies the
    project's standard per-layer block alongside, unchanged, so this defence and the
    probes report the same numbers by construction rather than by coincidence.
    """
    per_layer_values: dict[str, dict[str, float]] = {}
    pooled: list[float] = []
    for layer, state in update.adapters.items():
        s, _, _ = spectral_probe(state.A, state.B)
        vals = s.numpy()
        pooled.extend(float(v) for v in vals)
        block = _summarize(vals)
        block["sigma_min_topr"] = float(factored_spectral_summary(state)["sigma_min_topr"])
        per_layer_values[layer] = block
    out = _summarize(np.asarray(pooled))
    out["per_layer"] = per_layer_values
    return out


def _select(stats: dict, statistic: str, per_layer: bool) -> float:
    if not per_layer:
        return float(stats[statistic])
    vals = [b[statistic] for b in stats["per_layer"].values()]
    finite = [v for v in vals if math.isfinite(v)]
    return float(max(finite)) if finite else math.nan


@DEFENSES.register("spectral_check")
class SpectralCheckDefense(BaseDefense):
    """Robust outlier test on the client's singular-value profile."""

    def __init__(self, config: SpectralCheckConfig) -> None:
        super().__init__(config)
        if config.statistic not in STATISTICS:
            raise ValueError(f"unknown statistic {config.statistic!r}; expected {STATISTICS}")
        if config.direction not in DIRECTIONS:
            raise ValueError(f"unknown direction {config.direction!r}; expected {DIRECTIONS}")
        self.cfg = config

    def apply(self, updates, context) -> DefenseResult:
        p = np.asarray(context.base_weights, dtype=np.float64)
        if not updates:
            return DefenseResult(weights=p, per_client_stats={}, verdicts={})
        full = [client_spectral_stats(u, self.cfg.per_layer) for u in updates]
        scored = np.array(
            [_select(s, self.cfg.statistic, self.cfg.per_layer) for s in full], dtype=np.float64
        )
        finite = scored[np.isfinite(scored)]
        centre = float(np.median(finite)) if finite.size else math.nan
        mad = float(np.median(np.abs(finite - centre))) if finite.size else math.nan
        scale = MAD_TO_SIGMA * mad

        with np.errstate(invalid="ignore", divide="ignore"):
            if scale > 0:
                z = (scored - centre) / scale
            else:
                # A degenerate spread means every client agrees on this statistic. That is
                # unanimity, not anomaly: z = 0 for all, so nobody is rejected. Dividing by
                # zero here would reject the whole round on a rounding difference.
                z = np.zeros_like(scored)
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

        if self.cfg.direction == HIGH:
            flagged = z > self.cfg.z_threshold
        elif self.cfg.direction == LOW:
            flagged = z < -self.cfg.z_threshold
        else:
            flagged = np.abs(z) > self.cfg.z_threshold

        q = p * (~flagged).astype(np.float64)
        stats, verdicts = {}, {}
        for i, u in enumerate(updates):
            entry = {k: v for k, v in full[i].items() if k != "per_layer"}
            entry |= {
                "scored_statistic": self.cfg.statistic,
                "scored_value": float(scored[i]),
                "robust_z": float(z[i]),
                "median": centre,
                "mad_scaled": scale,
                "flagged": bool(flagged[i]),
            }
            if self.cfg.per_layer:
                entry["per_layer"] = full[i]["per_layer"]
            stats[u.client_id] = entry
            verdicts[u.client_id] = REJECT if flagged[i] else ACCEPT
        return DefenseResult(
            weights=q,
            per_client_stats=stats,
            verdicts=verdicts,
            round_stats={
                "statistic": self.cfg.statistic,
                "median": centre,
                "mad_scaled": scale,
                "z_threshold": self.cfg.z_threshold,
                "direction": self.cfg.direction,
                "per_layer": self.cfg.per_layer,
                "n_flagged": int(flagged.sum()),
            },
        )
