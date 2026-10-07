"""dW-norm bounding: the named deployed defence that C2 is a claim about. END-TO-END.

    q_i = p_i * min(1, C / ||dW_i||_F)

Clipping, not rejection — the canonical defence against the canonical untargeted attack
(scaling), in the lineage of Fang et al. and Shejwalkar-Houmansadr.

**Why it is *specifically* blind, not merely unlucky.** The bound is on `||dW_i||_F`,
which is a function of `dW_i` alone and therefore EXACTLY fixed by the gauge
`(A, B) -> (A inv(T), T B)`. A coalition can take `T^-1 = c*I` with `c` large, inflate
every transmitted `A_i` by `c`, scale the factor-averaged aggregate arbitrarily, and
this defence's statistic does not move by one ulp. No tuning of `C` helps: the quantity
being tuned is the wrong one. That is C2, and it is sharper than "the attack evades the
defence" — the defence is measuring something the manipulation provably preserves.

**And ADAPTING C does not help either, which is the sharper form.** With
``threshold_mode='median'`` the bound is set from the round's own traffic,
`C_t = gamma * median_i ||dW_i||_F`, as a deployed system would. An adaptive threshold
is normally *harder* to evade than a fixed one, because the attacker must move the
statistic without moving the population it is calibrated against. Here it is not harder
at all: Channel A leaves every `||dW_i||_F` untouched, and Channel B preserves singular
values and hence the Frobenius norm, so the threshold adapts to **exactly the same
value** under attack as under benign traffic. The defence recalibrates itself into the
same blindness. tests/test_norm_bound_median.py is that claim.

Computed from the factors via ``deltaw_frobenius_norm`` (two r x r Grams and a trace),
so nothing of size n*m is formed. Per plan v3 §6b this matters to the argument:
representation-independence is CHEAP, which removes the efficiency excuse for
factor-space statistics and makes the trap cell harder to defend, not easier.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..probes.geometry import deltaw_frobenius_norm
from .base import ACCEPT, CLIP, DEFENSES, BaseDefense, DefenseConfig, DefenseResult

FIXED = "fixed"
MEDIAN = "median"
THRESHOLD_MODES = (FIXED, MEDIAN)


@dataclass
class NormBoundConfig(DefenseConfig):
    name: str = "norm_bound"
    threshold_mode: str = MEDIAN
    """How C is chosen. ``median`` (default) sets it from the round's own traffic;
    ``fixed`` uses ``max_delta_norm`` as an absolute constant.

    ``median`` is the default because a fixed C cannot be chosen without knowing the
    scale of ||dW||_F for the model, rank, lr and round in question — the pre-v3 default
    of 1.0 clipped every client on every realistic config, which saturates the defence
    arm and makes any comparison meaningless. It is also what a deployed system does.
    """
    gamma: float = 1.5
    """C_t = gamma * median_i ||dW_i||_F. Only used when threshold_mode='median'."""
    max_delta_norm: float = 1.0
    """C. Only used when threshold_mode='fixed'."""
    per_layer: bool = False
    """Bound each layer's ||dW_l||_F instead of the network-wide norm.

    Default is network-wide sqrt(sum_l ||dW_l||_F^2), the quantity C2 is stated about.
    The per-layer variant is run alongside it in E-B: both are dW-measurable and hence
    equally blind, and showing that pre-empts the reading that the result is an artifact
    of summing over layers.

    Since `q` is one scalar per client, the per-layer variant clips by the MOST violating
    layer — `min_l min(1, C_l/||dW_l||_F)`. A genuinely per-layer reweighting would need
    a per-layer weight interface; the blindness claim is unaffected either way, because
    every per-layer norm is just as gauge-invariant as the global one. Under `median` the
    threshold is likewise per layer: `C_l = gamma * median_i ||dW_{i,l}||_F`.
    """

    def __post_init__(self) -> None:
        if self.threshold_mode not in THRESHOLD_MODES:
            raise ValueError(
                f"threshold_mode must be one of {THRESHOLD_MODES}, got {self.threshold_mode!r}"
            )


@DEFENSES.register("norm_bound")
class NormBoundDefense(BaseDefense):
    """Clip client weights by ||dW_i||_F (plan v3 C2, experiments E-A / E-B)."""

    def __init__(self, config: NormBoundConfig) -> None:
        super().__init__(config)
        self.cfg = config

    @staticmethod
    def _median(values: list[float]) -> float:
        return float(np.median(np.asarray(values, dtype=np.float64))) if values else 0.0

    def thresholds(
        self, per_client: list[dict[str, float]], global_norms: list[float]
    ) -> tuple[float, dict[str, float], float]:
        """(network-wide C, per-layer C_l, the median the threshold was set from).

        The median is taken over **all** clients, coalition included. Under both channels
        the coalition's ||dW_i|| is honest, so there is nothing for it to manipulate here;
        and excluding it would require knowing who it is, which is the problem the defence
        was supposed to solve.
        """
        if self.cfg.threshold_mode == FIXED:
            C = self.cfg.max_delta_norm
            layers = per_client[0].keys() if per_client else []
            return C, dict.fromkeys(layers, C), math.nan

        median_global = self._median(global_norms)
        per_layer = {
            layer: self.cfg.gamma * self._median([c[layer] for c in per_client])
            for layer in (per_client[0].keys() if per_client else [])
        }
        return self.cfg.gamma * median_global, per_layer, median_global

    def clip_factor(
        self, per_layer_norms: dict[str, float], global_norm: float, C: float, C_layer: dict
    ) -> float:
        """The clip factor in (0, 1] for one client."""
        if self.cfg.per_layer:
            return min(
                [1.0]
                + [
                    C_layer[layer] / v
                    for layer, v in per_layer_norms.items()
                    if v > 0.0 and C_layer.get(layer, 0.0) > 0.0
                ]
            )
        return min(1.0, C / global_norm) if global_norm > 0.0 and C > 0.0 else 1.0

    def apply(self, updates, context) -> DefenseResult:
        per_client = [
            {layer: deltaw_frobenius_norm(s.A, s.B) for layer, s in u.adapters.items()}
            for u in updates
        ]
        global_norms = [math.sqrt(sum(v * v for v in norms.values())) for norms in per_client]
        C, C_layer, median = self.thresholds(per_client, global_norms)

        weights, stats, verdicts = [], {}, {}
        for u, p, norms, g in zip(
            updates, context.base_weights, per_client, global_norms, strict=True
        ):
            factor = self.clip_factor(norms, g, C, C_layer)
            weights.append(p * factor)
            # Everything the defence judged on. E-B's claim is that THESE are identical
            # across a paired run, which is strictly stronger than the verdicts agreeing.
            stats[u.client_id] = {
                "deltaw_norm": g,
                "per_layer_norm": norms,
                "clip_factor": factor,
                "bound": C,
            }
            verdicts[u.client_id] = ACCEPT if factor >= 1.0 else CLIP
        return DefenseResult(
            weights=np.asarray(weights, dtype=np.float64),
            per_client_stats=stats,
            verdicts=verdicts,
            # NO gauge_variant tag, deliberately: unlike aggregate_norm_check every entry
            # here is a function of the dW_i, so all of it is gauge-invariant and may be
            # reported as a property of the round.
            round_stats={
                "threshold_mode": self.cfg.threshold_mode,
                "gamma": self.cfg.gamma,
                "threshold_C": C,
                "median_deltaw_norm": median,
                "per_layer_threshold": C_layer if self.cfg.per_layer else {},
                "num_clipped": sum(1 for v in verdicts.values() if v == CLIP),
            },
        )
