"""Aggregate-level norm check: the reviewer's first objection, measured. (T8 / E-G)

A loud Channel A attack — large `cond(T)` — inflates the transmitted `A_i`, so the
factor average `A_bar = sum_i q_i A_i` becomes anomalous even though every client's
`dW_i` is untouched. An aggregate-level sanity check therefore CAN see the attack, and a
reviewer will propose exactly that. The paper has to answer it rather than wait to be
asked, which is T8's whole point: *what is the maximum accuracy degradation achievable
subject to the aggregate passing this check?* This module is the constraint that
question is posed against.

**Its statistics are gauge-VARIANT, and that is deliberate.** `||A_bar||_F` is not a
function of any `dW`; it reads the transmitted representation. That makes it the one
check that can see Channel A, and simultaneously the one whose numbers may not be
reported as a property of a round (CLAUDE.md invariant 1). Every entry it emits is
tagged ``"gauge_variant": true`` so the analysis layer cannot mistake it for evidence
about an update. The legitimate reading is DRIFT ALONG THE ORBIT: the attacked arm's
value against its benign twin's at the same round, which is exactly the paired quantity
E-G needs.

`||A_bar B_bar||_F` is logged beside it and is gauge-variant for a subtler reason — it
is a norm of the *naive* aggregate, which is not a function of the `dW_i` either
(`mean(A) mean(B) != mean(AB)`); only under exact aggregation would it become invariant.
Both are tagged.

**Two actions.** ``monitor`` (default) reports pass/fail and leaves `q = p` untouched:
that is a detector, and T8 asks whether the attacker can stay under it, not whether it
stops them. ``clip`` additionally rescales every weight uniformly so the round's
`||A_bar||_F` returns to the tolerated band, giving E-G an arm where the check bites.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..fl.aggregation.base import active_clients
from .base import ACCEPT, CLIP, DEFENSES, BaseDefense, DefenseConfig, DefenseResult

MONITOR = "monitor"
CLIP_ACTION = "clip"
ACTIONS = (MONITOR, CLIP_ACTION)

INFLATION = "inflation"
TWO_SIDED = "two_sided"
DIRECTIONS = (INFLATION, TWO_SIDED)


def momentum_from_half_life(half_life: float) -> float:
    """EMA momentum `m` such that the baseline's weight on an old value halves every
    ``half_life`` rounds: `m = 0.5 ** (1 / half_life)`.

    Half-life is the interpretable axis — it is measured in ROUNDS, so it can be compared
    directly against how fast the attack moves the aggregate and how fast benign training
    grows it. Momentum cannot: 0.9 vs 0.95 is a factor of two in memory and does not look
    like one.

    ``inf`` gives `m = 1.0`, a baseline FROZEN at its first observation. That is the
    limiting case E-G needs, because it is the only EMA that cannot be walked along by a
    slow attack — and, as the sweep shows, it is also the one that cannot track benign
    non-stationarity.
    """
    if half_life == math.inf:
        return 1.0
    if half_life <= 0.0:
        raise ValueError(f"half_life must be > 0 (or inf), got {half_life}")
    return float(0.5 ** (1.0 / half_life))


@dataclass
class BaselineDecision:
    """One round's verdict, and the numbers behind it."""

    baseline: float | None
    limit_high: float | None
    limit_low: float | None
    passed: bool
    fired: str  # "" | "high" | "low"
    warmup: bool


class EMABaseline:
    """The check's decision rule, as a standalone object.

    Extracted from the defence so the OFFLINE REPLAY in
    ``analysis/monitor_replay.py`` runs the identical rule rather than a
    reimplementation of it. A monitor-action check leaves `q = p` untouched and so cannot
    perturb the trajectory it observes, which makes replaying it over a stored
    ``factor_norms`` series exact — but only as long as one rule exists in one place.
    Two copies would be a silent divergence waiting to happen, and the whole E-G result
    rests on the replay being faithful.
    """

    def __init__(
        self,
        momentum: float,
        tolerance: float,
        tolerance_low: float,
        direction: str,
        warmup_rounds: int,
    ) -> None:
        self.momentum = momentum
        self.tolerance = tolerance
        self.tolerance_low = tolerance_low
        self.direction = direction
        self.warmup_rounds = warmup_rounds
        self.baseline: float | None = None
        self.rounds_seen = 0

    def decide(self, observed: float) -> BaselineDecision:
        """Judge ``observed`` against the baseline as it stands BEFORE this round."""
        self.rounds_seen += 1
        warming = self.rounds_seen <= self.warmup_rounds
        high = None if self.baseline is None else self.tolerance * self.baseline
        low = (
            None
            if self.baseline is None or self.direction != TWO_SIDED
            else self.baseline / self.tolerance_low
        )
        fired = ""
        if not warming:
            if high is not None and observed > high:
                fired = "high"
            elif low is not None and observed < low:
                fired = "low"
        return BaselineDecision(self.baseline, high, low, fired == "", fired, warming)

    def commit(self, accepted: float) -> None:
        """Fold the ACCEPTED value into the baseline (see the defence's clip path)."""
        self.baseline = (
            accepted
            if self.baseline is None
            else self.momentum * self.baseline + (1.0 - self.momentum) * accepted
        )


@dataclass
class AggregateNormCheckConfig(DefenseConfig):
    name: str = "aggregate_norm_check"
    tolerance: float = 2.0
    """Fail the round when ||A_bar||_F exceeds tolerance x the running baseline."""
    tolerance_low: float = 2.0
    """Two-sided only: fail when ||A_bar||_F falls below baseline / tolerance_low.

    A separate knob from ``tolerance`` because the two sides are not symmetric threats.
    E-A2 §4 measured 0/500 upward excursions under a bounded gauge and 111-120/125
    downward, so the low side is the only one that can see Channel A at all — while the
    high side is the only one that sees `scaling`. A single tolerance would force one
    of the two to be mis-calibrated.
    """
    baseline_momentum: float = 0.9
    """EMA momentum for the baseline. The baseline tracks BENIGN rounds only in spirit;
    in practice it is seeded by the first round and drifts, which is itself part of what
    E-G measures — a slow attack can walk the baseline along with it.

    Ignored when ``baseline_half_life`` is set, which is the preferred parameterisation."""
    baseline_half_life: float | None = None
    """Half-life in ROUNDS; overrides ``baseline_momentum`` when set. ``.inf`` freezes."""
    direction: str = INFLATION
    """``inflation`` watches only for growth (the reviewer's check, and the one that
    catches `scaling`); ``two_sided`` also watches for shrinkage, which is the only side
    a norm-preserving gauge can move (E-A2 §4)."""
    action: str = MONITOR
    """``monitor`` reports only (q = p); ``clip`` rescales q uniformly to pass."""
    warmup_rounds: int = 1
    """Rounds accepted unconditionally while the baseline is established."""

    def __post_init__(self) -> None:
        if self.action not in ACTIONS:
            raise ValueError(f"action must be one of {ACTIONS}, got {self.action!r}")
        if self.direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {DIRECTIONS}, got {self.direction!r}")

    @property
    def momentum(self) -> float:
        if self.baseline_half_life is None:
            return self.baseline_momentum
        return momentum_from_half_life(float(self.baseline_half_life))


@DEFENSES.register("aggregate_norm_check")
class AggregateNormCheckDefense(BaseDefense):
    """Stateful aggregate-level check on ||A_bar||_F (plan v3 T8, experiment E-G).

    The first STATEFUL defence: it carries a running baseline across rounds, so one
    instance belongs to one run and must not be shared between the arms of a paired
    comparison (each arm keeps its own baseline, which is the point — the attacked arm's
    baseline is itself attackable).
    """

    def __init__(self, config: AggregateNormCheckConfig) -> None:
        super().__init__(config)
        self.cfg = config
        self.tracker = EMABaseline(
            momentum=config.momentum,
            tolerance=config.tolerance,
            tolerance_low=config.tolerance_low,
            direction=config.direction,
            warmup_rounds=config.warmup_rounds,
        )

    @property
    def baseline(self) -> float | None:
        return self.tracker.baseline

    @property
    def rounds_seen(self) -> int:
        return self.tracker.rounds_seen

    @staticmethod
    def _aggregate_factor_norms(updates, weights) -> tuple[float, float, dict[str, float]]:
        """(||A_bar||_F, ||A_bar B_bar||_F, per-layer ||A_bar_l||_F) for the naive average.

        Deliberately the FACTOR average, because that is the object a deployed FedIT
        server actually forms and the only place the inflation shows up.
        """
        from ..probes.geometry import deltaw_frobenius_norm

        per_layer: dict[str, float] = {}
        a_sq = 0.0
        prod_sq = 0.0
        for layer in updates[0].adapters:
            A_bar = sum(w * u.adapters[layer].A for w, u in zip(weights, updates, strict=True))
            B_bar = sum(w * u.adapters[layer].B for w, u in zip(weights, updates, strict=True))
            a_norm = float(A_bar.norm().item())
            per_layer[layer] = a_norm
            a_sq += a_norm**2
            prod_sq += deltaw_frobenius_norm(A_bar, B_bar) ** 2
        return math.sqrt(a_sq), math.sqrt(prod_sq), per_layer

    def apply(self, updates, context) -> DefenseResult:
        active, weights = active_clients(updates, context.base_weights)
        a_norm, prod_norm, per_layer = self._aggregate_factor_norms(active, weights)

        decision = self.tracker.decide(a_norm)
        passed, limit = decision.passed, decision.limit_high

        scale = 1.0
        if not passed and self.cfg.action == CLIP_ACTION and a_norm > 0.0:
            # A_bar is linear in q, so scaling every weight by limit/||A_bar|| brings the
            # round exactly to the tolerated band. Uniform on purpose: this check has no
            # per-client evidence and must not pretend to. Only the HIGH side is
            # actionable this way — a round that fired for being too SMALL cannot be
            # fixed by scaling weights up without inventing signal, so `clip` leaves it.
            if decision.fired == "high" and limit is not None:
                scale = limit / a_norm

        # The baseline follows what was ACCEPTED, so a clipped round does not teach the
        # detector to tolerate the inflation it just rejected.
        self.tracker.commit(a_norm * scale)

        q = np.asarray([scale * p for p in context.base_weights], dtype=np.float64)
        verdict = ACCEPT if (passed or scale == 1.0) else CLIP
        return DefenseResult(
            weights=q,
            per_client_stats={},  # this check has NO per-client evidence, by construction
            verdicts={u.client_id: verdict for u in updates},
            round_stats={
                # Gauge-VARIANT: readable only as drift against a benign twin at the same
                # round, never as a property of this round on its own (invariant 1).
                "gauge_variant": True,
                "aggregate_A_norm": a_norm,
                "aggregate_deltaw_norm": prod_norm,
                "per_layer_A_norm": per_layer,
                "baseline": self.baseline,
                "limit": limit,
                "limit_low": decision.limit_low,
                "passed": bool(passed),
                "fired": decision.fired,
                "scale_applied": scale,
                "warmup": bool(decision.warmup),
                "half_life": self.cfg.baseline_half_life,
                "direction": self.cfg.direction,
            },
        )
