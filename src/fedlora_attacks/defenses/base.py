"""Defense interface + registry (CLAUDE.md "defenses/", plan v3 §3 defence matrix).

A defense does NOT filter the update list. It returns a per-client WEIGHT vector
``q_i >= 0`` (plan v3 Phase 1.1), and the aggregator combines every client under those
weights. Filtering is then the special case ``q_i in {0, p_i}``, clipping is
``q_i = p_i * min(1, C / ||dW_i||_F)``, and a selection rule (Multi-Krum) is selection
followed by renormalization. One mechanism covers all three, and — the reason it is
worth the breaking change — a *reweighting* defence is not expressible as a filter at
all, so the previous list-returning interface could not represent the norm bound that
v3's C2 is about.

``q`` is NOT renormalized by the aggregator. A norm bound that clips deliberately
removes mass from the aggregate; renormalizing would silently undo the defence.

Alongside the weights every defence reports ``per_client_stats`` — the raw statistics
it judged on. That is E-B's entire data source: the claim under test is that the
statistics are bit-identical between a benign run and its attacked twin, which is
strictly stronger (and a far better figure) than the verdicts agreeing.

This module holds the interface and the pass-through. The real defences live in their
own modules and register on import: ``norm_bound`` (C2's named defence), ``multi_krum``
(the trap cell), ``cosdefense`` (direction), ``spectral_check`` (the statistic the
other three cannot see), ``fltrust`` (clean-root trust bootstrapping), and
``aggregate_norm_check`` (aggregate level).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..fl.types import AdapterDict, ClientUpdate
from ..utils.registry import Registry


@dataclass
class DefenseConfig:
    name: str = "none"


@dataclass
class DefenseContext:
    """What a defence may look at when scoring one round's submissions.

    ``base_weights`` are the protocol's FedAvg weights p_i (``normalized_weights``),
    supplied rather than recomputed so that every defence expresses its output in the
    same units: ``q_i = p_i * (something in [0, 1])`` for a clipping rule, ``q_i = 0``
    for a rejection. A defence that ignores p_i and returns bare multipliers would
    silently discard the data-size weighting.
    """

    round_idx: int
    base_weights: list[float]
    trusted_update: ClientUpdate | None = None
    """Optional clean server update used by trust-bootstrapping defenses.

    ``None`` for every existing defense and run.  Keeping the reference in the context
    rather than in the submitted update list prevents it from being counted as a client,
    attacked, aggregated, or included in client-level probes.
    """
    broadcast_adapters: AdapterDict | None = None
    """Actual server factors broadcast at the start of this round.

    Most defenses deliberately ignore their representation.  A factor-space defense
    must receive this state explicitly so it cannot accidentally compare against a
    post-filter aggregate, a canonical frame, or the previous run's trajectory.
    """


@dataclass
class DefenseResult:
    """A defence's full output for one round.

    ``weights`` is load-bearing: aligned positionally with the ``updates`` list handed
    to ``apply``, non-negative, and NOT required to sum to 1.

    ``per_client_stats`` maps client_id -> {statistic name: value} and is keyed by id
    (not position) because it is what the analysis layer joins across paired runs.
    ``verdicts`` maps client_id -> one of "accept" / "clip" / "reject": a coarse label
    for figures, never a substitute for the statistics.
    """

    weights: np.ndarray
    per_client_stats: dict[int, dict[str, Any]] = field(default_factory=dict)
    verdicts: dict[int, str] = field(default_factory=dict)
    round_stats: dict[str, Any] = field(default_factory=dict)
    """Statistics of the ROUND rather than of any client.

    Needed by the aggregate-level cell of the defence matrix: a check on ``||A_bar||_F``
    has no client to attach to. Kept separate from ``per_client_stats`` because the two
    are read differently — and, for the aggregate norm check specifically, because its
    statistics are gauge-VARIANT and must not be mistaken for per-client evidence. Such
    entries carry an explicit ``gauge_variant`` marker; see defenses/aggregate_norm_check.
    """

    def to_record(self) -> dict:
        """JSON-safe form for the per-round log (server writes this into the record)."""
        return {
            "weights": [float(w) for w in self.weights],
            "per_client_stats": {str(cid): s for cid, s in self.per_client_stats.items()},
            "verdicts": {str(cid): v for cid, v in self.verdicts.items()},
            "round_stats": dict(self.round_stats),
        }


ACCEPT = "accept"
CLIP = "clip"
REJECT = "reject"


@runtime_checkable
class Defense(Protocol):
    config: DefenseConfig

    def apply(self, updates: list[ClientUpdate], context: DefenseContext) -> DefenseResult: ...


DEFENSES: Registry[type[Defense]] = Registry("defense")


class BaseDefense:
    def __init__(self, config: DefenseConfig) -> None:
        self.config = config

    def apply(self, updates: list[ClientUpdate], context: DefenseContext) -> DefenseResult:
        """Default: accept everyone at the protocol weights p_i."""
        return DefenseResult(
            weights=np.asarray(context.base_weights, dtype=np.float64),
            per_client_stats={},
            verdicts={u.client_id: ACCEPT for u in updates},
        )


@DEFENSES.register("none")
class NoDefense(BaseDefense):
    """Pass-through: every client aggregated at its FedAvg weight p_i.

    Computes no statistics, so ``per_client_stats`` is empty — there is nothing to be
    blind with. This is the E-A/E-B baseline arm.
    """


def build_defense(config: DefenseConfig) -> Defense:
    return DEFENSES.get(config.name)(config)
