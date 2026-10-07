"""Probe interface + registry (CLAUDE.md invariant 1, "Extensibility conventions").

Probes are pure functions of one round's data. Each declares one of THREE categories,
and the category determines which test the probe owes:

* ``invariant``     — a function of dW; unchanged under (A,B) -> (A inv(T), T B) for
  every T in GL(r). Owes tests/test_gauge_invariance.py (both det signs) and
  tests/test_insitu_gauge_invariance.py on real trained factors.
* ``equivariant``   — deliberately not invariant, with a defined transformation law.
  The det-sign obstruction (w_1). Owes a DETECTION test: it must flag a known
  orientation reversal and not flag an orientation-preserving map. Never write a test
  forcing it to be invariant.
* ``gauge_variant`` — not a function of dW at all. Reportable ONLY as drift along a
  gauge orbit, never as a property of an update, a client or a round. Owes a VARIANCE
  test: it must move under a gauge for which theory says it moves, and not move under
  one for which theory says it does not. Needed by E-G (||A_bar||_F) and E-I (the
  magnitude-sensitive frame estimator), where the movement IS the finding.

Two categories were not enough: E-G and E-I measure quantities that are legitimately
gauge-variant, and folding them into ``invariant=False`` alongside the det-sign probe
would have conflated "has a transformation law we exploit" with "is not a property of
the update at all".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from ..fl.types import AdapterDict, AggregateResult, ClientUpdate
from ..utils.registry import Registry

INVARIANT = "invariant"
EQUIVARIANT = "equivariant"
GAUGE_VARIANT = "gauge_variant"

ProbeCategory = Literal["invariant", "equivariant", "gauge_variant"]
PROBE_CATEGORIES = (INVARIANT, EQUIVARIANT, GAUGE_VARIANT)


@dataclass
class ProbeContext:
    """Everything a probe may need for one round — all dW-derivable."""

    client_updates: list[ClientUpdate]
    aggregate: AggregateResult
    round_idx: int
    prev_aggregate: AggregateResult | None = None
    prev_client_updates: list[ClientUpdate] | None = None
    """Last round's client updates, matched by ``client_id``. Needed by the alignment
    probe's round-to-round (client_i^t vs client_i^{t-1}) pairing — the one signed
    quantity that requires the previous round's raw per-client factors, which are not
    reconstructable from stored logs. None on the first round."""
    weights: list[float] | None = None
    """The weights q_i the round actually aggregated under (defence output, or p_i).

    Positionally aligned with ``client_updates``. A probe that reconstructs an aggregate
    MUST use these rather than recomputing FedAvg weights from ``num_examples``, or
    under a reweighting defence it would report the spectrum of an aggregate the server
    never formed. None means "the protocol weights p_i".
    """
    coalition_ids: tuple[int, ...] = ()
    """Which clients count as the coalition, for probes that measure a K/H partition.

    Supplied rather than inferred from ``metadata["malicious"]`` because the partition is
    a MEASUREMENT parameter, not an attack one: E-A3's lever check wants the same split
    in the benign arm, where no client is malicious and the metadata would be empty.
    Empty tuple means "no partition" and the partitioned statistics are skipped.
    """
    canonical_frames: AdapterDict | None = None
    """Rank-r SVD-truncated frame of the EXACT aggregate, per layer.

    A gauge-canonical reference frame, computed when requested independently of which
    aggregation strategy is in force. Probes that need a reference (angles, det-sign)
    report against both this and the in-force consensus, so a canonical-gauge reading
    of the benign null needs no re-run. None when the server did not compute it.
    """

    @property
    def consensus(self):
        """The redistributed global adapters (the consensus frame this round)."""
        return self.aggregate.next_global


@dataclass
class ProbeResult:
    name: str
    category: ProbeCategory
    values: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.category not in PROBE_CATEGORIES:
            raise ValueError(
                f"probe '{self.name}': category must be one of {PROBE_CATEGORIES}, "
                f"got {self.category!r}"
            )

    @property
    def invariant(self) -> bool:
        """True only for the ``invariant`` category.

        Deliberately NOT the negation of "gauge_variant": an equivariant probe is not
        invariant either, and treating the split as binary is what the third category
        exists to prevent.
        """
        return self.category == INVARIANT


@runtime_checkable
class Probe(Protocol):
    name: str
    category: ProbeCategory

    def __call__(self, ctx: ProbeContext) -> ProbeResult: ...


PROBES: Registry[type[Probe]] = Registry("probe")


def build_probes(names: list[str]) -> list[Probe]:
    return [PROBES.get(n)() for n in names]


def run_probes(ctx: ProbeContext, probes: list[Probe]) -> dict[str, ProbeResult]:
    return {p.name: p(ctx) for p in probes}
