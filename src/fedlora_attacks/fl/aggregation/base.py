"""Aggregation strategy interface + registry (CLAUDE.md invariant 3).

Factor-wise averaging and exact-dW averaging differ:
``mean(A_i) @ mean(B_i) != mean(A_i @ B_i)`` in general. Both are implemented and
each strategy's test asserts its defining property. The discrepancy between them is
itself an object of study (§1 efficiency-vs-invariance gap) — ``deltaw_discrepancy``
exposes it rather than hiding it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from ...utils.registry import Registry
from ..types import AggregateResult, ClientUpdate, LayerName

Weights = Sequence[float]


@runtime_checkable
class Aggregation(Protocol):
    name: str

    def aggregate(
        self, updates: list[ClientUpdate], weights: Weights | None = None
    ) -> AggregateResult: ...


AGGREGATIONS: Registry[type[Aggregation]] = Registry("aggregation")


def build_aggregation(name: str, **kwargs) -> Aggregation:
    return AGGREGATIONS.get(name)(**kwargs)


def normalized_weights(updates: list[ClientUpdate]) -> list[float]:
    """FedAvg weights p_i = num_examples_i / sum_j num_examples_j.

    The protocol weights, and the default when no defence reweights. Falls back to
    uniform weights if all counts are zero.
    """
    counts = [max(u.num_examples, 0) for u in updates]
    total = sum(counts)
    if total <= 0:
        return [1.0 / len(updates)] * len(updates)
    return [c / total for c in counts]


def resolve_weights(updates: list[ClientUpdate], weights: Weights | None) -> list[float]:
    """Validate an aggregation weight vector, defaulting to the protocol weights p_i.

    Weights come from the defence (``DefenseResult.weights``) and are deliberately NOT
    renormalized: a norm bound that clips removes mass from the aggregate on purpose,
    and rescaling back to sum 1 would undo exactly the defence being measured.
    """
    if weights is None:
        return normalized_weights(updates)
    w = [float(x) for x in weights]
    if len(w) != len(updates):
        raise ValueError(f"expected {len(updates)} weights, got {len(w)}")
    if any(x < 0.0 for x in w):
        raise ValueError("aggregation weights must be non-negative")
    return w


def active_clients(
    updates: list[ClientUpdate], weights: Weights | None
) -> tuple[list[ClientUpdate], list[float]]:
    """Drop zero-weight clients, keeping order. Exact, not an approximation.

    A client at weight 0 contributes nothing to any weighted sum, so removing it leaves
    every aggregate bit-identical while (i) making a weight-expressed rejection produce
    literally the same arithmetic as the old list-filtering interface, and (ii) keeping
    the exact aggregate's rank ceiling honest at |S|*r rather than N*r — a rejected
    client should not inflate the reported rank inflation.

    If NO client has positive weight the original lists are returned unchanged, so the
    arithmetic degenerates to a zero aggregate instead of an empty-list crash.
    """
    w = resolve_weights(updates, weights)
    kept = [(u, x) for u, x in zip(updates, w, strict=True) if x > 0.0]
    if not kept:
        return updates, w
    return [u for u, _ in kept], [x for _, x in kept]


def _layer_names(updates: list[ClientUpdate]) -> list[LayerName]:
    names = list(updates[0].adapters.keys())
    for u in updates[1:]:
        if list(u.adapters.keys()) != names:
            raise ValueError("all client updates must share the same layer set/order")
    return names


def canonical_frames(updates: list[ClientUpdate], weights: Weights | None = None):
    """Per-layer rank-r SVD truncation of the EXACT aggregate — the canonical frame.

    Independent of whichever strategy is in force. This is a *gauge-fixed* reference:
    A carries the singular values of the exact mean dW, B has orthonormal rows, so the
    frame is determined by dW alone (up to sign/degeneracy) rather than by the clients'
    arbitrary factorizations. Probes needing a reference report against both this and
    the in-force consensus, which is what makes a canonical-gauge reading of the benign
    null available without a second sweep.

    Takes the same ``weights`` as the in-force aggregation: the exact aggregate (and so
    its canonical frame) is defined relative to a weighting, and a defence that
    reweights must move this reference too or probes would compare against a frame the
    round never had.
    """
    from ..types import AdapterDict
    from .exact_deltaw import ExactDeltaWAggregation

    result: AdapterDict = ExactDeltaWAggregation().aggregate(updates, weights).next_global
    return result


def deltaw_discrepancy(
    updates: list[ClientUpdate], weights: Weights | None = None
) -> dict[LayerName, float]:
    """Per-layer Frobenius distance ||fedit_naive.dW - exact_deltaw.dW||_F.

    The §1 gap: nonzero whenever clients' factorizations are not gauge-aligned.

    Computed WITHOUT materializing any n x m matrix. The difference is itself a low-rank
    product: with A_agg = sum w_i A_i and B_agg = sum w_i B_i,

        naive - exact = A_agg B_agg - sum_i w_i A_i B_i
                      = [A_agg | -w_1 A_1 | ... | -w_N A_N] @ [B_agg ; B_1 ; ... ; B_N]

    of inner dimension (N+1)r, so ``factored_frobenius_norm`` gives the exact norm from
    thin QRs (measured ~50x cheaper than forming both aggregates at bert-base scale).

    Measured under whatever ``weights`` the round actually aggregated with, so that a
    reweighting defence changes the reported gap instead of being averaged out of it.
    """
    from ...probes.geometry import factored_frobenius_norm
    from ..types import AdapterState, stacked_factors
    from .fedit_naive import FedITNaiveAggregation

    active, w = active_clients(updates, weights)
    naive = FedITNaiveAggregation().aggregate(active, w)
    out: dict[LayerName, float] = {}
    for name in _layer_names(active):
        agg = naive.aggregate_state[name]
        states = [agg] + [u.adapters[name] for u in active]
        signed = [1.0] + [-x for x in w]
        diff: AdapterState = stacked_factors(states, signed)
        out[name] = factored_frobenius_norm(diff.A, diff.B)
    return out
