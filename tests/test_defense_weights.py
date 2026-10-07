"""Defense weights must be applied consistently across aggregation and metrics."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fedlora_attacks.defenses.base import (
    ACCEPT,
    DefenseContext,
    DefenseResult,
    NoDefense,
    build_defense,
)
from fedlora_attacks.defenses.base import DefenseConfig as DCfg
from fedlora_attacks.fl.aggregation import (
    ExactDeltaWAggregation,
    FedITNaiveAggregation,
    deltaw_discrepancy,
    normalized_weights,
    resolve_weights,
)
from fedlora_attacks.fl.server import FederatedServer
from fedlora_attacks.fl.types import AdapterState, ClientUpdate

TOL = 1e-12


def _make_updates(n=6, m=5, r=3, num_clients=4, with_head=True):
    """Distinct per-client counts so p_i is non-uniform and a missed site would show."""
    g = torch.Generator().manual_seed(0)
    updates = []
    for c in range(num_clients):
        A = torch.randn(n, r, generator=g, dtype=torch.float64)
        B = torch.randn(r, m, generator=g, dtype=torch.float64)
        head = (
            {"classifier.weight": torch.randn(2, n, generator=g, dtype=torch.float64)}
            if with_head
            else None
        )
        updates.append(
            ClientUpdate(
                c, {"layer0": AdapterState(A, B)}, num_examples=(c + 1) * 10, head_state=head
            )
        )
    return updates


def _renormalized_over(updates, keep: set[int]) -> list[float]:
    """p_i restricted to ``keep`` and renormalized — the weight form of a filter."""
    total = sum(u.num_examples for u in updates if u.client_id in keep)
    return [(u.num_examples / total if u.client_id in keep else 0.0) for u in updates]


# -- the interface ---------------------------------------------------------------------


def test_no_defense_returns_the_protocol_weights_and_accepts_everyone():
    updates = _make_updates()
    ctx = DefenseContext(round_idx=0, base_weights=normalized_weights(updates))
    result = NoDefense(DCfg(name="none")).apply(updates, ctx)

    assert isinstance(result, DefenseResult)
    assert result.weights == pytest.approx(normalized_weights(updates))
    assert result.verdicts == {u.client_id: ACCEPT for u in updates}
    # NoDefense judges nothing, so it has no statistics to be blind with.
    assert result.per_client_stats == {}


def test_weight_vector_is_validated():
    updates = _make_updates(num_clients=3)
    with pytest.raises(ValueError, match="expected 3 weights"):
        resolve_weights(updates, [0.5, 0.5])
    with pytest.raises(ValueError, match="non-negative"):
        resolve_weights(updates, [0.5, -0.1, 0.6])


def test_build_defense_still_resolves_by_name():
    assert isinstance(build_defense(DCfg(name="none")), NoDefense)


# -- 1.1's acceptance test: a filter expressed as weights is bit-identical --------------


@pytest.mark.parametrize("strategy", [FedITNaiveAggregation, ExactDeltaWAggregation])
def test_filter_expressed_as_weights_reproduces_old_filtering_bit_identically(strategy):
    """The pre-v3 server did ``aggregate(defense.apply(updates))`` on a FILTERED list.

    The new server aggregates the FULL list under q, with q_i = 0 for a rejected client
    and p renormalized over the survivors — which is exactly what the old path's
    ``normalized_weights(filtered)`` computed. Bit-identical, not merely close: zero-
    weight clients are dropped before any arithmetic (``active_clients``), so the
    surviving terms are summed in the same order over the same values.
    """
    updates = _make_updates()
    keep = {0, 2, 3}  # the defence "rejects" client 1

    old = strategy().aggregate([u for u in updates if u.client_id in keep])
    new = strategy().aggregate(updates, _renormalized_over(updates, keep))

    for layer in old.next_global:
        assert torch.equal(old.next_global[layer].A, new.next_global[layer].A)
        assert torch.equal(old.next_global[layer].B, new.next_global[layer].B)
        assert torch.equal(old.aggregate_state[layer].A, new.aggregate_state[layer].A)
        assert torch.equal(old.aggregate_state[layer].B, new.aggregate_state[layer].B)


def test_filtered_head_aggregation_is_bit_identical_too():
    """Same claim for the classifier head, which is federated separately."""
    updates = _make_updates()
    keep = {0, 2, 3}
    survivors = [u for u in updates if u.client_id in keep]

    # the old path: plain num_examples FedAvg over the filtered list
    total = sum(u.num_examples for u in survivors)
    old = {
        k: sum((u.num_examples / total) * u.head_state[k] for u in survivors)
        for k in survivors[0].head_state
    }
    new = FederatedServer._aggregate_heads(updates, _renormalized_over(updates, keep))

    for k in old:
        assert torch.equal(old[k], new[k])


def test_no_defense_path_is_unchanged_from_the_pre_v3_default():
    """The benign arm must not move: q = p reproduces the old unweighted call exactly."""
    updates = _make_updates()
    for strategy in (FedITNaiveAggregation, ExactDeltaWAggregation):
        default = strategy().aggregate(updates)
        explicit = strategy().aggregate(updates, normalized_weights(updates))
        for layer in default.next_global:
            assert torch.equal(default.next_global[layer].A, explicit.next_global[layer].A)
            assert torch.equal(default.next_global[layer].B, explicit.next_global[layer].B)


# -- q is a weight, not a mask: clipping must actually remove mass ----------------------


def test_aggregator_does_not_renormalize_clipped_weights():
    """A clip that halves every q must halve the exact aggregate, not be normalized away.

    This is the property a norm bound depends on: renormalizing q back to sum 1 inside
    the aggregator would silently undo the defence.
    """
    updates = _make_updates()
    p = normalized_weights(updates)
    full = ExactDeltaWAggregation().aggregate(updates, p)
    clipped = ExactDeltaWAggregation().aggregate(updates, [0.5 * x for x in p])
    assert torch.allclose(clipped.delta_w["layer0"], 0.5 * full.delta_w["layer0"], atol=TOL)


def test_factor_averaging_responds_quadratically_to_a_uniform_clip():
    """fedit_naive scales BOTH factors, so dW picks up c^2 where exact picks up c.

    Not a bug — it is the §1 gap showing up in the defence's own units, and it means a
    clip's effect on the deployed aggregator is not the effect the defence intends.
    Worth pinning: it is the kind of thing that would otherwise be discovered mid-E-A.
    """
    updates = _make_updates()
    p = normalized_weights(updates)
    full = FedITNaiveAggregation().aggregate(updates, p)
    clipped = FedITNaiveAggregation().aggregate(updates, [0.5 * x for x in p])

    assert torch.allclose(clipped.next_global["layer0"].A, 0.5 * full.next_global["layer0"].A)
    assert torch.allclose(clipped.delta_w["layer0"], 0.25 * full.delta_w["layer0"], atol=TOL)


def test_zero_weight_client_does_not_inflate_the_exact_rank_ceiling():
    """A rejected client must not count toward the |S|*r rank inflation we report."""
    updates = _make_updates(num_clients=4, r=3)
    kept = ExactDeltaWAggregation().aggregate(updates, _renormalized_over(updates, {0, 2, 3}))
    assert kept.aggregate_state["layer0"].rank == 3 * 3  # |S| * r, not N * r


def test_all_zero_weights_give_a_zero_aggregate_and_keep_the_previous_head():
    """A defence rejecting everyone is catastrophic but must not crash."""
    updates = _make_updates()
    zeros = [0.0] * len(updates)
    for strategy in (FedITNaiveAggregation, ExactDeltaWAggregation):
        agg = strategy().aggregate(updates, zeros)
        assert torch.allclose(
            agg.delta_w["layer0"], torch.zeros_like(agg.delta_w["layer0"]), atol=TOL
        )
    # the head falls back to "keep the previous global" rather than zeroing the classifier
    assert FederatedServer._aggregate_heads(updates, zeros) is None


# -- one q, all five weighted sites -----------------------------------------------------


def test_head_weighting_uses_q_and_is_renormalized():
    """q REPLACES num_examples for the head (not multiplied into it), and sums to 1 there.

    The head is an absolute parameter with no zero reference, so a uniform clip must not
    decay the classifier toward zero — that would manufacture an accuracy drop out of
    the defence's scale. Relative weights are the only meaning q has for the head.
    """
    updates = _make_updates()
    p = normalized_weights(updates)
    full = FederatedServer._aggregate_heads(updates, p)
    clipped = FederatedServer._aggregate_heads(updates, [0.5 * x for x in p])
    for k in full:
        assert torch.equal(full[k], clipped[k])

    # and it is q, not num_examples, that decides the mix: equal q => plain mean
    uniform = FederatedServer._aggregate_heads(updates, [1.0] * len(updates))
    expected = sum(u.head_state["classifier.weight"] for u in updates) / len(updates)
    assert torch.allclose(uniform["classifier.weight"], expected, atol=TOL)


def test_deltaw_discrepancy_uses_the_supplied_weights():
    """Site 3: the §1 gap is measured under the weights the round aggregated with."""
    updates = _make_updates()
    keep = {0, 2, 3}
    q = _renormalized_over(updates, keep)

    under_q = deltaw_discrepancy(updates, q)["layer0"]
    filtered = deltaw_discrepancy([u for u in updates if u.client_id in keep])["layer0"]
    assert under_q == pytest.approx(filtered, rel=1e-12)
    # and it is genuinely weight-dependent, so this is not a vacuous agreement
    assert under_q != pytest.approx(deltaw_discrepancy(updates)["layer0"], rel=1e-6)


def test_aggregate_spectrum_probe_follows_the_round_weights():
    """Site 4: the probe must describe the aggregate the server actually formed."""
    from fedlora_attacks.probes.base import ProbeContext
    from fedlora_attacks.probes.probes import ExactAggregateSpectrumProbe

    updates = _make_updates()
    keep = {0, 2, 3}
    q = _renormalized_over(updates, keep)
    aggregate = ExactDeltaWAggregation().aggregate(updates, q)

    values = ExactAggregateSpectrumProbe()(
        ProbeContext(updates, aggregate, round_idx=0, weights=q)
    ).values
    # rank ceiling reflects the SELECTED set, and the spectrum matches the real aggregate
    assert values["layer0"]["stack_dim"] == 3 * 3
    expected = torch.linalg.svdvals(aggregate.delta_w["layer0"])
    got = torch.tensor(values["layer0"]["singular_values"], dtype=torch.float64)
    assert torch.allclose(got[: expected.numel()], expected, atol=1e-10)


def test_probe_weights_default_to_the_protocol_weights():
    """weights=None must keep meaning p_i, so every pre-v3 probe call is unchanged."""
    from fedlora_attacks.probes.base import ProbeContext
    from fedlora_attacks.probes.probes import ExactAggregateSpectrumProbe

    updates = _make_updates()
    aggregate = ExactDeltaWAggregation().aggregate(updates)
    default = ExactAggregateSpectrumProbe()(ProbeContext(updates, aggregate, 0)).values
    explicit = ExactAggregateSpectrumProbe()(
        ProbeContext(updates, aggregate, 0, weights=normalized_weights(updates))
    ).values
    assert default["layer0"]["singular_values"] == explicit["layer0"]["singular_values"]


def test_defense_result_record_is_json_safe():
    """The round log carries the per-client STATISTICS, which are E-B's data source."""
    result = DefenseResult(
        weights=np.array([0.5, 0.5]),
        per_client_stats={0: {"deltaw_norm": 1.25}, 1: {"deltaw_norm": 0.75}},
        verdicts={0: ACCEPT, 1: ACCEPT},
    )
    record = result.to_record()
    assert record["weights"] == [0.5, 0.5]
    assert record["per_client_stats"]["0"]["deltaw_norm"] == 1.25
    assert record["verdicts"]["1"] == ACCEPT

    import json

    json.dumps(record)  # must not raise
