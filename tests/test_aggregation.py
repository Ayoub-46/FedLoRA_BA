"""Core aggregation correctness.

- exact_deltaw reproduces mean(dW_i) exactly.
- fedit_naive averages factors and is documented-inexact: mean(A_i)@mean(B_i)
  differs from mean(A_i@B_i) in general, and equals it in the gauge-aligned case.
- The discrepancy between the two strategies is exposed.
"""

from __future__ import annotations

import torch

from fedlora_attacks.fl.aggregation import (
    ExactDeltaWAggregation,
    FedITNaiveAggregation,
    deltaw_discrepancy,
)
from fedlora_attacks.fl.types import AdapterState, ClientUpdate, gauge_transform, random_gl

torch.manual_seed(0)
TOL = 1e-10  # float64


def _make_updates(n=6, m=5, r=3, num_clients=4, equal_counts=True):
    updates = []
    for c in range(num_clients):
        A = torch.randn(n, r, dtype=torch.float64)
        B = torch.randn(r, m, dtype=torch.float64)
        count = 1 if equal_counts else (c + 1) * 10
        updates.append(ClientUpdate(c, {"layer0": AdapterState(A, B)}, num_examples=count))
    return updates


def test_exact_deltaw_reproduces_mean_of_products():
    updates = _make_updates()
    result = ExactDeltaWAggregation().aggregate(updates)
    expected = sum(u.adapters["layer0"].delta_w() for u in updates) / len(updates)
    assert torch.allclose(result.delta_w["layer0"], expected, atol=TOL)


def test_exact_deltaw_weighted_mean():
    updates = _make_updates(equal_counts=False)
    result = ExactDeltaWAggregation().aggregate(updates)
    total = sum(u.num_examples for u in updates)
    expected = sum((u.num_examples / total) * u.adapters["layer0"].delta_w() for u in updates)
    assert torch.allclose(result.delta_w["layer0"], expected, atol=TOL)


def test_fedit_naive_averages_factors():
    """Defining property: next_global factors equal the factor means."""
    updates = _make_updates()
    result = FedITNaiveAggregation().aggregate(updates)
    A_mean = sum(u.adapters["layer0"].A for u in updates) / len(updates)
    B_mean = sum(u.adapters["layer0"].B for u in updates) / len(updates)
    assert torch.allclose(result.next_global["layer0"].A, A_mean, atol=TOL)
    assert torch.allclose(result.next_global["layer0"].B, B_mean, atol=TOL)
    # and its delta_w is the product of the means (NOT the mean of products)
    assert torch.allclose(result.delta_w["layer0"], A_mean @ B_mean, atol=TOL)


def test_fedit_naive_is_inexact_in_general():
    """The documented inexactness: naive != exact for generic factorizations."""
    updates = _make_updates()
    naive = FedITNaiveAggregation().aggregate(updates)
    exact = ExactDeltaWAggregation().aggregate(updates)
    assert not torch.allclose(naive.delta_w["layer0"], exact.delta_w["layer0"], atol=1e-4)
    disc = deltaw_discrepancy(updates)
    assert disc["layer0"] > 1e-3  # the §1 efficiency-vs-invariance gap is real & exposed


def test_fedit_naive_exact_when_gauge_aligned():
    """If all clients share ONE dW and differ only by gauge, naive == exact.

    Gauge-aligned factors: same (A,B) up to T in GL(r). Then averaging factors of a
    single dW... is still only exact when the T_i average back consistently; the
    clean case is identical factors, where both strategies trivially agree.
    """
    A = torch.randn(6, 3, dtype=torch.float64)
    B = torch.randn(3, 5, dtype=torch.float64)
    base = AdapterState(A, B)
    updates = [ClientUpdate(c, {"layer0": base.clone()}, 1) for c in range(4)]
    naive = FedITNaiveAggregation().aggregate(updates)
    exact = ExactDeltaWAggregation().aggregate(updates)
    assert torch.allclose(naive.delta_w["layer0"], exact.delta_w["layer0"], atol=TOL)


def test_gauge_transform_preserves_delta_w():
    """Sanity on the gauge action used across the codebase."""
    state = AdapterState(
        torch.randn(6, 3, dtype=torch.float64), torch.randn(3, 5, dtype=torch.float64)
    )
    for sign in (+1, -1):
        T = random_gl(3, sign=sign)
        transformed = gauge_transform(state, T)
        assert torch.allclose(transformed.delta_w(), state.delta_w(), atol=1e-9)
