"""Dirichlet partition: reproducibility, exact coverage, and heterogeneity behavior."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fedlora_attacks.data.partition import dirichlet_partition


def _labels(n=400, num_classes=2, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, num_classes, size=n).tolist()


def test_partition_is_reproducible():
    labels = _labels()
    p1 = dirichlet_partition(labels, num_clients=5, alpha=0.5, seed=42)
    p2 = dirichlet_partition(labels, num_clients=5, alpha=0.5, seed=42)
    assert p1.client_indices == p2.client_indices


def test_different_seed_changes_partition():
    labels = _labels()
    p1 = dirichlet_partition(labels, num_clients=5, alpha=0.5, seed=1)
    p2 = dirichlet_partition(labels, num_clients=5, alpha=0.5, seed=2)
    assert p1.client_indices != p2.client_indices


def test_exact_coverage_and_disjointness():
    labels = _labels(n=400)
    p = dirichlet_partition(labels, num_clients=7, alpha=0.3, seed=3)
    all_idx = [i for idx in p.client_indices for i in idx]
    assert sorted(all_idx) == list(range(len(labels)))  # every index exactly once
    assert sum(p.sizes) == len(labels)


def test_iid_alpha_inf_is_balanced():
    labels = _labels(n=600)
    p = dirichlet_partition(labels, num_clients=6, alpha=float("inf"), seed=4)
    sizes = np.array(p.sizes)
    # uniform split: sizes close to N/num_clients
    assert sizes.std() < 0.15 * sizes.mean()


def test_low_alpha_is_more_heterogeneous_than_high():
    """Smaller alpha -> more skewed per-client label distributions.

    Uses min_client_size=0: this asserts the RAW Dirichlet skew, which the min-size
    floor would partly condition away.
    """
    labels = _labels(n=800, num_classes=2)

    def label0_fraction_spread(alpha):
        p = dirichlet_partition(labels, num_clients=10, alpha=alpha, seed=5, min_client_size=0)
        fracs = []
        for h in p.label_histograms:
            total = sum(h.values())
            fracs.append((h.get(0, 0) / total) if total else 0.5)
        return np.std(fracs)

    assert label0_fraction_spread(0.1) > label0_fraction_spread(10.0)


def test_min_client_size_floor_is_enforced():
    """No client may be left too small to produce a genuine update.

    An empty client returns the broadcast global unchanged; the probes would then read
    a zero angle and a guaranteed det>0 off a client that never trained, poisoning the
    benign null. The floor is what prevents that.
    """
    labels = _labels(n=4000, num_classes=2)
    p = dirichlet_partition(labels, num_clients=5, alpha=0.1, seed=0, min_client_size=16)
    assert min(p.sizes) >= 16
    assert p.draw_attempts >= 1
    assert sum(p.sizes) == len(labels)  # resampling must not drop examples


def test_degenerate_partition_is_reachable_without_the_floor():
    """Teeth: the floor is doing work — without it, alpha=0.1 does starve a client."""
    labels = _labels(n=4000, num_classes=2)
    sizes = [
        min(dirichlet_partition(labels, 5, 0.1, seed=s, min_client_size=0).sizes) for s in range(20)
    ]
    assert min(sizes) < 16


def test_infeasible_floor_raises():
    labels = _labels(n=50)
    with pytest.raises(ValueError, match="cannot give"):
        dirichlet_partition(labels, num_clients=10, alpha=0.5, seed=0, min_client_size=16)


def test_partition_to_dict_is_serializable():
    labels = _labels()
    p = dirichlet_partition(labels, num_clients=3, alpha=float("inf"), seed=6)
    d = p.to_dict()
    assert d["alpha"] == "inf"
    assert not math.isnan(sum(d["sizes"]))
    assert len(d["label_histograms"]) == 3
