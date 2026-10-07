"""Seeded Dirichlet non-IID partitioning (CLAUDE.md invariant 4, plan §3).

Splits a labelled dataset across ``num_clients`` by drawing, per class, a
Dirichlet(alpha) vector of client proportions. Small alpha -> highly non-IID;
``alpha = inf`` -> exact uniform (IID). The partition is fully determined by
(labels, num_clients, alpha, seed) and is serialized into the run for provenance.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from ..utils.seeding import numpy_rng


@dataclass
class Partition:
    """A reproducible client-index partition + its provenance and label histograms."""

    client_indices: list[list[int]]
    num_clients: int
    alpha: float
    seed: int
    label_histograms: list[dict[int, int]] = field(default_factory=list)
    draw_attempts: int = 1
    """How many Dirichlet draws were needed to satisfy the min-size floor (provenance:
    a high count means the requested alpha barely admits a viable split)."""

    @property
    def sizes(self) -> list[int]:
        return [len(idx) for idx in self.client_indices]

    def to_dict(self) -> dict:
        return {
            "num_clients": self.num_clients,
            "alpha": self.alpha if math.isfinite(self.alpha) else "inf",
            "seed": self.seed,
            "sizes": self.sizes,
            "draw_attempts": self.draw_attempts,
            "label_histograms": [{str(k): v for k, v in h.items()} for h in self.label_histograms],
        }


# Smallest shard a client may receive. At alpha=0.1 with few clients the Dirichlet
# routinely hands someone 0 examples; such a client trains on nothing and returns the
# broadcast global unchanged. FedAvg weights it 0 so it cannot move the aggregate — but
# the PROBES would still count it as a client, and its frame is *identically* the
# consensus frame, injecting a fake zero-angle, guaranteed-det>0 observation into the
# benign null. Resampling until every client is non-trivial is the standard Dirichlet-FL
# practice and keeps the partition seeded and reproducible.
MIN_CLIENT_SIZE = 16
_MAX_PARTITION_ATTEMPTS = 100


def _draw_partition(
    labels, num_clients: int, alpha: float, rng, balance_sizes: bool = True
) -> list[list[int]]:
    """One Dirichlet draw, optionally size-balanced.

    ``balance_sizes`` is the standard non-IID-benchmark variant: before cutting each
    class, the Dirichlet proportions of clients already holding at least their share
    (N / num_clients) are zeroed. This caps SIZE skew while leaving LABEL skew — the
    heterogeneity actually under study — untouched. Without it, Dirichlet(0.1) hands one
    client most of the data and starves the rest, which both wastes draws against the
    min-size floor and makes the FedAvg weights degenerate.
    """
    client_indices: list[list[int]] = [[] for _ in range(num_clients)]
    target = len(labels) / num_clients
    for cls in np.unique(labels):
        idx_c = np.where(labels == cls)[0]
        rng.shuffle(idx_c)
        if math.isinf(alpha):
            proportions = np.full(num_clients, 1.0 / num_clients)
        else:
            proportions = rng.dirichlet(np.full(num_clients, alpha))
            if balance_sizes:
                room = np.array([len(ci) < target for ci in client_indices], dtype=float)
                if room.sum() > 0 and (proportions * room).sum() > 0:
                    proportions = proportions * room
            proportions = proportions / proportions.sum()
        # cut points along the shuffled class indices
        cuts = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        for i, split in enumerate(np.split(idx_c, cuts)):
            client_indices[i].extend(int(j) for j in split)
    return client_indices


def dirichlet_partition(
    labels: Sequence[int],
    num_clients: int,
    alpha: float,
    seed: int,
    min_client_size: int = MIN_CLIENT_SIZE,
) -> Partition:
    """Partition indices [0..N) across clients by per-class Dirichlet(alpha).

    ``alpha = float('inf')`` yields an (approximately) uniform IID split.
    Every index is assigned to exactly one client.

    Draws are repeated (from the same seeded stream, so the result stays reproducible)
    until every client holds at least ``min_client_size`` examples — see
    ``MIN_CLIENT_SIZE`` for why an empty client would corrupt the probe statistics.
    Pass ``min_client_size=0`` to allow degenerate splits.
    """
    if num_clients < 1:
        raise ValueError("num_clients must be >= 1")
    rng = numpy_rng(seed)
    labels = np.asarray(list(labels))
    feasible = min_client_size * num_clients <= len(labels)
    if min_client_size > 0 and not feasible:
        raise ValueError(
            f"cannot give {num_clients} clients >= {min_client_size} examples "
            f"from {len(labels)} total"
        )

    attempts = 0
    while True:
        attempts += 1
        client_indices = _draw_partition(labels, num_clients, alpha, rng)
        if min(len(idx) for idx in client_indices) >= min_client_size:
            break
        if attempts >= _MAX_PARTITION_ATTEMPTS:
            raise RuntimeError(
                f"no Dirichlet(alpha={alpha}) draw gave all {num_clients} clients "
                f">= {min_client_size} examples in {_MAX_PARTITION_ATTEMPTS} attempts; "
                "lower min_client_size, raise alpha, or use fewer clients"
            )

    for i in range(num_clients):
        rng.shuffle(client_indices[i])

    histograms = [dict(Counter(int(labels[j]) for j in idx)) for idx in client_indices]
    return Partition(client_indices, num_clients, float(alpha), seed, histograms, attempts)


def disjoint_label_partition(
    labels: Sequence[int],
    num_clients: int,
    seed: int,
    min_client_size: int = MIN_CLIENT_SIZE,
) -> Partition:
    """Every LABEL belongs to exactly one client — ownership, not a proportion split.

    WHY THIS EXISTS ALONGSIDE ``dirichlet_partition``. For Spider the label IS the
    database, and the realistic federation is that an organisation OWNS its databases and
    never sees anyone else's schemas. A Dirichlet does not produce that, and the gap is
    not subtle: measured at alpha=0.3 with N=10 over Spider's 140 databases, every client
    still saw 69-120 of them, the mean pairwise Jaccard of client database sets was 0.453,
    and NOT ONE database was held exclusively. The Dirichlet concentrates each class, but
    with 140 independent draws the clients' *sets* end up overlapping almost completely.

    Here each label is assigned whole. At N=10 over 140 databases a client holds ~14 of
    them and has literally never seen the other 126 — which is the point, and is a
    materially harder problem than the Dirichlet version.

    BALANCED BY EXAMPLE COUNT, not by label count. Spider's databases differ ~20x in how
    many questions they carry, so handing each client 14 arbitrary databases would swing
    shard sizes by the same factor and confound "federation is hard" with "this client had
    almost no data". Labels are therefore walked largest-first and each is given to the
    currently-smallest client — the standard greedy for multiway number partitioning,
    which is deterministic given the seeded shuffle that breaks ties.

    ``alpha`` is recorded as ``-1.0``: this scheme has no concentration parameter, and
    writing a number that looks like one into the provenance would invite a reader to
    compare it against a Dirichlet run's alpha as though they were the same axis.
    """
    labels = [int(x) for x in labels]
    if num_clients < 1:
        raise ValueError(f"num_clients must be >= 1, got {num_clients}")
    by_label: dict[int, list[int]] = {}
    for i, y in enumerate(labels):
        by_label.setdefault(y, []).append(i)
    if len(by_label) < num_clients:
        raise ValueError(
            f"disjoint_label_partition needs at least one label per client: "
            f"{len(by_label)} distinct labels for {num_clients} clients. With fewer labels "
            "than clients some client would own nothing and train on an empty shard."
        )

    rng = numpy_rng(seed)
    # Shuffle first so equal-sized labels do not always fall the same way, THEN sort by
    # size: the shuffle is what makes the seed matter, the sort is what balances.
    order = list(by_label)
    rng.shuffle(order)
    order.sort(key=lambda y: -len(by_label[y]))

    client_indices: list[list[int]] = [[] for _ in range(num_clients)]
    for y in order:
        target = min(range(num_clients), key=lambda c: len(client_indices[c]))
        client_indices[target].extend(by_label[y])

    for c in range(num_clients):
        rng.shuffle(client_indices[c])

    sizes = [len(x) for x in client_indices]
    if min(sizes) < min_client_size:
        raise ValueError(
            f"disjoint_label_partition left a client with {min(sizes)} examples "
            f"(floor {min_client_size}); sizes={sizes}. Too few labels, or one label "
            "dominates the corpus."
        )
    histograms = [dict(Counter(labels[i] for i in idx)) for idx in client_indices]
    return Partition(
        client_indices=client_indices,
        num_clients=num_clients,
        alpha=-1.0,  # not a Dirichlet; see the docstring
        seed=seed,
        label_histograms=histograms,
        draw_attempts=1,
    )
