"""FLTrust-DeltaW: trust bootstrapping on effective LoRA updates.

The original FLTrust rule scores each client model update against a clean server
update, projects client magnitudes to the server-update norm, and averages under
ReLU-clipped cosine trust.  In this project the measurable object is the effective
LoRA update ``DeltaW = A @ B``.  This port therefore computes every statistic from
the per-layer ``DeltaW`` objects, without materialising them.

The framework requires one non-negative weight vector to drive every aggregation
site.  We fold FLTrust's trust and magnitude factors into raw weights and normalize
them to sum to one.  This preserves their relative contributions and, importantly,
keeps the server's absolute adapter-state aggregation well-defined.  It differs from
the paper's unnormalised effective coefficients by one round-wide positive scalar;
that distinction is reported as ``formulation='effective_deltaw_unit_sum'``.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256

import numpy as np
import torch

from ..data.base import FederatedData
from ..data.partition import Partition
from ..fl.types import ClientUpdate
from ..probes.geometry import cross_gram, deltaw_frobenius_norm
from .base import ACCEPT, DEFENSES, REJECT, BaseDefense, DefenseConfig, DefenseResult

PAPER = "paper_uniform"
FEDAVG = "fedavg"
WEIGHTING_MODES = (PAPER, FEDAVG)

STRATIFIED = "stratified"
UNIFORM = "uniform"
ROOT_SAMPLING_MODES = (STRATIFIED, UNIFORM)


@dataclass
class FLTrustConfig(DefenseConfig):
    name: str = "fltrust"
    root_size: int = 100
    root_seed: int = 0
    root_sampling: str = STRATIFIED
    client_weighting: str = PAPER
    """Base measure used before unit-sum normalization.

    ``paper_uniform`` reproduces FLTrust's equal-client convention. ``fedavg`` is
    an explicit FedLoRA ablation that additionally multiplies by the protocol's
    data-size weights. It is not the NDSS paper's default rule.
    """

    def __post_init__(self) -> None:
        if self.root_size <= 0:
            raise ValueError(f"root_size must be positive, got {self.root_size}")
        if self.root_sampling not in ROOT_SAMPLING_MODES:
            raise ValueError(
                f"root_sampling must be one of {ROOT_SAMPLING_MODES}, "
                f"got {self.root_sampling!r}"
            )
        if self.client_weighting not in WEIGHTING_MODES:
            raise ValueError(
                f"client_weighting must be one of {WEIGHTING_MODES}, "
                f"got {self.client_weighting!r}"
            )


@dataclass
class TrustedRootShard:
    """A server-owned training shard reserved from the simulation's train split."""

    indices: tuple[int, ...]
    texts: list[str]
    labels: list[int]
    responses: list[str] | None = None
    images: object | None = None

    def to_record(self, sampling: str, seed: int) -> dict:
        encoded = ",".join(str(i) for i in self.indices).encode()
        return {
            "size": len(self.indices),
            "seed": int(seed),
            "sampling": sampling,
            "indices_sha256": sha256(encoded).hexdigest(),
            "label_histogram": {str(k): int(v) for k, v in sorted(Counter(self.labels).items())},
        }


def _stratified_indices(labels: list[int], size: int, rng: np.random.Generator) -> list[int]:
    """A deterministic proportional allocation with largest-remainder rounding."""

    by_label: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        by_label.setdefault(int(label), []).append(idx)
    for indices in by_label.values():
        rng.shuffle(indices)

    n = len(labels)
    exact = {label: size * len(indices) / n for label, indices in by_label.items()}
    take = {label: min(len(by_label[label]), int(math.floor(v))) for label, v in exact.items()}
    remaining = size - sum(take.values())
    order = sorted(
        by_label,
        key=lambda label: (exact[label] - math.floor(exact[label]), len(by_label[label]), -label),
        reverse=True,
    )
    while remaining:
        progressed = False
        for label in order:
            if take[label] < len(by_label[label]):
                take[label] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:  # guarded by size < n, but keeps failure explicit
            raise RuntimeError("could not allocate the requested trusted-root sample")

    selected = [idx for label in sorted(by_label) for idx in by_label[label][: take[label]]]
    rng.shuffle(selected)
    return selected


def reserve_trusted_root(
    fed: FederatedData, size: int, seed: int, sampling: str = STRATIFIED
) -> tuple[FederatedData, TrustedRootShard]:
    """Reserve clean root examples and remove them from every client partition.

    The underlying ``TaskData`` remains shared; only the partition is replaced.  This
    keeps all existing loaders untouched while ensuring a root example is never also a
    client example.  A reservation that empties a client fails rather than changing the
    number of participants silently.
    """

    total = fed.data.num_train
    if size <= 0 or size >= total:
        raise ValueError(f"root_size must be in [1, {total - 1}], got {size}")
    if sampling not in ROOT_SAMPLING_MODES:
        raise ValueError(f"root_sampling must be one of {ROOT_SAMPLING_MODES}, got {sampling!r}")

    rng = np.random.default_rng(seed)
    if sampling == STRATIFIED:
        selected = _stratified_indices(fed.data.train_labels, size, rng)
    else:
        selected = [int(i) for i in rng.choice(total, size=size, replace=False)]
    selected_set = set(selected)

    client_indices = [
        [idx for idx in indices if idx not in selected_set]
        for indices in fed.partition.client_indices
    ]
    empty = [cid for cid, indices in enumerate(client_indices) if not indices]
    if empty:
        raise ValueError(
            f"trusted-root reservation emptied client(s) {empty}; reduce defense.root_size"
        )
    labels = fed.data.train_labels
    histograms = [dict(Counter(int(labels[i]) for i in indices)) for indices in client_indices]
    partition = Partition(
        client_indices=client_indices,
        num_clients=fed.partition.num_clients,
        alpha=fed.partition.alpha,
        seed=fed.partition.seed,
        label_histograms=histograms,
        draw_attempts=fed.partition.draw_attempts,
    )

    root_labels = [int(labels[i]) for i in selected]
    if fed.data.modality == "image":
        shard = TrustedRootShard(
            indices=tuple(selected),
            texts=[],
            labels=root_labels,
            images=fed.data.train_images.subset(selected),
        )
    else:
        responses = (
            [fed.data.train_responses[i] for i in selected]
            if fed.data.train_responses is not None
            else None
        )
        shard = TrustedRootShard(
            indices=tuple(selected),
            texts=[fed.data.train_texts[i] for i in selected],
            labels=root_labels,
            responses=responses,
        )
    return FederatedData(fed.data, partition), shard


def fltrust_deltaw_statistics(
    updates: list[ClientUpdate], trusted_update: ClientUpdate
) -> tuple[list[float], list[float], float]:
    """Return client-to-root cosines, client norms, and the network-wide root norm."""

    if not updates:
        return [], [], 0.0
    layers = list(trusted_update.adapters)
    if not layers:
        raise ValueError("the trusted root update contains no LoRA adapters")
    for update in updates:
        if list(update.adapters) != layers:
            raise ValueError("trusted root and clients must share the same layer set/order")

    root_sq = 0.0
    client_sq = np.zeros(len(updates), dtype=np.float64)
    inner = np.zeros(len(updates), dtype=np.float64)
    for layer in layers:
        root = trusted_update.adapters[layer]
        root_sq += deltaw_frobenius_norm(root.A, root.B) ** 2
        for idx, update in enumerate(updates):
            state = update.adapters[layer]
            client_sq[idx] += deltaw_frobenius_norm(state.A, state.B) ** 2
            inner[idx] += float(torch.trace(cross_gram(state.A, state.B, root.A, root.B)))

    root_norm = math.sqrt(max(root_sq, 0.0))
    norms = np.sqrt(np.clip(client_sq, 0.0, None))
    cosines = np.full(len(updates), math.nan, dtype=np.float64)
    if root_norm > 0:
        nonzero = norms > 0
        cosines[nonzero] = inner[nonzero] / (norms[nonzero] * root_norm)
        cosines[nonzero] = np.clip(cosines[nonzero], -1.0, 1.0)
    return [float(v) for v in cosines], [float(v) for v in norms], root_norm


@DEFENSES.register("fltrust")
class FLTrustDefense(BaseDefense):
    """Root-anchored direction and magnitude reweighting on effective ``DeltaW``."""

    def __init__(self, config: FLTrustConfig) -> None:
        super().__init__(config)
        self.cfg = config

    def apply(self, updates, context) -> DefenseResult:
        if not updates:
            return DefenseResult(weights=np.asarray([], dtype=np.float64))
        root = context.trusted_update
        if root is None:
            raise ValueError("FLTrust requires a trusted server update in DefenseContext")

        cosines, norms, root_norm = fltrust_deltaw_statistics(updates, root)
        if not math.isfinite(root_norm) or root_norm <= 0.0:
            raise ValueError("FLTrust requires a finite, non-zero trusted DeltaW update")

        cosine_array = np.asarray(cosines, dtype=np.float64)
        norm_array = np.asarray(norms, dtype=np.float64)
        trust = np.clip(np.nan_to_num(cosine_array, nan=0.0), 0.0, 1.0)
        scale = np.divide(
            root_norm,
            norm_array,
            out=np.zeros_like(norm_array),
            where=np.isfinite(norm_array) & (norm_array > 0.0),
        )
        base = (
            np.asarray(context.base_weights, dtype=np.float64)
            if self.cfg.client_weighting == FEDAVG
            else np.ones(len(updates), dtype=np.float64)
        )
        raw = base * trust * scale
        mass = float(raw.sum())
        if not math.isfinite(mass) or mass <= 0.0:
            raise ValueError("FLTrust found no client with positive trust in the root update")
        weights = raw / mass
        # Treat unit mass as part of the public contract, not a numerical accident.
        weights /= weights.sum()

        stats, verdicts = {}, {}
        for idx, update in enumerate(updates):
            stats[update.client_id] = {
                "cosine_to_root": cosines[idx],
                "trust_score": float(trust[idx]),
                "deltaw_norm": norms[idx],
                "root_deltaw_norm": root_norm,
                "normalization_scale": float(scale[idx]),
                "raw_weight": float(raw[idx]),
                "effective_weight": float(weights[idx]),
                "base_weight": float(context.base_weights[idx]),
            }
            verdicts[update.client_id] = ACCEPT if trust[idx] > 0.0 else REJECT

        return DefenseResult(
            weights=weights,
            per_client_stats=stats,
            verdicts=verdicts,
            round_stats={
                "formulation": "effective_deltaw_unit_sum",
                "client_weighting": self.cfg.client_weighting,
                "root_deltaw_norm": root_norm,
                "trust_sum": float(trust.sum()),
                "raw_weight_sum": mass,
                "weight_sum": float(weights.sum()),
                "num_positive_trust": int((trust > 0.0).sum()),
                "root_train_loss": root.metadata.get("train_loss", math.nan),
                "root_size": int(root.num_examples),
            },
        )
