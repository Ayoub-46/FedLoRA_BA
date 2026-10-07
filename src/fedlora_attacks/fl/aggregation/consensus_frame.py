"""Gauge-aware aggregation in a server-estimated shared frame.

The server picks ONE frame `U` per layer and re-expresses every client in it. With the
default weighted-mean coordinate rule:

    A~_i = U,   B~_i = U^T dW_i   ->   aggregate = U (sum_i q_i U^T dW_i)
                                                = P_U (sum_i q_i dW_i).

This aggregate is a LINEAR function of the `dW_j` alone, so `G_agg = G` and **Channel A
is closed**. Robust coordinate rules are generally nonlinear, but operate on comparable
`B~_i` after the same gauge-invariant frame and projection. We are therefore NOT claiming
GLoRA-style alignment fails.

**The precondition nobody has stated (T9), and it is the point of this module.** Closure
holds only if `U` is itself computed gauge-invariantly. Two estimators that agree on
benign traffic and diverge sharply under attack:

* ``projector``  — top-r subspace of `sum_i q_i A_i (A_i^T A_i)^{-1} A_i^T`. Each `P_i`
  depends on `col(A_i)` only, which the gauge fixes. **Invariant.** The escape works.
* ``magnitude``  — top-r subspace of `sum_i q_i A_i A_i^T`. Under a gauge this becomes
  `sum_i q_i A_i (T_i^T T_i)^{-1} A_i^T`, invariant only for `T_i in O(r)`.
  **Gauge-variant.** An attacker takes `T^-1 = c I` with `c` large, leaves `dW_i`
  untouched, and its inflated `A_i` dominates the estimate: the consensus subspace tilts
  toward the coalition and every subsequent projection `P_U dW_j` discards more of the
  honest updates.

So Channel A does not disappear under absolute gauge fixing — it RELOCATES into the
frame estimator. E-D uses ``projector`` (an escape, displacement at the fp64 floor);
E-I attacks both and the disagreement is the finding. Do not merge the two experiments,
or the escape claim becomes unconditional and false.

**Implementation.** `sum_i q_i M_i M_i^T` has the same top-r left singular subspace as
the stacked `[sqrt(q_1) M_1 | ... | sqrt(q_N) M_N]`, so both estimators are one thin SVD
of an `n x Nr` matrix — no `n x n` Gram is ever formed (768x768 per layer per round
would not be free). The estimators then differ by exactly one line: whether the blocks
are orthonormalised (`Q_i`, magnitude-free) or raw (`A_i`, magnitude-sensitive). That
one line is T9.

`U^T dW_i` is computed as `(U^T A_i) B_i` — `dW_i` is never materialised.  Once
projected, every client is represented by a tensor with the same meaning and shape, so
the final coordinate rule may be an ordinary weighted mean or a Byzantine-robust rule.
"""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

import torch

from ..types import AdapterDict, AdapterState, AggregateResult, ClientUpdate
from .base import AGGREGATIONS, Weights, _layer_names, active_clients

FRAME_ESTIMATORS = ("projector", "magnitude")
COORDINATE_RULES = ("mean", "median", "trimmed_mean", "geometric_median")


@runtime_checkable
class CoordinateAggregation(Protocol):
    """Aggregate same-frame client coordinate matrices into one server coordinate."""

    name: str

    def aggregate(self, coordinates: list[torch.Tensor], weights: list[float]) -> torch.Tensor: ...


def _weight_tensor(coordinates: list[torch.Tensor], weights: list[float]) -> torch.Tensor:
    if not coordinates:
        raise ValueError("cannot aggregate an empty coordinate list")
    if len(coordinates) != len(weights):
        raise ValueError(f"expected {len(coordinates)} coordinate weights, got {len(weights)}")
    if any(w < 0.0 for w in weights):
        raise ValueError("coordinate weights must be non-negative")
    return torch.as_tensor(weights, dtype=torch.float64, device=coordinates[0].device)


class WeightedMeanCoordinates:
    """The GLoRA rule: the unnormalised weighted sum of shared coordinates."""

    name = "mean"

    def aggregate(self, coordinates, weights):
        w = _weight_tensor(coordinates, weights)
        stacked = torch.stack([z.to(torch.float64) for z in coordinates])
        return torch.einsum("i,i...->...", w, stacked)


class CoordinateMedian:
    """Weighted coordinate-wise median, rescaled by the defense's surviving mass."""

    name = "median"

    def aggregate(self, coordinates, weights):
        w = _weight_tensor(coordinates, weights)
        values = torch.stack([z.to(torch.float64) for z in coordinates])
        mass = w.sum()
        if mass <= 0:
            return torch.zeros_like(values[0])
        order = torch.argsort(values, dim=0)
        expanded_w = w.reshape((-1,) + (1,) * (values.ndim - 1)).expand_as(values)
        sorted_w = torch.gather(expanded_w, 0, order)
        first_at_half = torch.argmax((sorted_w.cumsum(0) >= mass / 2).to(torch.int64), dim=0)
        median = torch.gather(values, 0, order).gather(0, first_at_half.unsqueeze(0)).squeeze(0)
        # q is deliberately not normalized elsewhere in the framework. Preserve that
        # convention so uniformly clipping all clients still shrinks the adapter update.
        return mass * median


class TrimmedMeanCoordinates:
    """Weighted coordinate-wise trimmed mean with fractional tail-mass removal."""

    name = "trimmed_mean"

    def __init__(self, trim_fraction: float = 0.1) -> None:
        if not 0.0 <= trim_fraction < 0.5:
            raise ValueError("trim_fraction must be in [0, 0.5)")
        self.trim_fraction = float(trim_fraction)

    def aggregate(self, coordinates, weights):
        w = _weight_tensor(coordinates, weights)
        values = torch.stack([z.to(torch.float64) for z in coordinates])
        mass = w.sum()
        if mass <= 0:
            return torch.zeros_like(values[0])
        order = torch.argsort(values, dim=0)
        sorted_values = torch.gather(values, 0, order)
        expanded_w = w.reshape((-1,) + (1,) * (values.ndim - 1)).expand_as(values)
        sorted_w = torch.gather(expanded_w, 0, order)

        # Remove exactly `trim_fraction` of the total weight at each end. Fractional
        # boundary weights make the result meaningful for unequal FedAvg/defense weights.
        trim = self.trim_fraction * mass
        before = sorted_w.cumsum(0) - sorted_w
        after = torch.flip(torch.flip(sorted_w, (0,)).cumsum(0), (0,)) - sorted_w
        kept_w = torch.minimum(sorted_w, torch.clamp(before + sorted_w - trim, min=0.0))
        kept_w = torch.minimum(kept_w, torch.clamp(after + sorted_w - trim, min=0.0))
        kept_mass = kept_w.sum(0)
        centre = (kept_w * sorted_values).sum(0) / kept_mass.clamp_min(
            torch.finfo(torch.float64).eps
        )
        return mass * centre


class GeometricMedianCoordinates:
    """Weighted geometric median of flattened shared coordinates (Weiszfeld)."""

    name = "geometric_median"

    def __init__(self, max_iterations: int = 50, tolerance: float = 1e-6) -> None:
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if tolerance <= 0:
            raise ValueError("tolerance must be positive")
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)

    def aggregate(self, coordinates, weights):
        w = _weight_tensor(coordinates, weights)
        points = torch.stack([z.to(torch.float64).reshape(-1) for z in coordinates])
        mass = w.sum()
        if mass <= 0:
            return torch.zeros_like(coordinates[0], dtype=torch.float64)
        estimate = torch.einsum("i,ij->j", w / mass, points)
        eps = torch.finfo(torch.float64).eps
        for _ in range(self.max_iterations):
            distances = torch.linalg.vector_norm(points - estimate, dim=1)
            coincident = distances <= eps
            if coincident.any():
                # A data point is a geometric median when its coincident mass dominates
                # the pull from all other points; selecting it is also the stable limit
                # of Weiszfeld rather than dividing by zero.
                idx = int(torch.argmax(w * coincident).item())
                candidate = points[idx]
                others = ~coincident
                inverse_distances = w[others] / distances[others]
                pull = (inverse_distances.unsqueeze(1) * (points[others] - candidate)).sum(0)
                pull_norm = torch.linalg.vector_norm(pull)
                coincident_mass = w[coincident].sum()
                if pull_norm <= coincident_mass:
                    estimate = candidate
                    break
                ordinary = torch.einsum(
                    "i,ij->j", inverse_distances / inverse_distances.sum(), points[others]
                )
                alpha = coincident_mass / pull_norm
                updated = alpha * candidate + (1.0 - alpha) * ordinary
            else:
                inv = w / distances
                updated = torch.einsum("i,ij->j", inv / inv.sum(), points)
            relative = torch.linalg.vector_norm(updated - estimate) / torch.linalg.vector_norm(
                estimate
            ).clamp_min(eps)
            estimate = updated
            if relative <= self.tolerance:
                break
        return (mass * estimate).reshape_as(coordinates[0])


def build_coordinate_aggregation(
    name: str,
    *,
    trim_fraction: float = 0.1,
    max_iterations: int = 50,
    tolerance: float = 1e-6,
) -> CoordinateAggregation:
    if name == "mean":
        return WeightedMeanCoordinates()
    if name == "median":
        return CoordinateMedian()
    if name == "trimmed_mean":
        return TrimmedMeanCoordinates(trim_fraction)
    if name == "geometric_median":
        return GeometricMedianCoordinates(max_iterations, tolerance)
    raise ValueError(f"coordinate_rule must be one of {COORDINATE_RULES}, got {name!r}")


def estimate_frame(
    states: list[AdapterState], weights: list[float], estimator: str, r: int
) -> torch.Tensor:
    """Top-r consensus frame U (n x r), orthonormal, from the clients' column factors.

    ``projector`` orthonormalises each A_i first, so only col(A_i) — a gauge invariant —
    reaches the estimate. ``magnitude`` uses A_i raw, so the transmitted scale reaches it
    too. Everything in fp64: this decides the subspace every update is projected onto.
    """
    if estimator not in FRAME_ESTIMATORS:
        raise ValueError(f"estimator must be one of {FRAME_ESTIMATORS}, got {estimator!r}")
    blocks = []
    for state, w in zip(states, weights, strict=True):
        A = state.A.to(dtype=torch.float64)
        if estimator == "projector":
            # A (A^T A)^-1 A^T == Q Q^T for the thin QR A = QR: the orthogonal projector
            # onto col(A), computed without inverting a Gram.
            A, _ = torch.linalg.qr(A, mode="reduced")
        blocks.append(math.sqrt(w) * A)  # sqrt so the implied Gram carries weight w
    stacked = torch.cat(blocks, dim=1)  # (n, N*r)
    U, _, _ = torch.linalg.svd(stacked, full_matrices=False)
    return U[:, : min(r, U.shape[1])].contiguous()


def project_coordinates(states: list[AdapterState], frame: torch.Tensor) -> list[torch.Tensor]:
    """Express each intrinsic update in ``frame`` without materializing ``dW``."""
    return [
        (frame.transpose(-2, -1) @ state.A.to(torch.float64)) @ state.B.to(torch.float64)
        for state in states
    ]


def _pad_rank(state: AdapterState, target_rank: int) -> AdapterState:
    """Zero-pad a lower-rank server update for a fixed-rank homogeneous PEFT model."""
    if state.rank == target_rank:
        return state
    if state.rank > target_rank:
        raise ValueError(f"cannot pad rank {state.rank} down to {target_rank}")
    A = state.A.new_zeros((state.A.shape[0], target_rank))
    B = state.B.new_zeros((target_rank, state.B.shape[1]))
    A[:, : state.rank] = state.A
    B[: state.rank] = state.B
    return AdapterState(A.contiguous(), B.contiguous())


@AGGREGATIONS.register("consensus_frame")
class ConsensusFrameAggregation:
    """Absolute gauge fixing to an estimated frame (plan v3 T9, experiments E-D / E-I).

    ``estimator='projector'`` is the escape; ``'magnitude'`` is the cautionary
    construction. Both are needed and they are one line apart. ``coordinate_rule='mean'``
    is the GLoRA projected mean; optional robust rules consume the same projected tensors.
    """

    name = "consensus_frame"

    def __init__(
        self,
        estimator: str = "projector",
        server_rank: int | None = None,
        coordinate_rule: str = "mean",
        trim_fraction: float = 0.1,
        geometric_median_max_iterations: int = 50,
        geometric_median_tolerance: float = 1e-6,
    ) -> None:
        if estimator not in FRAME_ESTIMATORS:
            raise ValueError(f"estimator must be one of {FRAME_ESTIMATORS}, got {estimator!r}")
        self.estimator = estimator
        if server_rank is not None and server_rank <= 0:
            raise ValueError("server_rank must be positive or None")
        self.server_rank = server_rank
        self.coordinate_aggregation = build_coordinate_aggregation(
            coordinate_rule,
            trim_fraction=trim_fraction,
            max_iterations=geometric_median_max_iterations,
            tolerance=geometric_median_tolerance,
        )

    def aggregate(
        self, updates: list[ClientUpdate], weights: Weights | None = None
    ) -> AggregateResult:
        updates, weights = active_clients(updates, weights)
        aggregate_state: AdapterDict = {}
        next_global: AdapterDict = {}
        layer_diagnostics = {}
        for layer in _layer_names(updates):
            states = [u.adapters[layer] for u in updates]
            model_rank = states[0].rank
            if any(state.rank != model_rank for state in states):
                raise ValueError("consensus_frame currently requires homogeneous client ranks")
            rank = model_rank if self.server_rank is None else self.server_rank
            if rank > model_rank:
                raise ValueError(
                    f"server_rank={rank} exceeds homogeneous model rank {model_rank}; "
                    "increase model.lora_r or use server_rank <= model.lora_r"
                )
            U = estimate_frame(states, weights, self.estimator, rank)

            # A_agg = U, NOT (sum_i q_i) U. Coordinate rules themselves preserve the
            # surviving q mass, so putting it into U too would apply clipping twice. For
            # the default mean this gives exactly P_U(sum_i q_i dW_i).
            coordinates = project_coordinates(states, U)
            B_agg = self.coordinate_aggregation.aggregate(coordinates, weights)
            dtype, device = states[0].A.dtype, states[0].A.device
            state = AdapterState(
                U.to(device=device, dtype=dtype).contiguous(),
                B_agg.to(device=device, dtype=dtype).contiguous(),
            )
            # Already rank r by construction: absolute gauge fixing needs no truncation,
            # which is its efficiency advantage over exact_deltaw. Its cost is instead
            # what P_U discards from the honest updates (E-I's second row).
            aggregate_state[layer] = state
            next_global[layer] = _pad_rank(state, model_rank)

            # Orthogonal projection contracts Frobenius norm. This retained-norm
            # diagnostic separates defense/aggregation effects from a tight rank budget.
            from ...probes.geometry import deltaw_frobenius_norm
            from ..types import stacked_factors

            exact = stacked_factors(states, list(weights))
            exact_norm = deltaw_frobenius_norm(exact.A, exact.B)
            projected_mean = AdapterState(
                U, WeightedMeanCoordinates().aggregate(coordinates, weights)
            )
            projected_mean_norm = deltaw_frobenius_norm(projected_mean.A, projected_mean.B)
            aggregate_norm = deltaw_frobenius_norm(state.A, state.B)
            layer_diagnostics[layer] = {
                "server_rank": rank,
                "model_rank": model_rank,
                "exact_aggregate_norm": exact_norm,
                "projected_mean_norm": projected_mean_norm,
                "retained_norm_ratio": (
                    projected_mean_norm / exact_norm if exact_norm > 0 else 1.0
                ),
                "coordinate_aggregate_norm": aggregate_norm,
            }
        return AggregateResult(
            aggregate_state=aggregate_state,
            next_global=next_global,
            strategy=self.name,
            diagnostics={
                "frame_estimator": self.estimator,
                "coordinate_rule": self.coordinate_aggregation.name,
                "effective_weight_sum": float(sum(weights)),
                "layers": layer_diagnostics,
            },
        )


@AGGREGATIONS.register("consensus_frame_magnitude")
class MagnitudeConsensusFrameAggregation(ConsensusFrameAggregation):
    """The magnitude-sensitive variant, registered so a config can select it by name.

    NOT an escape. Registered separately only because the aggregation registry selects by
    name with no arguments; E-I compares it against ``consensus_frame``.
    """

    name = "consensus_frame_magnitude"

    def __init__(self, **kwargs) -> None:
        super().__init__(estimator="magnitude", **kwargs)
