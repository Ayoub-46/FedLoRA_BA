"""Core domain types and the gauge action (CLAUDE.md "Domain model").

Notation (plan convention, §0):
    A in R^{n x r}  (output / column factor),
    B in R^{r x m}  (input / row factor),
    dW = A @ B in R^{n x m}   (the effective update; the *complete* gauge invariant).

Gauge action of T in GL(r):  (A, B) -> (A @ inv(T), T @ B), which leaves dW fixed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

Tensor = torch.Tensor
LayerName = str


@dataclass
class AdapterState:
    """One LoRA adapter's factors for one adapted layer, in plan convention.

    PEFT mapping (see models/lora_wrapper.py): A = scaling * lora_B.weight (n x r),
    B = lora_A.weight (r x m). dW = A @ B has the shape of the base weight (out x in).
    """

    A: Tensor  # (n, r)
    B: Tensor  # (r, m)

    def __post_init__(self) -> None:
        if self.A.ndim != 2 or self.B.ndim != 2:
            raise ValueError("A and B must be 2-D")
        if self.A.shape[1] != self.B.shape[0]:
            raise ValueError(
                f"rank mismatch: A is {tuple(self.A.shape)}, B is {tuple(self.B.shape)}"
            )

    @property
    def rank(self) -> int:
        return self.A.shape[1]

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of dW = (n, m)."""
        return (self.A.shape[0], self.B.shape[1])

    def delta_w(self) -> Tensor:
        return self.A @ self.B

    def clone(self) -> AdapterState:
        return AdapterState(self.A.clone(), self.B.clone())


AdapterDict = dict[LayerName, AdapterState]


def stacked_factors(states: list[AdapterState], weights: list[float] | None = None) -> AdapterState:
    """Stack per-client factors into ONE exact factorization of the weighted mean.

    Given (A_i, B_i) and weights w_i,
        [w_1 A_1 | ... | w_N A_N] @ [B_1 ; ... ; B_N]  ==  sum_i w_i A_i B_i
    exactly, with inner dimension N*r. This is how the exact aggregate is represented
    and spectrally probed: rank up to N*r, never materialized (CLAUDE.md invariant 3 —
    the exact aggregate is NOT rank-r, and that inflation is a first-class measurement).
    """
    if weights is None:
        weights = [1.0] * len(states)
    A = torch.cat([w * s.A for w, s in zip(weights, states, strict=True)], dim=1)
    B = torch.cat([s.B for s in states], dim=0)
    return AdapterState(A, B)


def gauge_transform(state: AdapterState, T: Tensor) -> AdapterState:
    """Apply T in GL(r):  (A, B) -> (A @ inv(T), T @ B).  Leaves dW invariant.

    The inverse and both products are taken in fp64 and only the RESULT is cast back to
    the factors' dtype. dW is invariant in exact arithmetic but not in floating point:
    the round-trip error is ~eps * cond(T), so inverting an ill-conditioned T in fp32
    (which is what the factors are) would amplify eps32 = 1e-7 by cond(T) twice over.
    Measured at cond(T) = 1e4 on bert-scale factors: 2.4e-4 relative in fp32 throughout
    versus 4.2e-5 with this fp64 path — a ~6x improvement for free.

    What the fp64 path CANNOT fix is that the transmitted factors are stored in the
    model's dtype. The residual floor is eps(dtype) * cond(T) regardless, which is why
    an attack sweeping cond(T) must MEASURE its own dW displacement rather than assume
    zero (see attacks/x1_gauge.py) — at cond(T) = 1e6 in fp32 it reaches 4e-3 relative,
    large enough for a tight norm bound to notice for purely numerical reasons.
    """
    if T.shape != (state.rank, state.rank):
        raise ValueError(f"T must be ({state.rank}, {state.rank}), got {tuple(T.shape)}")
    T64 = T.to(device=state.A.device, dtype=torch.float64)  # r x r: negligible on GPU
    T_inv = torch.linalg.inv(T64)
    A = (state.A.to(dtype=torch.float64) @ T_inv).to(dtype=state.A.dtype)
    B = (T64 @ state.B.to(dtype=torch.float64)).to(dtype=state.B.dtype)
    return AdapterState(A, B)


def random_gl(
    r: int,
    *,
    generator: torch.Generator | None = None,
    sign: int = +1,
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    """A random well-conditioned matrix in GL(r) with a chosen det sign.

    ``sign`` in {+1, -1} selects the connected component (det>0 / det<0), used by
    the gauge-invariance property test which must cover BOTH components.
    """
    if sign not in (+1, -1):
        raise ValueError("sign must be +1 or -1")
    while True:
        T = torch.randn(r, r, generator=generator, dtype=dtype)
        det = torch.linalg.det(T)
        if det.abs() > 1e-3:  # keep it well away from the singular locus
            break
    if (det > 0) != (sign > 0):
        T[0] = -T[0]  # flip one row -> flips det sign
    return T


@dataclass
class ClientUpdate:
    """A client's post-local-training adapters plus bookkeeping for aggregation.

    ``head_state`` carries the non-LoRA trainable parameters (the classification head /
    modules_to_save). LoRA adapters are aggregated by the pluggable strategy; the head
    is federated by plain weighted FedAvg in the server so main-task accuracy is
    meaningful. The head is NOT part of the gauge-bundle geometry and is never probed.
    """

    client_id: int
    adapters: AdapterDict
    num_examples: int
    head_state: dict[str, Tensor] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GlobalState:
    """The server's broadcast state at the start of a round."""

    round: int
    adapters: AdapterDict


@dataclass
class AggregateResult:
    """Output of an aggregation strategy.

    ``aggregate_state`` is the canonical MEASURED update per layer (ground truth for
    probes), kept as FACTORS and never materialized (CLAUDE.md "Spectral computation").
    Its inner dimension is the true rank bound:
      - ``fedit_naive``  -> r     (A_agg = sum w_i A_i, B_agg = sum w_i B_i)
      - ``exact_deltaw`` -> N*r   (STACKED factors; exact mean of products)
    That N*r inflation is invariant 3 made concrete: the exact aggregate is NOT rank-r.

    ``next_global`` is the rank-r factorization actually redistributed to clients. For
    ``fedit_naive`` it coincides with ``aggregate_state``; for ``exact_deltaw`` it is
    the rank-r truncation of the exact mean — the efficiency-vs-invariance gap (§1),
    so probes see the true object while the redistributed factors carry the error.
    """

    aggregate_state: AdapterDict
    next_global: AdapterDict
    strategy: str
    diagnostics: dict[str, Any] = field(default_factory=dict)
    """JSON-safe strategy-specific measurements for the current round.

    Aggregators normally need return only the state that is measured and broadcast.
    Geometry-aware strategies additionally need to expose approximation/conditioning
    information (for example shared-frame projection loss) without coupling the server
    to their implementation.  The server writes this mapping into the round record.
    """

    @property
    def delta_w(self) -> dict[LayerName, Tensor]:
        """Materialize the aggregate dW per layer. TEST/TOY USE ONLY.

        The probe path goes through the factors; this exists so aggregation tests can
        assert each strategy's defining property against the literal product.
        """
        return {layer: state.delta_w() for layer, state in self.aggregate_state.items()}
