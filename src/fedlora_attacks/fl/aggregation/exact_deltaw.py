"""exact_deltaw: exact mean of the effective updates (the gauge-invariant target).

Defining property (asserted in tests/test_aggregation.py):
    dW_agg = sum_i w_i (A_i @ B_i)   exactly.

This is the quantity a *security*-correct aggregator should act on, but it lives in
R^{n x m} with rank up to N*r, which efficiency forbids transmitting or materializing
at scale — the §1 tension. We therefore represent it EXACTLY and cheaply by STACKED
factors (CLAUDE.md "Spectral computation" / invariant 3):

    A_stack = [w_1 A_1 | ... | w_N A_N]   (n x Nr)
    B_stack = [B_1 ; ... ; B_N]           (Nr x m)
    A_stack @ B_stack = sum_i w_i A_i B_i  exactly, never formed.

For redistribution we truncate back to the adapter rank r, computed from the Nr x Nr
core rather than from a materialized n x m matrix — same Eckart-Young optimum, at
O((n+m)(Nr)^2) instead of O(n m min(n,m)).
"""

from __future__ import annotations

import torch

from ..types import AdapterState, AggregateResult, ClientUpdate, stacked_factors
from .base import AGGREGATIONS, Weights, _layer_names, active_clients


def rank_r_factorization(delta_w: torch.Tensor, r: int) -> AdapterState:
    """Best rank-r factorization of a MATERIALIZED ``delta_w`` (toy/test path).

    Returns (A, B) with A = U_r @ diag(S_r), B = Vh_r, so A @ B is the optimal rank-r
    approximation. Production code uses ``rank_r_from_factors``, which never forms dW.
    """
    U, S, Vh = torch.linalg.svd(delta_w, full_matrices=False)
    r = min(r, S.shape[0])
    A = U[:, :r] * S[:r].unsqueeze(0)
    B = Vh[:r, :]
    return AdapterState(A.contiguous(), B.contiguous())


def rank_r_from_factors(A: torch.Tensor, B: torch.Tensor, r: int) -> AdapterState:
    """Optimal rank-r truncation of dW = A @ B, computed from the factors alone.

    With A = Qa Ra and B^T = Qb Rb, dW = Qa (Ra Rb^T) Qb^T. An SVD of the small core
    Ra Rb^T = U S V^T gives the exact SVD of dW as (Qa U) S (Qb V)^T, so truncating the
    core truncates dW — Eckart-Young optimal, with no n x m matrix ever formed.

    The result is the CANONICAL (SVD-gauge-fixed) frame for this aggregate: A carries
    the singular values, B is orthonormal-rowed. That canonicality is why it doubles as
    the reference frame the det-sign probe measures client orientations against.
    """
    dtype = A.dtype
    A64 = A.detach().to(device="cpu", dtype=torch.float64)
    B64 = B.detach().to(device="cpu", dtype=torch.float64)
    Qa, Ra = torch.linalg.qr(A64, mode="reduced")
    Qb, Rb = torch.linalg.qr(B64.transpose(-2, -1), mode="reduced")
    U, S, Vh = torch.linalg.svd(Ra @ Rb.transpose(-2, -1))
    k = min(r, S.shape[0])
    A_r = (Qa @ U[:, :k]) * S[:k].unsqueeze(0)  # (n, k)
    B_r = (Qb @ Vh[:k, :].transpose(-2, -1)).transpose(-2, -1)  # (k, m)
    return AdapterState(
        A_r.to(device=A.device, dtype=dtype).contiguous(),
        B_r.to(device=B.device, dtype=dtype).contiguous(),
    )


@AGGREGATIONS.register("exact_deltaw")
class ExactDeltaWAggregation:
    name = "exact_deltaw"

    def aggregate(
        self, updates: list[ClientUpdate], weights: Weights | None = None
    ) -> AggregateResult:
        """``weights`` default to the protocol weights p_i; a defence supplies q_i.

        Zero-weight clients are dropped before stacking, so a rejection shrinks the
        stack's inner dimension to |S|*r: the reported rank inflation stays the rank of
        what was actually aggregated.
        """
        updates, weights = active_clients(updates, weights)
        aggregate_state = {}
        next_global = {}
        for layer in _layer_names(updates):
            r = updates[0].adapters[layer].rank
            states = [u.adapters[layer] for u in updates]
            # exact mean of products, held factored (rank <= N*r)
            stacked = stacked_factors(states, weights)
            aggregate_state[layer] = stacked
            next_global[layer] = rank_r_from_factors(stacked.A, stacked.B, r)  # lossy
        return AggregateResult(
            aggregate_state=aggregate_state, next_global=next_global, strategy=self.name
        )
