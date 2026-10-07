"""canonicalized_fedit: orthonormalise each client's A, then average factors.

The obvious repair, and the paper kills it constructively (C3/C4, experiment E-C). The
server does not trust the client's choice of factorisation; it re-expresses each
submission in a canonical one before averaging:

    A_i = Q_i R_i  (thin QR)  ->  A~_i = Q_i,  B~_i = R_i B_i,   A~_i B~_i = dW_i.

Every client's effective update is untouched — the server has only gauge-fixed the
representation. What this buys, and what it does not:

**Does not close Channel A (Thm 4.18).** Orthonormalising `A_i` fixes the gauge only up
to `O(r)`: any `Q_i -> Q_i P` with `P` orthogonal leaves `A~_i` orthonormal, so a
coalition can still choose freely within `O(r)` and the cross terms
`A~_i P_i^T P_j B~_j` still move the factor average.

**Does bound it (T7).** `O(r)` is COMPACT, which the full `GL(r)` is not. With
`||A~_i||_2 = 1` and `||Q^T Q' - I||_2 <= 2`, the displacement obeys
`|| sum_{i != j} p_i p_j A~_i C_ij B~_j ||_F <= 2 max_j ||dW_j||_F` for a single
attacker, so canonicalisation plus a dW-norm bound `C` gives displacement `<= 2 kappa C`.
An unbounded channel becomes a bounded one — the paper's constructive corner.

Honest framing, and it is E-C's actual question: the legitimate aggregate is itself
O(C), so the guarantee is meaningful but weak. Whether the bounded displacement still
degrades the model is empirical, and is what makes T7 useful rather than merely true.
"""

from __future__ import annotations

import torch

from ..types import AdapterDict, AdapterState, AggregateResult, ClientUpdate
from .base import AGGREGATIONS, Weights, _layer_names, active_clients


def canonicalize(state: AdapterState) -> AdapterState:
    """(A, B) -> (Q, R B) with Q orthonormal. dW is unchanged, exactly.

    QR runs in fp64 and the result is cast back, for the same reason
    ``gauge_transform`` does: this is a change of representation, and it should not cost
    significant digits in the object being represented.
    """
    A64 = state.A.to(dtype=torch.float64)
    Q, R = torch.linalg.qr(A64, mode="reduced")
    return AdapterState(
        Q.to(device=state.A.device, dtype=state.A.dtype).contiguous(),
        (R @ state.B.to(dtype=torch.float64)).to(device=state.B.device, dtype=state.B.dtype),
    )


@AGGREGATIONS.register("canonicalized_fedit")
class CanonicalizedFedITAggregation:
    """Per-client QR gauge-fixing, then weighted factor averaging (plan v3 C3/C4, E-C)."""

    name = "canonicalized_fedit"

    def aggregate(
        self, updates: list[ClientUpdate], weights: Weights | None = None
    ) -> AggregateResult:
        updates, weights = active_clients(updates, weights)
        aggregate_state: AdapterDict = {}
        next_global: AdapterDict = {}
        for layer in _layer_names(updates):
            canon = [canonicalize(u.adapters[layer]) for u in updates]
            A_agg = sum(w * s.A for w, s in zip(weights, canon, strict=True))
            B_agg = sum(w * s.B for w, s in zip(weights, canon, strict=True))
            state = AdapterState(A_agg, B_agg)
            # Like fedit_naive, the measured and redistributed states coincide: this is
            # still factor averaging (rank r, no truncation step). Only the gauge the
            # factors are expressed in has changed.
            aggregate_state[layer] = state
            next_global[layer] = state
        return AggregateResult(
            aggregate_state=aggregate_state, next_global=next_global, strategy=self.name
        )
