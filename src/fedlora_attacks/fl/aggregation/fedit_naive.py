"""fedit_naive: factor-wise averaging (INEXACT, documented as such).

Defining property (asserted in tests/test_aggregation.py):
    A_agg = sum_i w_i A_i,  B_agg = sum_i w_i B_i,  dW = A_agg @ B_agg.
This is what standard FedIT/FedAvg-over-LoRA does. It does NOT equal
mean(A_i @ B_i) unless the clients' factors are gauge-aligned — that error is the
attacker's operating room (§1).
"""

from __future__ import annotations

from ..types import AdapterState, AggregateResult, ClientUpdate
from .base import AGGREGATIONS, Weights, _layer_names, active_clients


@AGGREGATIONS.register("fedit_naive")
class FedITNaiveAggregation:
    name = "fedit_naive"

    def aggregate(
        self, updates: list[ClientUpdate], weights: Weights | None = None
    ) -> AggregateResult:
        """``weights`` default to the protocol weights p_i; a defence supplies q_i."""
        updates, weights = active_clients(updates, weights)
        aggregate_state = {}
        next_global = {}
        for layer in _layer_names(updates):
            A_agg = sum(w * u.adapters[layer].A for w, u in zip(weights, updates, strict=False))
            B_agg = sum(w * u.adapters[layer].B for w, u in zip(weights, updates, strict=False))
            state = AdapterState(A_agg, B_agg)
            # measured and redistributed states coincide here: what the server holds IS
            # the product of the averaged factors, rank r (no truncation step, no gap
            # between what probes see and what clients receive).
            aggregate_state[layer] = state
            next_global[layer] = state
        return AggregateResult(
            aggregate_state=aggregate_state, next_global=next_global, strategy=self.name
        )
