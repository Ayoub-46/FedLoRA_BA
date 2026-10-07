"""Federated-learning simulation: domain types, aggregation, client/server loop."""

# NOTE: keep this package init to LEAF exports (types + aggregation) only.
# Client/Server are imported by submodule path (``fedlora_attacks.fl.client``/
# ``.server``) to avoid a circular import: attacks/defenses annotate on fl.types,
# and client/server import attacks/defenses — importing them here would cycle.
from .aggregation import (
    AGGREGATIONS,
    Aggregation,
    build_aggregation,
    deltaw_discrepancy,
)
from .types import (
    AdapterDict,
    AdapterState,
    AggregateResult,
    ClientUpdate,
    GlobalState,
)

__all__ = [
    "AdapterState",
    "AdapterDict",
    "ClientUpdate",
    "GlobalState",
    "AggregateResult",
    "Aggregation",
    "AGGREGATIONS",
    "build_aggregation",
    "deltaw_discrepancy",
]
