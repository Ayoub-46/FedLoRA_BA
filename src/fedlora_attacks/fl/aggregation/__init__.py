"""Pluggable aggregation strategies (registered on import)."""

from .base import (
    AGGREGATIONS,
    Aggregation,
    active_clients,
    build_aggregation,
    canonical_frames,
    deltaw_discrepancy,
    normalized_weights,
    resolve_weights,
)
from .canonicalized_fedit import CanonicalizedFedITAggregation, canonicalize
from .consensus_frame import (
    COORDINATE_RULES,
    FRAME_ESTIMATORS,
    ConsensusFrameAggregation,
    CoordinateAggregation,
    CoordinateMedian,
    GeometricMedianCoordinates,
    MagnitudeConsensusFrameAggregation,
    TrimmedMeanCoordinates,
    WeightedMeanCoordinates,
    build_coordinate_aggregation,
    estimate_frame,
    project_coordinates,
)
from .exact_deltaw import ExactDeltaWAggregation, rank_r_factorization
from .fedit_naive import FedITNaiveAggregation

__all__ = [
    "Aggregation",
    "AGGREGATIONS",
    "build_aggregation",
    "canonical_frames",
    "deltaw_discrepancy",
    "normalized_weights",
    "resolve_weights",
    "active_clients",
    "FedITNaiveAggregation",
    "ExactDeltaWAggregation",
    "CanonicalizedFedITAggregation",
    "canonicalize",
    "ConsensusFrameAggregation",
    "MagnitudeConsensusFrameAggregation",
    "estimate_frame",
    "FRAME_ESTIMATORS",
    "COORDINATE_RULES",
    "CoordinateAggregation",
    "WeightedMeanCoordinates",
    "CoordinateMedian",
    "TrimmedMeanCoordinates",
    "GeometricMedianCoordinates",
    "build_coordinate_aggregation",
    "project_coordinates",
    "rank_r_factorization",
]
