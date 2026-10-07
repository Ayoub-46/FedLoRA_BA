"""Defenses: weight-returning interface and registered robust scoring rules.

Importing this package registers every built-in defence.
"""

from .aggregate_norm_check import (  # noqa: F401 (registers aggregate_norm_check)
    AggregateNormCheckConfig,
    AggregateNormCheckDefense,
)
from .base import (
    ACCEPT,
    CLIP,
    DEFENSES,
    REJECT,
    BaseDefense,
    Defense,
    DefenseConfig,
    DefenseContext,
    DefenseResult,
    NoDefense,
    build_defense,
)
from .cosdefense import (  # noqa: F401 (registers cosdefense)
    CosDefense,
    CosDefenseConfig,
    consensus_cosines,
)
from .factor_broadcast_cosine import (  # noqa: F401 (registers factor_broadcast_cosine)
    FactorBroadcastCosineConfig,
    FactorBroadcastCosineDefense,
    factor_broadcast_scores,
)
from .factor_broadcast_topk import (  # noqa: F401 (registers factor_broadcast_topk)
    FactorBroadcastTopKConfig,
    FactorBroadcastTopKDefense,
    factor_broadcast_euclidean_scores,
)
from .fltrust import (  # noqa: F401 (registers fltrust)
    FLTrustConfig,
    FLTrustDefense,
    TrustedRootShard,
    fltrust_deltaw_statistics,
    reserve_trusted_root,
)
from .multi_krum import (  # noqa: F401 (import registers multi_krum)
    MultiKrumConfig,
    MultiKrumDefense,
    krum_scores,
    squared_deltaw_distance_matrix,
)
from .norm_bound import NormBoundConfig, NormBoundDefense  # noqa: F401 (registers norm_bound)
from .spectral_check import (  # noqa: F401 (registers spectral_check)
    SpectralCheckConfig,
    SpectralCheckDefense,
    client_spectral_stats,
)

__all__ = [
    "CosDefense",
    "CosDefenseConfig",
    "consensus_cosines",
    "FLTrustDefense",
    "FLTrustConfig",
    "TrustedRootShard",
    "fltrust_deltaw_statistics",
    "reserve_trusted_root",
    "FactorBroadcastCosineDefense",
    "FactorBroadcastCosineConfig",
    "factor_broadcast_scores",
    "FactorBroadcastTopKDefense",
    "FactorBroadcastTopKConfig",
    "factor_broadcast_euclidean_scores",
    "SpectralCheckDefense",
    "SpectralCheckConfig",
    "client_spectral_stats",
    "Defense",
    "DefenseConfig",
    "DefenseContext",
    "DefenseResult",
    "BaseDefense",
    "NoDefense",
    "DEFENSES",
    "build_defense",
    "ACCEPT",
    "CLIP",
    "REJECT",
    "AggregateNormCheckConfig",
    "AggregateNormCheckDefense",
    "NormBoundConfig",
    "NormBoundDefense",
    "MultiKrumConfig",
    "MultiKrumDefense",
    "krum_scores",
    "squared_deltaw_distance_matrix",
]
