"""Probes: gauge-invariant observables + the det-sign obstruction.

Importing this package registers all built-in probes.
"""

from . import geometry, obstruction
from .base import PROBES, Probe, ProbeContext, ProbeResult, build_probes, run_probes
from .probes import (  # noqa: F401 (import registers the probes)
    ClientConsensusAnglesProbe,
    DetSignObstructionProbe,
    RoundToRoundAnglesProbe,
    SignedAlignmentProbe,
    SpectrumProbe,
)

__all__ = [
    "PROBES",
    "Probe",
    "ProbeContext",
    "ProbeResult",
    "build_probes",
    "run_probes",
    "geometry",
    "obstruction",
]
