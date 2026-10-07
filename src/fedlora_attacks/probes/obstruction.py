"""The det-sign orientation obstruction (CLAUDE.md invariant 2, plan T3/T4a).

This is deliberately NOT gauge-invariant. It is the Z/2 orientation class w_1: the
relative orientation of a client frame with respect to a consensus frame spanning an
aligned r-plane. It *flips* under a det<0 per-client (or per-consensus) gauge, and a
component change can occur only when the trajectory passes through the singular
locus V_{<=r-1} (T3(iii)). Do NOT test it for invariance — its test (test_detsign.py)
verifies it correctly DETECTS a known orientation reversal and does not flag an
orientation-preserving map.

Construction. For frames A_i (client) and A_c (consensus), the relative
change-of-basis on the common plane is M = pinv(A_c) @ A_i in GL(r) (generically),
and the invariant is sign(det(M)). Under A_i -> A_i @ inv(T): M -> M @ inv(T), so the
sign flips iff det(T) < 0 — the required equivariance. When col(A_i) = col(A_c),
M is the exact relative frame map and sign(det(M)) is the honest relative orientation.
"""

from __future__ import annotations

import torch

Tensor = torch.Tensor


def _frame(x) -> Tensor:
    """Extract the column-frame A from an AdapterState or accept a raw A tensor.

    Materialized on CPU in float64: the det/pinv here decide a discrete Z/2 invariant,
    so precision matters more than device locality (and consumer GPUs are slow in fp64).
    """
    t = x.A if hasattr(x, "A") else x
    return t.detach().to(device="cpu", dtype=torch.float64)


def relative_orientation_sign(client_frame, consensus_frame) -> int:
    """sign(det(pinv(A_c) @ A_i)) in {+1, -1, 0}. Equivariant (NOT invariant)."""
    a_i = _frame(client_frame)
    a_c = _frame(consensus_frame)
    if a_i.shape[1] != a_c.shape[1]:
        raise ValueError("frames must have the same number of columns (rank)")
    m = torch.linalg.pinv(a_c) @ a_i  # (r x r)
    det = torch.linalg.det(m).item()
    if det > 0:
        return +1
    if det < 0:
        return -1
    return 0


def relative_orientation_magnitude(client_frame, consensus_frame) -> float:
    """|det(pinv(A_c) @ A_i)| — small values flag proximity to a component boundary."""
    a_i = _frame(client_frame)
    a_c = _frame(consensus_frame)
    m = torch.linalg.pinv(a_c) @ a_i
    return abs(torch.linalg.det(m).item())


def orientation_obstruction(client_frames: list, consensus_frame) -> dict[str, object]:
    """Per-client relative orientation signs + whether an obstruction is present.

    An obstruction (w_1 != 0 at graph level) is signalled by a mismatch of signs
    across clients relative to the consensus.
    """
    signs = [relative_orientation_sign(f, consensus_frame) for f in client_frames]
    nonzero = [s for s in signs if s != 0]
    obstructed = len(set(nonzero)) > 1  # not all clients share one orientation
    return {
        "signs": signs,
        "num_reversed": sum(1 for s in signs if s < 0),
        "obstructed": obstructed,
    }
