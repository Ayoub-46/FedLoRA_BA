"""Gauge-INVARIANT geometry kernels (CLAUDE.md invariant 1, plan §1/T1).

Every function here is a function of dW = A @ B (equivalently of col(dW), row(dW),
and the singular values of dW) and therefore invariant under the gauge action
(A, B) -> (A @ inv(T), T @ B). They are the only legal building blocks for measured
statistics. tests/test_gauge_invariance.py enforces this on synthetic factors and
tests/test_insitu_gauge_invariance.py on REAL trained ones.

CANONICAL SPECTRAL METHOD (CLAUDE.md "Spectral computation") — every probe uses it:
per adapted layer, thin-QR of the factors + one r x r core SVD, in fp64, and dW is
NEVER materialized. ``spectral_probe`` below is that method; everything else derives
from its outputs. A single global SVD over stacked layers is forbidden: each layer
carries an independent gauge T_l, so a global decomposition conflates independent
gauge fibers and is not invariant under the true group prod_l GL(r).
"""

from __future__ import annotations

import torch

Tensor = torch.Tensor


def as_delta_w(x) -> Tensor:
    """Materialize dW. TEST/TOY USE ONLY — the probe path never calls this.

    Kept so tests can assert the factored path agrees with the literal product, and
    for tiny hand-built examples. Production probes go through ``spectral_probe``.
    """
    if hasattr(x, "delta_w"):
        return x.delta_w()
    return x


def _to_cpu64(t: Tensor) -> Tensor:
    """Probe linear algebra runs on CPU in float64.

    Deliberate: consumer GPUs cripple fp64 (measured 767 ms for a 768x768 fp64 SVD on an
    RTX 2000 Ada vs 32 ms on CPU), and these decompositions are small and off the
    training hot path. Accuracy near the singular strata matters more than device
    locality here (invariant 6) — and training runs in bf16, so the upcast to fp64 in
    the probe is exactly where the precision is restored.
    """
    return t.detach().to(device="cpu", dtype=torch.float64)


def _factors(x) -> tuple[Tensor, Tensor]:
    """(A, B) in fp64 on CPU from an AdapterState or an explicit pair."""
    if hasattr(x, "A") and hasattr(x, "B"):
        return _to_cpu64(x.A), _to_cpu64(x.B)
    A, B = x
    return _to_cpu64(A), _to_cpu64(B)


# -- THE canonical spectral routine ---------------------------------------------------


def spectral_probe(A: Tensor, B: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Exact singular values and subspaces of dW = A @ B from the thin factors.

    A: (n, r), B: (r, m), r << n, m. Returns (S, Qa, Qb) where

        S  : (r,)    the EXACT singular values of dW, descending
        Qa : (n, r)  orthonormal, span(Qa) = col(A) = col(dW)
        Qb : (m, r)  orthonormal, span(Qb) = row(B) = row(dW)

    Exact, not approximate: with A = Qa Ra and B^T = Qb Rb, dW = Qa (Ra Rb^T) Qb^T,
    and Qa, Qb orthonormal makes the singular values of the r x r core Ra Rb^T the
    singular values of dW. Cost O((n+m)r^2 + r^3) instead of O(nmr) for forming dW
    (~100x cut at r=8 on 768x768), so fp64 on the core is free.

    Gauge behaviour: under (A, B) -> (A inv(T), T B), Qa and Qb pick up an orthogonal
    factor and Ra, Rb change — but span(Qa), span(Qb) and S do NOT. The individual
    matrices are gauge-variant; the three returned *geometric objects* are invariant.
    That is precisely what the in-situ invariance test confirms on real updates.
    """
    A = _to_cpu64(A)
    B = _to_cpu64(B)
    Qa, Ra = torch.linalg.qr(A, mode="reduced")  # (n, r), (r, r)
    Qb, Rb = torch.linalg.qr(B.transpose(-2, -1), mode="reduced")  # (m, r), (r, r)
    S = torch.linalg.svdvals(Ra @ Rb.transpose(-2, -1))
    return S, Qa, Qb


def column_space_of_factor(A) -> Tensor:
    """Orthonormal basis (n x r) of col(A) via QR — gauge-invariant subspace.

    col(A @ inv(T)) = col(A) for T in GL(r). On the top stratum (B full row rank)
    col(A) = col(dW), the plan's coordinate col(A) in Gr(r,n). Costs O(n r^2)
    (~0.2 ms at 768x8) versus a full SVD of dW (~32 ms at 768x768).
    """
    if hasattr(A, "A"):  # an AdapterState
        A = A.A
    Q, _ = torch.linalg.qr(_to_cpu64(A))
    return Q


def row_space_of_factor(B) -> Tensor:
    """Orthonormal basis (m x r) of row(B) via QR — the Gr(r,m) coordinate.

    row(T @ B) = row(B) for T in GL(r), so this is gauge-invariant for the same
    reason as ``column_space_of_factor``. The plan coordinatizes dW by BOTH
    col(A) and row(B); measuring only col(A) sees half the geometry.
    """
    if hasattr(B, "B"):  # an AdapterState
        B = B.B
    Q, _ = torch.linalg.qr(_to_cpu64(B).transpose(-2, -1))
    return Q


def subspace_distance_from_bases(basis1: Tensor, basis2: Tensor) -> dict[str, float]:
    """Angle summary between two already-computed subspace bases."""
    angles = principal_angles(basis1, basis2)
    if angles.numel() == 0:
        return {"max_angle": 0.0, "mean_angle": 0.0, "grassmann_dist": 0.0}
    return {
        "max_angle": angles.max().item(),
        "mean_angle": angles.mean().item(),
        "grassmann_dist": angles.pow(2).sum().sqrt().item(),
    }


# -- spectrum-derived invariants, computed from a PRE-COMPUTED singular vector --------
# These `*_from_spectrum` helpers are the single source of truth; the public functions
# below delegate to them. Callers needing several statistics at once must use
# ``spectral_summary`` so the decomposition happens exactly ONCE.


# Relative floor (times sigma_max) below which a singular value counts as zero.
# Rationale (invariant 6): an ill-conditioned gauge T inflates the float noise floor of
# the factored reconstruction to ~cond(T)*eps*sigma_max, which exceeds a bare
# eps*sigma_max threshold and would make the INTEGER rank flip under a pure gauge
# change. A 1e-10 relative floor absorbs cond(T) up to ~1e5 while staying far below any
# physically meaningful singular value. Integer rank is inherently discontinuous at the
# singular strata; the CONTINUOUS instruments (sigma_min_topr, rank_gap, effective_rank)
# are the primary ones the science uses -- this threshold only stabilizes reported rank.
DEFAULT_RANK_RTOL = 1e-10


def _numerical_rank_from(s: Tensor, shape: tuple[int, int], tol: float | None = None) -> int:
    if s.numel() == 0:
        return 0
    if tol is None:
        eps_tol = max(shape) * torch.finfo(s.dtype).eps
        tol = max(eps_tol, DEFAULT_RANK_RTOL) * s[0].item()
    return int((s > tol).sum().item())


def _sigma_min_from(s: Tensor, r: int) -> float:
    if s.numel() == 0:
        return 0.0
    return s[min(r, s.numel()) - 1].item()


def _rank_gap_from(s: Tensor, r: int) -> float:
    if s.numel() == 0:
        return 0.0
    sr = s[min(r, s.numel()) - 1].item()
    s_next = s[r].item() if s.numel() > r else 0.0
    return sr - s_next


def _condition_from(s: Tensor, k: int) -> float:
    if s.numel() == 0:
        return float("inf")
    k = max(min(k, s.numel()), 1)
    sr = s[k - 1].item()
    return float("inf") if sr <= 0 else s[0].item() / sr


def _stable_rank_from(s: Tensor) -> float:
    if s.numel() == 0 or s[0].item() == 0:
        return 0.0
    return (s.pow(2).sum() / s[0].pow(2)).item()


def _effective_rank_from(s: Tensor) -> float:
    """Singular-value entropy effective rank exp(-sum p_i log p_i), p_i = s_i / sum s.

    THE effective-rank metric for this project (see README "Benign baseline"). Chosen
    over stable rank because it is bounded in [1, len(s)] with the direct reading
    "number of effectively active directions", is scale-invariant, and responds to the
    whole spectrum instead of saturating near 1 whenever sigma_1 dominates — which is
    the regime LoRA updates are expected to sit in, i.e. exactly where stable rank
    would compress the signal E2 needs. Stable rank is logged alongside as a
    cross-check, never as the headline.
    """
    if s.numel() == 0:
        return 0.0
    total = s.sum()
    if total <= 0:
        return 0.0
    p = s / total
    p = p[p > 0]
    entropy = -(p * p.log()).sum()
    return float(torch.exp(entropy).item())


def _summarize_spectrum(s: Tensor, r: int, shape: tuple[int, int]) -> dict:
    """All spectrum-derived invariants from one already-computed singular vector."""
    return {
        # The factored method returns EXACTLY the nonzero singular values (at most the
        # inner dimension), so the full vector is logged — no truncation heuristic.
        "singular_values": s.tolist(),
        "sigma_max": s[0].item() if s.numel() else 0.0,
        "sigma_min_topr": _sigma_min_from(s, r),
        "rank_gap": _rank_gap_from(s, r),
        "condition_number": _condition_from(s, r),
        "effective_rank": _effective_rank_from(s),  # PRIMARY (entropy)
        "stable_rank": _stable_rank_from(s),  # secondary cross-check
        "numerical_rank": _numerical_rank_from(s, shape),
        "frobenius_norm": s.pow(2).sum().sqrt().item(),
        "nominal_rank": min(r, s.numel()),
    }


def factored_spectral_summary(state, r: int | None = None) -> dict:
    """Canonical entry point: all spectral invariants of dW from the FACTORS.

    ``state`` is an AdapterState (or an (A, B) pair). ``r`` defaults to the factor's
    own inner dimension, which is the nominal rank; pass it explicitly for a stacked
    aggregate whose inner dimension is N*r but whose *nominal* per-client rank is r.
    """
    A, B = _factors(state)
    s, _, _ = spectral_probe(A, B)
    r = A.shape[1] if r is None else r
    return _summarize_spectrum(s, r, (A.shape[0], B.shape[1]))


def spectral_summary(x, r: int, log_topk: int | None = None) -> dict:
    """Spectral invariants, accepting either factors (canonical) or a raw dW (toy).

    Prefer ``factored_spectral_summary``. This wrapper exists so hand-built toy
    examples and tests can pass a literal matrix.
    """
    if hasattr(x, "A") and hasattr(x, "B"):
        return factored_spectral_summary(x, r)
    dw = _to_cpu64(x)
    s = torch.linalg.svdvals(dw)
    if log_topk is not None:
        s = s[: max(log_topk, r)]
    return _summarize_spectrum(s, r, tuple(dw.shape))


def _svdvals(dw: Tensor) -> Tensor:
    return torch.linalg.svdvals(_to_cpu64(dw))


def singular_spectrum(x) -> Tensor:
    """All (nonzero) singular values of dW, descending. Gauge-invariant.

    Uses the factored path when given an AdapterState, the direct SVD for a raw matrix.
    """
    if hasattr(x, "A") and hasattr(x, "B"):
        A, B = _factors(x)
        s, _, _ = spectral_probe(A, B)
        return s
    return _svdvals(x)


def numerical_rank(x, tol: float | None = None) -> int:
    """Rank via singular-value thresholding. tol defaults to max(n,m)*eps*sigma_max."""
    s = singular_spectrum(x)
    shape = x.shape if hasattr(x, "shape") else (s.numel(), s.numel())
    return _numerical_rank_from(s, tuple(shape), tol)


def sigma_min_topr(x, r: int) -> float:
    """The r-th singular value sigma_r: distance of dW to the stratum V_{<=r-1}.

    This is the E2/T2 knob — as dW approaches lower rank, sigma_r -> 0. Eckart-Young
    makes it exactly the spectral distance to the nearest rank-(r-1) matrix.
    """
    return _sigma_min_from(singular_spectrum(x), r)


def rank_gap(x, r: int) -> float:
    """Spectral gap sigma_r - sigma_{r+1} (0 if there is no sigma_{r+1})."""
    return _rank_gap_from(singular_spectrum(x), r)


def condition_number(x, r: int | None = None) -> float:
    """sigma_1 / sigma_r over the top-r block (r defaults to numerical rank).

    Large values flag ill-conditioned gauge-fixing near the singular strata.
    """
    s = singular_spectrum(x)
    k = _numerical_rank_from(s, tuple(x.shape)) if r is None else r
    return _condition_from(s, k)


def stable_rank(x) -> float:
    """||dW||_F^2 / ||dW||_2^2 — smooth rank surrogate (secondary metric)."""
    return _stable_rank_from(singular_spectrum(x))


def effective_rank(x) -> float:
    """Singular-value entropy effective rank — the PRIMARY effective-rank metric."""
    return _effective_rank_from(singular_spectrum(x))


def column_space(x, r: int | None = None) -> Tensor:
    """Orthonormal basis of col(dW). Gauge-invariant.

    For factors this is the QR path (span(Qa) = col(dW)); for a raw matrix, the left
    singular vectors.
    """
    if hasattr(x, "A") and hasattr(x, "B"):
        A, B = _factors(x)
        _, Qa, _ = spectral_probe(A, B)
        return Qa if r is None else Qa[:, : max(min(r, Qa.shape[1]), 1)]
    dw = _to_cpu64(x)
    U, _, _ = torch.linalg.svd(dw, full_matrices=False)
    k = numerical_rank(dw) if r is None else min(r, U.shape[1])
    return U[:, : max(k, 1)]


def row_space(x, r: int | None = None) -> Tensor:
    """Orthonormal basis of row(dW). Gauge-invariant."""
    if hasattr(x, "A") and hasattr(x, "B"):
        A, B = _factors(x)
        _, _, Qb = spectral_probe(A, B)
        return Qb if r is None else Qb[:, : max(min(r, Qb.shape[1]), 1)]
    dw = _to_cpu64(x)
    _, _, Vh = torch.linalg.svd(dw, full_matrices=False)
    k = numerical_rank(dw) if r is None else min(r, Vh.shape[0])
    return Vh[: max(k, 1), :].transpose(-2, -1)


def principal_angles(basis1: Tensor, basis2: Tensor) -> Tensor:
    """Principal angles (radians, ascending) between the spans of two bases.

    Bases need not be orthonormal; they are re-orthonormalized via QR, so the result
    depends only on the two subspaces — hence gauge-invariant when the bases are
    column/row spaces of dW.
    """
    q1, _ = torch.linalg.qr(_to_cpu64(basis1))
    q2, _ = torch.linalg.qr(_to_cpu64(basis2))
    s = torch.linalg.svdvals(q1.transpose(-2, -1) @ q2).clamp(-1.0, 1.0)
    return torch.arccos(s)  # svdvals descending -> arccos ascending


def subspace_distance(x1, x2, r: int | None = None) -> dict[str, float]:
    """Summary of the angle between col(dW1) and col(dW2): max/mean/Grassmann."""
    angles = principal_angles(column_space(x1, r), column_space(x2, r))
    if angles.numel() == 0:
        return {"max_angle": 0.0, "mean_angle": 0.0, "grassmann_dist": 0.0}
    return {
        "max_angle": angles.max().item(),
        "mean_angle": angles.mean().item(),
        "grassmann_dist": angles.pow(2).sum().sqrt().item(),
    }


def factored_frobenius_norm(A: Tensor, B: Tensor) -> float:
    """||A @ B||_F without materializing the product, via thin QR.

    ||A B||_F = ||Ra Rb^T||_F for thin QRs A = Qa Ra, B^T = Qb Rb, because orthonormal
    factors preserve the Frobenius norm. Used by the naive-vs-exact aggregation gap,
    whose difference is itself a product of stacked factors of inner dimension (N+1)r.

    THE STABLE ONE. Prefer this whenever the product involves CANCELLATION — a stacked
    difference, where the answer is small relative to the terms that produced it. See
    ``deltaw_frobenius_norm`` for the cheaper Gram route and why the two coexist.
    """
    A = _to_cpu64(A)
    B = _to_cpu64(B)
    _, Ra = torch.linalg.qr(A, mode="reduced")
    _, Rb = torch.linalg.qr(B.transpose(-2, -1), mode="reduced")
    return torch.linalg.norm(Ra @ Rb.transpose(-2, -1)).item()


# -- dW-measurable kernels a robust aggregator needs (plan v3 §6b) --------------------
# Every statistic a deployed defence computes is available from the factors at
# O((n+m)r^2 + r^3), with nothing of size n*m ever formed. That is the point of §6b:
# representation-independence is CHEAP, so factor-space statistics have no efficiency
# excuse and the trap cell of the defence matrix is harder to defend, not easier.


def deltaw_frobenius_norm(A: Tensor, B: Tensor) -> float:
    """||A @ B||_F = sqrt(tr((A^T A)(B B^T))) — two r x r Grams and a trace.

    The quantity ``norm_bound`` clips on, and the one C2 is a claim about: it is a
    function of dW alone, so the gauge (A, B) -> (A inv(T), T B) leaves it EXACTLY
    fixed, and a bound on it cannot see Channel A no matter how it is tuned.

    Same asymptotic cost as the QR route but a smaller constant (no orthogonalization).
    The trade is numerical: the Gram form squares the condition number, so it is right
    for the norm of ONE update — where the answer is the same size as the terms — and
    wrong for a difference. ``frobenius_factored_difference`` therefore takes the QR
    route instead. Do not "deduplicate" the two; the split is the numerics.
    """
    return _frob_sq_from_factors(A, B) ** 0.5


def frobenius_factored_difference(A1: Tensor, B1: Tensor, A2: Tensor, B2: Tensor) -> float:
    """||A1 B1 - A2 B2||_F from the factors, via a single 2r x 2r core.

    The difference of two rank-r products is itself a factored product of inner
    dimension 2r:  A1 B1 - A2 B2 = [A1 | -A2] @ [B1 ; B2]  (the same construction as
    ``stacked_factors`` with signed weights). Thin-QR both stacks and the norm is
    ||R_C R_D^T||_F on the small core.

    Deliberately the QR route, not the Gram route: the case this kernel exists for is
    NEAR-CANCELLATION. Multi-Krum's blindness (E-B) is the statement that the distance
    between an update and its gauge re-expression is zero, and a Gram-based difference
    would lose most of its significant digits exactly there. Measured on gauge pairs it
    returns ~1e-16 relative; the Gram route does not.
    """
    A1 = _to_cpu64(A1)
    B1 = _to_cpu64(B1)
    A2 = _to_cpu64(A2)
    B2 = _to_cpu64(B2)
    C = torch.cat([A1, -A2], dim=1)  # (n, 2r)
    D = torch.cat([B1, B2], dim=0)  # (2r, m)
    return factored_frobenius_norm(C, D)


def full_svd_from_factors(A: Tensor, B: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """The EXACT thin SVD of dW = A @ B from the factors: (U, S, V) with dW = U diag(S) V^T.

    ``spectral_probe`` returns the singular values and the two SUBSPACES, which is all a
    gauge-invariant statistic may report. Channel B needs more: it acts as
    dW -> U P Sigma Q^T V^T, so it needs the singular VECTORS paired with their values,
    not merely the spans they sit in.

    Same construction, one extra multiply: with A = Qa Ra, B^T = Qb Rb and
    Ra Rb^T = U~ S V~^T, dW = (Qa U~) S (Qb V~)^T. Exact, and still O((n+m)r^2 + r^3).

    Gauge behaviour, and it is the reason this is a kernel rather than a probe: U and V
    are gauge-INVARIANT up to the sign/rotation freedom of the SVD itself (they are
    determined by dW, not by the factorization), but individual columns flip sign under
    a degenerate spectrum. Report spans or spectra, never raw U/V entries.
    """
    A = _to_cpu64(A)
    B = _to_cpu64(B)
    Qa, Ra = torch.linalg.qr(A, mode="reduced")
    Qb, Rb = torch.linalg.qr(B.transpose(-2, -1), mode="reduced")
    U_core, S, Vh_core = torch.linalg.svd(Ra @ Rb.transpose(-2, -1))
    return Qa @ U_core, S, Qb @ Vh_core.transpose(-2, -1)


def pairwise_deltaw_distances(states, weights: list[float] | None = None) -> Tensor:
    """N x N matrix of ||dW_i - dW_j||_F, factored throughout. Multi-Krum's input.

    ``states`` are AdapterStates (or (A, B) pairs) for ONE layer. ``weights`` optionally
    pre-scales each update; None (the default) is the convention Krum is defined under,
    where scores compare the raw submissions rather than their weighted contributions.

    Symmetric with a zero diagonal, fp64, cost O(N^2 (n+m) r^2) — the §6b table's last
    row. A defence built on this is fully dW-measurable and therefore, by C1, exactly as
    blind to Channel A as the norm bound is: every entry is unchanged when any subset of
    clients re-expresses its factors.
    """
    pairs = [_factors(s) for s in states]
    if weights is not None:
        if len(weights) != len(pairs):
            raise ValueError(f"expected {len(pairs)} weights, got {len(weights)}")
        pairs = [(float(w) * A, B) for (A, B), w in zip(pairs, weights, strict=True)]
    n = len(pairs)
    out = torch.zeros(n, n, dtype=torch.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = frobenius_factored_difference(*pairs[i], *pairs[j])
            out[i, j] = d
            out[j, i] = d
    return out


# -- SIGNED cross-update alignment (the rho batch) ------------------------------------
# These measure the relationship BETWEEN two updates dW1, dW2 and are the first probes
# that read the SIGN the (col, row, sigma) triple discards (CLAUDE.md domain model:
# that triple cannot tell dW from -dW). They are still gauge-invariant: every output is
# a function of dW1 and dW2 alone, so independent gauge transforms of each argument
# leave them fixed. The whole batch is built from ONE r x r cross-Gram, computed from
# factors without materializing either dW.


def _frob_sq_from_factors(A: Tensor, B: Tensor) -> float:
    """||A @ B||_F^2 = tr((A^T A)(B B^T)) = sum_i sigma_i^2, from factors (r x r work)."""
    A = _to_cpu64(A)
    B = _to_cpu64(B)
    return torch.trace((A.transpose(-2, -1) @ A) @ (B @ B.transpose(-2, -1))).item()


def cross_gram(A1: Tensor, B1: Tensor, A2: Tensor, B2: Tensor) -> Tensor:
    """The r x r cross-Gram M = (A1^T A2)(B2 B1^T). tr(M) = <dW1, dW2>_F.

    Gauge behaviour: under independent gauges (A1,B1)->(A1 inv(T1), T1 B1) and
    (A2,B2)->(A2 inv(T2), T2 B2), M -> T1^-T M T1^T — a SIMILARITY transform (T2 cancels
    entirely). So M's *similarity invariants* (trace, determinant, eigenvalues, the whole
    characteristic polynomial) are gauge-invariant, but its SINGULAR values are NOT
    (similarity does not preserve singular values). This is why the cross-Gram spectrum
    logged below is its EIGENVALUES, never its singular values — the latter would fail
    the gauge test.
    """
    A1 = _to_cpu64(A1)
    B1 = _to_cpu64(B1)
    A2 = _to_cpu64(A2)
    B2 = _to_cpu64(B2)
    return (A1.transpose(-2, -1) @ A2) @ (B2 @ B1.transpose(-2, -1))


def signed_alignment(x1, x2) -> float:
    """rho(dW1, dW2) = <dW1, dW2>_F / (||dW1||_F ||dW2||_F): the signed Frobenius cosine.

    The core batch scalar and the E2 predictor. In [-1, 1]; +1 iff dW1 = c dW2 (c>0),
    -1 iff c<0. Unlike principal angles (which see only the unsigned column/row spaces),
    rho distinguishes dW from -dW. Cost O((n+m)r^2 + r^3), factored, fp64.
    """
    A1, B1 = _factors(x1)
    A2, B2 = _factors(x2)
    num = torch.trace(cross_gram(A1, B1, A2, B2)).item()
    n1 = _frob_sq_from_factors(A1, B1) ** 0.5
    n2 = _frob_sq_from_factors(A2, B2) ** 0.5
    if n1 == 0.0 or n2 == 0.0:
        return float("nan")
    return num / (n1 * n2)


def alignment_spectrum(x1, x2, log_topk: int | None = None) -> dict:
    """Gauge-invariant alignment of two updates: rho + the cross-Gram EIGENVALUE spectrum.

    From the single r x r cross-Gram M (see ``cross_gram``):
      * ``rho``            tr(M) / (||dW1|| ||dW2||)          — signed cosine (the scalar)
      * ``eig_real/imag``  eigenvalues of M / (||dW1|| ||dW2||), sorted by |.| desc.
                           Their real parts SUM to rho; this is rho resolved per coupling
                           mode. Complex parts flag rotational (non-collinear) coupling
                           between the two updates' singular structure. All invariant.
      * ``orientation_sign``  sign(det M) in {+1,-1,0} — a gauge-INVARIANT Z/2 relative
                           orientation. (Distinct from the equivariant det-sign probe,
                           which flips under gauge; this one does not.)
      * ``det_normalized`` det(M) / (||dW1|| ||dW2||)^r — scale-free magnitude of the
                           orientation, small when the coupling is near-degenerate.

    All entries are functions of dW1, dW2 only -> gauge-invariant (tested).
    """
    A1, B1 = _factors(x1)
    A2, B2 = _factors(x2)
    M = cross_gram(A1, B1, A2, B2)
    r = M.shape[0]
    n1 = _frob_sq_from_factors(A1, B1) ** 0.5
    n2 = _frob_sq_from_factors(A2, B2) ** 0.5
    scale = n1 * n2
    if scale == 0.0:
        return {
            "rho": float("nan"),
            "eig_real": [],
            "eig_imag": [],
            "orientation_sign": 0,
            "det_normalized": float("nan"),
        }
    eig = torch.linalg.eigvals(M) / scale  # complex, length r; sum(real) = rho
    # Lexicographic order (|.| desc, Re desc, Im desc), NOT magnitude alone: a complex
    # conjugate pair shares a magnitude, so magnitude-only sorting leaves its order
    # ambiguous and an independent gauge could swap the two — breaking the element-wise
    # invariance test. The full key is deterministic on the (gauge-invariant) eigenvalue
    # multiset. Rounding the keys makes near-ties order consistently across the two runs.
    triples = sorted(
        ((abs(v), v.real, v.imag) for v in eig.tolist()),
        key=lambda t: (round(t[0], 9), round(t[1], 9), round(t[2], 9)),
        reverse=True,
    )
    if log_topk is not None:
        triples = triples[:log_topk]
    det = torch.linalg.det(M).item()
    return {
        "rho": (torch.trace(M).item() / scale),
        "eig_real": [t[1] for t in triples],
        "eig_imag": [t[2] for t in triples],
        "orientation_sign": (1 if det > 0 else -1 if det < 0 else 0),
        "det_normalized": det / (scale**r) if scale > 0 else float("nan"),
    }
