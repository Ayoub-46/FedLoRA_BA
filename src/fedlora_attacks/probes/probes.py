"""Registered probes (the per-round measurement suite, CLAUDE.md invariant 1).

INVARIANT probes: spectrum, aggregate_spectrum, client_consensus_angles,
round_to_round_angles, alignment.
EQUIVARIANT probe: detsign_obstruction.
GAUGE_VARIANT probe: factor_norms (||A_bar||_F) — E-A2 P1, E-G.

All statistics are dW-derived, computed via the canonical factored method
(thin-QR + r x r core, fp64 — see probes/geometry.spectral_probe); dW is never
materialized. Everything is logged PER LAYER; collapsing across layers happens only
in the analysis layer (CLAUDE.md "Storage"), which keeps E3/T4b's per-layer profile
available for free.
"""

from __future__ import annotations

from ..fl.types import stacked_factors
from . import geometry as geo
from . import obstruction as obs
from .base import EQUIVARIANT, GAUGE_VARIANT, INVARIANT, PROBES, ProbeContext, ProbeResult


@PROBES.register("spectrum")
class SpectrumProbe:
    """Per-layer singular-value spectrum: EVERY client's dW plus the in-force aggregate.

    Per-client spectra are the benign null distribution E2 calibrates "proximity to the
    singular locus" against — sigma_min must be read relative to what benign training
    actually produces, not against an absolute constant.
    """

    name = "spectrum"
    category = INVARIANT

    def __call__(self, ctx: ProbeContext) -> ProbeResult:
        values = {}
        for layer, agg_state in ctx.aggregate.aggregate_state.items():
            r = ctx.consensus[layer].rank
            per_client = {
                str(u.client_id): geo.factored_spectral_summary(u.adapters[layer], r)
                for u in ctx.client_updates
            }
            values[layer] = {
                "per_client": per_client,
                # The aggregate as the in-force strategy actually produced it. For
                # fedit_naive its inner dim is r; for exact_deltaw it is N*r.
                "aggregate": geo.factored_spectral_summary(agg_state, r),
                # What clients receive next round (rank-r for every strategy).
                "consensus": geo.factored_spectral_summary(ctx.consensus[layer], r),
            }
        return ProbeResult(self.name, self.category, values)


@PROBES.register("aggregate_spectrum")
class ExactAggregateSpectrumProbe:
    """Spectrum of the EXACT aggregate sum_i w_i dW_i from stacked factors.

    Computed regardless of which aggregation strategy is in force, so a single sweep
    yields both the deployed consensus and the exact object. Its rank runs up to N*r,
    not r: that inflation is CLAUDE.md invariant 3 / the §1 efficiency-vs-invariance
    gap made measurable, and is logged first-class rather than hidden.
    """

    name = "aggregate_spectrum"
    category = INVARIANT

    def __call__(self, ctx: ProbeContext) -> ProbeResult:
        # The weights the round ACTUALLY aggregated under (q_i from the defence), not a
        # freshly recomputed p_i: under a reweighting defence the two differ and the
        # latter would describe an aggregate the server never formed.
        from ..fl.aggregation.base import active_clients

        active, weights = active_clients(ctx.client_updates, ctx.weights)
        values = {}
        for layer in ctx.aggregate.aggregate_state:
            r = ctx.consensus[layer].rank
            states = [u.adapters[layer] for u in active]
            stacked = stacked_factors(states, weights)
            summary = geo.factored_spectral_summary(stacked, r)
            summary["nominal_rank"] = r  # per-client nominal rank, for erank/r
            summary["stack_dim"] = stacked.rank  # N*r, the rank ceiling
            values[layer] = summary
        return ProbeResult(self.name, self.category, values)


@PROBES.register("client_consensus_angles")
class ClientConsensusAnglesProbe:
    """Principal angles from each client's dW to the consensus, in BOTH coordinates.

    The plan coordinatizes dW by col(A) in Gr(r,n) AND row(B) in Gr(r,m); measuring
    only col(A) sees half the geometry. Both bases come from QR of the thin factors —
    gauge-invariant subspaces, O(n r^2), no SVD of dW.

    Angles are reported against two reference frames: the IN-FORCE consensus (what
    clients actually receive) and the exact aggregate's rank-r CANONICAL frame (the
    SVD-gauge-fixed reference), so a canonical-gauge reading needs no re-run.
    """

    name = "client_consensus_angles"
    category = INVARIANT
    requires_canonical_frames = True

    def __call__(self, ctx: ProbeContext) -> ProbeResult:
        values = {}
        canonical = ctx.canonical_frames
        for layer in ctx.aggregate.aggregate_state:
            refs = {"consensus": ctx.consensus[layer]}
            if canonical is not None and layer in canonical:
                refs["canonical"] = canonical[layer]

            layer_values = {}
            for ref_name, ref_state in refs.items():
                # reference bases computed ONCE per layer, not once per client
                q_col_ref = geo.column_space_of_factor(ref_state.A)
                q_row_ref = geo.row_space_of_factor(ref_state.B)
                per_client = {}
                for u in ctx.client_updates:
                    state = u.adapters[layer]
                    col = geo.subspace_distance_from_bases(
                        geo.column_space_of_factor(state.A), q_col_ref
                    )
                    row = geo.subspace_distance_from_bases(
                        geo.row_space_of_factor(state.B), q_row_ref
                    )
                    per_client[str(u.client_id)] = {"col": col, "row": row}
                col_max = [v["col"]["max_angle"] for v in per_client.values()]
                row_max = [v["row"]["max_angle"] for v in per_client.values()]
                layer_values[ref_name] = {
                    "per_client": per_client,
                    "mean_max_angle_col": _mean(col_max),
                    "max_max_angle_col": max(col_max) if col_max else 0.0,
                    "mean_max_angle_row": _mean(row_max),
                    "max_max_angle_row": max(row_max) if row_max else 0.0,
                    # spread across clients = the heterogeneity signal (should grow as
                    # alpha falls); std, not just mean, because the sanity check is
                    # about dispersion of the client cloud around the consensus.
                    "std_max_angle_col": _std(col_max),
                    "std_max_angle_row": _std(row_max),
                }
            values[layer] = layer_values
        return ProbeResult(self.name, self.category, values)


@PROBES.register("round_to_round_angles")
class RoundToRoundAnglesProbe:
    """Principal angles between this round's consensus subspaces and the previous.

    The convergence instrument: as training settles, the consensus frame should stop
    moving, so these angles must DECREASE over rounds. An increasing trend is a
    pipeline bug, not a finding (CLAUDE.md sanity checks).
    """

    name = "round_to_round_angles"
    category = INVARIANT

    def __call__(self, ctx: ProbeContext) -> ProbeResult:
        values = {}
        if ctx.prev_aggregate is not None:
            for layer in ctx.aggregate.aggregate_state:
                prev_state = ctx.prev_aggregate.next_global.get(layer)
                if prev_state is None:
                    continue
                curr = ctx.consensus[layer]
                values[layer] = {
                    "col": geo.subspace_distance_from_bases(
                        geo.column_space_of_factor(curr.A),
                        geo.column_space_of_factor(prev_state.A),
                    ),
                    "row": geo.subspace_distance_from_bases(
                        geo.row_space_of_factor(curr.B),
                        geo.row_space_of_factor(prev_state.B),
                    ),
                }
        return ProbeResult(self.name, self.category, values)


@PROBES.register("alignment")
class SignedAlignmentProbe:
    """Signed cross-update alignment rho + cross-Gram eigenvalue spectrum (INVARIANT).

    The first probe that reads the SIGN the (col, row, sigma) triple discards: it tells
    dW from -dW, which principal angles cannot (CLAUDE.md domain model). Every output is
    a function of the two dW's alone, so it is gauge-invariant (rule 1), NOT equivariant
    like the det-sign obstruction. Logs three pairings per layer, from raw factors:

      * client_i <-> consensus        (the E2 predictor)
      * client_i^t <-> client_i^{t-1} (directional stability; needs prev-round factors)
      * client_i <-> client_j         (pairwise cross-client, symmetric)

    Cost is one r x r cross-Gram per pair, same order as the spectral probe. This
    information exists only in-flight (invariant 8): it cannot be recomputed from the
    stored spectra, which is exactly why the probe is added as its own batch.
    """

    name = "alignment"
    category = INVARIANT
    SPECTRUM_TOPK = 16  # eigenvalues logged per pair; r<=32 so this only truncates r=32

    def _full(self, s1, s2) -> dict:
        """rho + the full (top-k) cross-Gram eigenvalue spectrum."""
        return geo.alignment_spectrum(s1, s2, log_topk=self.SPECTRUM_TOPK)

    def _compact(self, s1, s2) -> dict:
        """rho + orientation only, no eigenvalue list.

        Used for the PAIRWISE set, which grows as N(N-1)/2: logging the per-mode
        spectrum for every client pair would dominate the record for information the
        scalar rho already summarizes. The full spectrum is kept for the two linear
        pairings (E2 predictor + directional stability), where the per-mode coupling is
        actually used.
        """
        full = geo.alignment_spectrum(s1, s2, log_topk=1)
        return {
            "rho": full["rho"],
            "orientation_sign": full["orientation_sign"],
            "det_normalized": full["det_normalized"],
        }

    def __call__(self, ctx: ProbeContext) -> ProbeResult:
        values = {}
        prev_by_id = None
        if ctx.prev_client_updates is not None:
            prev_by_id = {u.client_id: u for u in ctx.prev_client_updates}
        for layer in ctx.aggregate.aggregate_state:
            consensus = ctx.consensus[layer]
            ids = [u.client_id for u in ctx.client_updates]

            client_consensus = {
                str(u.client_id): self._full(u.adapters[layer], consensus)
                for u in ctx.client_updates
            }

            round_to_round = {}
            if prev_by_id is not None:
                for u in ctx.client_updates:
                    prev = prev_by_id.get(u.client_id)
                    if prev is not None and layer in prev.adapters:
                        round_to_round[str(u.client_id)] = self._full(
                            u.adapters[layer], prev.adapters[layer]
                        )

            pairwise = {}
            for i in range(len(ctx.client_updates)):
                for j in range(i + 1, len(ctx.client_updates)):
                    ui, uj = ctx.client_updates[i], ctx.client_updates[j]
                    pairwise[f"{ui.client_id}_{uj.client_id}"] = self._compact(
                        ui.adapters[layer], uj.adapters[layer]
                    )

            values[layer] = {
                "client_consensus": client_consensus,
                "round_to_round": round_to_round,
                "pairwise": pairwise,
                "client_ids": ids,
            }
        return ProbeResult(self.name, self.category, values)


@PROBES.register("detsign_obstruction")
class DetSignObstructionProbe:
    """Z/2 orientation obstruction: client frames vs consensus frame (EQUIVARIANT).

    Reported against BOTH reference frames (in-force consensus and the exact
    aggregate's rank-r canonical frame). The in-force number is the one with
    operational meaning: it is the false-positive rate a deployed det-sign monitor
    would see on benign traffic (H1a).
    """

    name = "detsign_obstruction"
    category = EQUIVARIANT  # rule 2: deliberately not gauge-invariant
    requires_canonical_frames = True

    def __call__(self, ctx: ProbeContext) -> ProbeResult:
        values = {}
        canonical = ctx.canonical_frames
        for layer in ctx.aggregate.aggregate_state:
            refs = {"consensus": ctx.consensus[layer].A}
            if canonical is not None and layer in canonical:
                refs["canonical"] = canonical[layer].A
            frames = [u.adapters[layer].A for u in ctx.client_updates]
            client_ids = [u.client_id for u in ctx.client_updates]

            layer_values = {}
            for ref_name, ref_frame in refs.items():
                summary = obs.orientation_obstruction(frames, ref_frame)
                summary["client_ids"] = client_ids
                summary["magnitudes"] = [
                    obs.relative_orientation_magnitude(f, ref_frame) for f in frames
                ]
                layer_values[ref_name] = summary
            values[layer] = layer_values
        return ProbeResult(self.name, self.category, values)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mu = _mean(xs)
    return (sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


@PROBES.register("factor_norms")
class FactorNormsProbe:
    """||A_bar||_F, ||B_bar||_F and the per-client factor norms. GAUGE-VARIANT.

    These are NOT functions of dW: they read the transmitted representation, which is
    exactly why they can see a gauge at all. CLAUDE.md invariant 1 therefore restricts
    them to being reported as DRIFT ALONG A GAUGE ORBIT — the attacked arm against its
    benign twin at the same round — never as a property of an update, a client or a
    round on its own. Every value here carries that restriction.

    Why it exists: E-A2's P1 asks whether a bounded (orthogonal) gauge keeps ||A_bar||_F
    inside the benign band, where the non-compact `scaling` gauge blew it up by 3.6e5x.
    Orthogonal T preserves Frobenius norm, so ||A_i T^-1||_F = ||A_i||_F and the triangle
    inequality caps ||A_bar||; a violation is an implementation bug, not a finding. That
    prediction cannot be checked from any invariant probe, because the quantity it is
    about is not invariant.

    Its test is a VARIANCE test (tests/test_gauge_invariance.py): it must move under a
    gauge for which theory says it moves (scaling) and NOT move under one for which
    theory says it does not (orthogonal). Both directions, or "gauge-variant" would be
    indistinguishable from "broken".
    """

    name = "factor_norms"
    category = GAUGE_VARIANT

    def __call__(self, ctx: ProbeContext) -> ProbeResult:
        from ..fl.aggregation.base import active_clients

        active, weights = active_clients(ctx.client_updates, ctx.weights)
        values = {}
        for layer in ctx.aggregate.aggregate_state:
            A_bar = sum(w * u.adapters[layer].A for w, u in zip(weights, active, strict=True))
            B_bar = sum(w * u.adapters[layer].B for w, u in zip(weights, active, strict=True))
            entry = {
                "gauge_variant": True,  # machine-readable: read as orbit drift only
                "aggregate_A_norm": float(A_bar.norm().item()),
                "aggregate_B_norm": float(B_bar.norm().item()),
                "aggregate_product_norm": geo.deltaw_frobenius_norm(A_bar, B_bar),
                "per_client_A_norm": {
                    str(u.client_id): float(u.adapters[layer].A.norm().item()) for u in active
                },
            }
            if ctx.coalition_ids:
                entry.update(_lever_terms(active, weights, layer, ctx.coalition_ids))
            values[layer] = entry
        return ProbeResult(self.name, self.category, values)


def _lever_terms(active, weights, layer, coalition_ids) -> dict:
    """The cross-term lever ||M1||_2 ||M2||_F and ||M3||_2 ||M4||_F (E-A3's mechanism check).

    For a shared coalition gauge T the aggregate's displacement is

        dA(T) = M1 (T^-1 - I) M2 + M3 (T - I) M4,
        M1 = sum_{i in K} q_i A_i,  M2 = sum_{j in H} q_j B_j,
        M3 = sum_{j in H} q_j A_j,  M4 = sum_{i in K} q_i B_i,

    bounded by 2(||M1||_2 ||M2||_F + ||M3||_2 ||M4||_F). E-A3 Stage A asks whether that
    lever GROWS with convergence — it scales with the accumulated global dW rather than
    with the per-round gradient, which is the "late favours the attacker" mechanism.
    Logging it costs one spectral norm of an n x r matrix per layer.

    GAUGE-VARIANT, like everything else this probe reports: M1..M4 are built from the
    transmitted factors, not from dW.
    """
    import torch

    coalition = set(coalition_ids)
    K = [(w, u) for w, u in zip(weights, active, strict=True) if u.client_id in coalition]
    H = [(w, u) for w, u in zip(weights, active, strict=True) if u.client_id not in coalition]
    if not K or not H:
        return {}

    def _sum(items, part):
        return sum(w * getattr(u.adapters[layer], part) for w, u in items)

    M1, M2 = _sum(K, "A").double(), _sum(H, "B").double()
    M3, M4 = _sum(H, "A").double(), _sum(K, "B").double()
    m1 = float(torch.linalg.matrix_norm(M1, ord=2).item())
    m3 = float(torch.linalg.matrix_norm(M3, ord=2).item())
    m2, m4 = float(M2.norm().item()), float(M4.norm().item())
    return {
        "lever_KH": m1 * m2,  # ||M1||_2 ||M2||_F
        "lever_HK": m3 * m4,  # ||M3||_2 ||M4||_F
        "lever_bound": 2.0 * (m1 * m2 + m3 * m4),
        "coalition_size": len(K),
    }
