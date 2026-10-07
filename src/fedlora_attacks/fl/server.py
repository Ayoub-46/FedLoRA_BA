"""FL server: the round loop with per-round probe + metric hooks (CLAUDE.md "fl/").

Each round: broadcast global adapters -> clients train locally -> the defense SCORES
every client and returns weights q_i -> aggregate under q -> run probes on (client
updates, aggregate, previous aggregate) -> evaluate main-task accuracy and ASR on the
aggregated model -> log everything to the run dir. The naive-vs-exact aggregation
discrepancy (§1 gap) is logged every round.

The defence never filters the update list (plan v3 Phase 1.1): it returns weights, and
rejection is the weight-0 special case. ONE weight vector q drives every weighted
quantity in the round — both aggregators, the head FedAvg, the optional canonical frame, the
naive-vs-exact gap, and the probes' reconstruction of the exact aggregate. They used to
recompute p_i independently, which was harmless only while no defence reweighted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ..attacks.base import Attack
from ..data.base import TextClassificationData
from ..defenses.base import Defense, DefenseContext
from ..metrics.evaluation import accuracy, attack_success_rate
from ..metrics.generation import response_perplexity
from ..metrics.outcome import outcome_metrics
from ..metrics.persistence import PersistenceCurve
from ..models.lora_wrapper import LoRAModel
from ..probes.base import Probe, ProbeContext, run_probes
from ..utils.logging import RunLogger, get_logger
from .aggregation.base import (
    Aggregation,
    canonical_frames,
    deltaw_discrepancy,
    normalized_weights,
)
from .client import LoRAClient
from .types import AggregateResult, GlobalState

log = get_logger()


def aggregate_or_retain(aggregation, updates, weights, broadcast):
    """Aggregate positive-weight reports, or retain the previous broadcast if none exist."""
    if any(float(weight) > 0.0 for weight in weights):
        return aggregation.aggregate(updates, weights), False
    return (
        AggregateResult(
            aggregate_state={k: v.clone() for k, v in broadcast.items()},
            next_global={k: v.clone() for k, v in broadcast.items()},
            strategy=aggregation.name,
            diagnostics={"skipped_aggregation": True},
        ),
        True,
    )


@dataclass
class ServerConfig:
    num_rounds: int = 10
    clients_per_round: int | None = None  # None -> all clients each round
    eval_batch_size: int = 64
    eval_examples: int | None = None
    """Held-out IMAGES scored per round; None = the whole split. Ignored by text runs.

    Separate from `generation_prompts`, which gates DECODING — orders of magnitude more
    expensive than a forward pass and therefore sampled. This gates one forward pass.
    """
    persist_canonical_frames: bool = False
    """Write the exact aggregate's rank-r canonical frame to canonical_frames.pt.

    Gives post-hoc analyses (E1 re-derivations, E4 trajectory work) a gauge-canonical
    reference per round without re-running the sweep. Costs ~2.2 MB * r per 15-round
    run at bert-base scale, so it is opt-in.
    """
    persist_client_factors: bool = False
    """Write raw per-client per-layer (A,B) to client_factors.pt (the probe-dev set).

    CLAUDE.md invariant 8: a new gauge invariant cannot be recovered from stored probe
    outputs, so a small slice of the RAW factors is kept to prototype future invariants
    offline against real trained updates. Large (~18.4*r MB / run), so enabled only for
    the one-seed dev slice, never the full sweep.
    """
    probe_coalition_ids: tuple[int, ...] = ()
    """Client ids the probes should treat as the coalition (measurement partition).

    Defaults to the attack's malicious ids in the runner. Set explicitly to measure the
    same K/H split in a BENIGN twin, where no client is malicious.
    """
    generation_every: int = 5
    """Rounds between generation evaluations (BLEU / BERTScore), causal-LM runs only.

    Decoding costs far more than a forward pass, so the smooth per-round signal is
    perplexity and generation is sampled. The FINAL round is always evaluated regardless
    of the stride, because that is the number the paper reports.
    """
    generation_prompts: int = 150
    """Held-out prompts decoded per generation evaluation."""
    generation_max_new_tokens: int = 128
    bertscore_model: str = "roberta-large"
    generation_final_prompts: int | None = None
    """Prompts decoded at the FINAL round only. None = same as ``generation_prompts``.

    A trajectory and a headline need different sample sizes. 300 examples give +/-2.9pp
    binomial error at 50% execution accuracy — enough to watch a curve move, not enough to
    be quoted against another run. The final round pays for the full held-out split once,
    and that is the number a centralized comparison uses.
    """
    db_root: str | None = None
    """Root of the text-to-SQL databases; None for every other task.

    Its presence is what selects execution accuracy over BLEU/BERTScore — see
    ``metrics/outcome.py``. Keyed on capability rather than on a dataset name so a config
    that forgot it fails at the first evaluation instead of scoring SQL with BLEU.
    """
    sql_timeout_s: float = 10.0

    persist_client_factors_rounds: int | None = None
    """Persist raw factors for the FIRST k rounds only (None = every round).

    What makes the paired dW comparison affordable at sweep scale: the round-0
    bit-identity control needs exactly one round of raw factors per arm, and paying
    ~18.4*r MB per run for 25 rounds to read one of them is not a trade worth making.
    """

    persist_client_factors_at_rounds: tuple[int, ...] = ()
    """Persist isolated round snapshots to ``client_factor_snapshots.pt``.

    Unlike ``persist_client_factors_rounds``, this selects exact zero-based round indices
    without retaining every preceding round. The separate filename and round-keyed schema
    keep the longstanding prefix checkpoint contract unambiguous.
    """

    def __post_init__(self) -> None:
        rounds = tuple(int(round_idx) for round_idx in self.persist_client_factors_at_rounds)
        if len(rounds) != len(set(rounds)):
            raise ValueError("persist_client_factors_at_rounds contains duplicate rounds")
        if any(round_idx < 0 or round_idx >= self.num_rounds for round_idx in rounds):
            raise ValueError(
                "persist_client_factors_at_rounds must contain zero-based indices in "
                f"[0, {self.num_rounds})"
            )
        if rounds and not self.persist_client_factors:
            raise ValueError(
                "persist_client_factors_at_rounds requires persist_client_factors=true"
            )
        self.persist_client_factors_at_rounds = tuple(sorted(rounds))


def nonfinite_clients(updates) -> list[int]:
    """Client ids whose transmitted factors contain a non-finite entry.

    A sufficiently strong Channel A attack does not merely collapse the model: the
    factor-averaged global grows by ~(kappa/N)*c per round, and once the adapters are
    large enough the forward pass overflows and NaN propagates back into the factors.
    That is a legitimate OUTCOME of an extreme gauge, not an error to crash on — the run
    has diverged, and the sweep needs to record that and carry on rather than die and
    take the remaining cells with it.
    """
    bad = []
    for u in updates:
        if any(
            not torch.isfinite(s.A).all() or not torch.isfinite(s.B).all()
            for s in u.adapters.values()
        ):
            bad.append(u.client_id)
    return bad


@dataclass
class RunResult:
    rounds: list[int]
    accuracy: list[float]
    asr: list[float | None]
    persistence: PersistenceCurve
    final_accuracy: float
    final_asr: float | None
    metrics: dict[str, list[float]] = field(default_factory=dict)
    """Task-appropriate per-round series, by name (``eval_perplexity``, ``train_loss``...).

    Kept beside ``accuracy`` rather than replacing it so the classification runs, their
    stored summaries and every analysis that reads ``final_accuracy`` keep working
    unchanged. For a causal-LM run ``accuracy`` stays empty and the reportable trajectory
    lives here."""
    run_dir: str | None = None
    per_round: list[dict] = field(default_factory=list)
    diverged_at_round: int | None = None
    """Round at which some client's factors went non-finite, if any.

    None means the run completed its full schedule. An integer means the run STOPPED
    there: everything downstream of non-finite factors is meaningless, so the loop exits
    and the recorded series ends at the last valid round.
    """


class FederatedServer:
    def __init__(
        self,
        model: LoRAModel,
        aggregation: Aggregation,
        defense: Defense,
        probes: list[Probe],
        cfg: ServerConfig,
        device="cpu",
        logger: RunLogger | None = None,
        trusted_client: LoRAClient | None = None,
    ) -> None:
        self.model = model
        self.aggregation = aggregation
        self.defense = defense
        self.probes = probes
        self.cfg = cfg
        self.device = device
        self.logger = logger
        self.trusted_client = trusted_client
        # Gold SQL result sets never change across rounds, so one cache spans the run —
        # the difference between executing two queries per example per evaluation and one.
        # None for every non-SQL task, where it is simply never consulted.
        self._gold_cache = None
        if cfg.db_root:
            from ..metrics.execution import GoldCache

            self._gold_cache = GoldCache()

    @staticmethod
    def _aggregate_heads(updates, weights: list[float]) -> dict | None:
        """Weighted FedAvg of the non-LoRA head params under the defence weights q.

        q REPLACES the head's former num_examples weighting rather than multiplying into
        it — one normalisation for the whole round (plan v3 Phase 1.1). A downweighted
        client is downweighted everywhere, and the head sits outside the gauge bundle so
        nothing geometric objects.

        The one place the head must differ from the adapters: q is renormalized to sum 1
        HERE and nowhere else. The adapters are the model's entire deviation from the
        frozen base, so clipping them to sum(q) < 1 legitimately shrinks dW toward the
        base — that is the norm bound biting, and it is the measurement. The head is an
        ABSOLUTE parameter with no such zero reference; scaling it by sum(q) < 1 would
        decay the classifier toward zero and compound over rounds, manufacturing an
        accuracy drop that is an artifact of the defence's scale rather than of the
        attack. Relative weights are the only meaning q has for the head.
        (experiment/e_a_frozen_head exists to show the choice does not carry the result.)

        Returns None when no client carries positive weight, leaving the previous global
        head in force rather than zeroing the classifier.
        """
        heads = [
            (u.head_state, w)
            for u, w in zip(updates, weights, strict=True)
            if u.head_state is not None and w > 0.0
        ]
        if not heads:
            return None
        total = sum(w for _, w in heads)
        if total <= 0.0:
            return None
        keys = heads[0][0].keys()
        return {k: sum((w / total) * hs[k] for hs, w in heads) for k in keys}

    def _select(self, clients, round_idx, rng):
        k = self.cfg.clients_per_round
        if k is None or k >= len(clients):
            return clients
        chosen = rng.choice(len(clients), size=k, replace=False)
        return [clients[int(i)] for i in chosen]

    def run(
        self,
        clients: list[LoRAClient],
        data: TextClassificationData,
        attack: Attack,
        seed: int = 0,
    ) -> RunResult:
        rng = np.random.default_rng(seed)
        global_adapters = self.model.get_adapters()  # zero-dW LoRA init
        global_head = self.model.get_head_state()  # federated classifier head
        prev_aggregate: AggregateResult | None = None

        rounds, acc_series, asr_series, per_round = [], [], [], []
        metric_series: dict[str, list[float]] = {}
        diverged_at: int | None = None
        canonical_history: list[dict] = []
        needs_canonical = self.cfg.persist_canonical_frames or any(
            getattr(probe, "requires_canonical_frames", False) for probe in self.probes
        )
        client_factor_history: list[dict] = []
        client_factor_snapshots: dict[int, dict] = {}
        prev_updates: list | None = None
        for r in range(self.cfg.num_rounds):
            selected = self._select(clients, r, rng)
            # Per-client isolated RNG, deterministic in (seed, round, client_id) for BOTH
            # numpy (data order / poisoning) AND torch (dropout). This is what makes E2's
            # paired design exact: a malicious client consuming RNG for poisoning or a
            # steering penalty cannot shift the stochasticity of the benign clients, so
            # clients 1..N-1 are BIT-IDENTICAL between the attacked run and its
            # benign-control twin (same seed) — only the manipulated client differs.
            # Also the correct FL semantics: each client's randomness is its own.
            updates = []
            train_losses: list[float] = []
            for c in selected:
                cseed = np.random.default_rng([seed, r, c.client_id])
                torch.manual_seed(int(cseed.integers(2**63)))  # isolate dropout etc.
                updates.append(
                    c.local_train(
                        self.model,
                        global_adapters,
                        r,
                        self.device,
                        cseed,
                        global_head=global_head,
                    )
                )
                # The client's own mean over its local steps. Averaged across clients it
                # is the federation's training-loss trajectory, which the instruction-
                # tuning arm reports beside held-out perplexity: a run where training
                # loss falls while held-out perplexity rises is overfitting, not damage,
                # and only having both distinguishes them.
                loss = getattr(c, "last_train_loss", float("nan"))
                if loss == loss:  # NaN-safe: an untrained (empty-shard) client is skipped
                    train_losses.append(float(loss))
            diverged = nonfinite_clients(updates)
            if diverged:
                log.error(
                    "round %d | DIVERGED: non-finite factors from client(s) %s — stopping "
                    "this run. Everything downstream of non-finite factors is meaningless; "
                    "the recorded series ends at round %d.",
                    r,
                    diverged,
                    r - 1,
                )
                diverged_at = r
                break

            trusted_update = None
            if self.trusted_client is not None:
                # The root stream is independent of every client and of coalition draws.
                # The client helper resets the shared model to ``global_adapters`` before
                # training, so running it after the ordinary clients cannot inherit the
                # last client's parameters.
                root_rng = np.random.default_rng([seed, r, 0xF17A57])
                torch.manual_seed(int(root_rng.integers(2**63)))
                trusted_update = self.trusted_client.local_train(
                    self.model,
                    global_adapters,
                    r,
                    self.device,
                    root_rng,
                    global_head=global_head,
                )
                trusted_update.metadata["train_loss"] = float(self.trusted_client.last_train_loss)
                root_nonfinite = nonfinite_clients([trusted_update])
                if root_nonfinite:
                    raise RuntimeError(
                        f"round {r}: trusted root training produced non-finite factors"
                    )

            # The defence SCORES; it never filters. q_i = 0 is how it rejects, and the
            # same q drives every weighted quantity below.
            defense_result = self.defense.apply(
                updates,
                DefenseContext(
                    round_idx=r,
                    base_weights=normalized_weights(updates),
                    trusted_update=trusted_update,
                    broadcast_adapters=global_adapters,
                ),
            )
            q = [float(w) for w in defense_result.weights]
            aggregate, skipped_aggregation = aggregate_or_retain(
                self.aggregation, updates, q, global_adapters
            )
            aggregated_head = None if skipped_aggregation else self._aggregate_heads(updates, q)

            # The exact rank-r SVD is only needed for canonical-reference probes or
            # persistence. Other probes and the in-force aggregator never consume it.
            canonical = None
            if needs_canonical:
                canonical = (
                    {k: v.clone() for k, v in global_adapters.items()}
                    if skipped_aggregation
                    else canonical_frames(updates, q)
                )
            if self.cfg.persist_canonical_frames:
                canonical_history.append({k: (v.A.cpu(), v.B.cpu()) for k, v in canonical.items()})
            # Probe-development set (invariant 8): raw per-client (A,B) for a small slice,
            # so future invariants can be prototyped offline against real trained factors.
            keep_factors = self.cfg.persist_client_factors and (
                self.cfg.persist_client_factors_rounds is None
                or r < self.cfg.persist_client_factors_rounds
            )
            keep_snapshot = r in self.cfg.persist_client_factors_at_rounds
            if keep_factors or keep_snapshot:
                factor_snapshot = {
                    u.client_id: {k: (s.A.cpu(), s.B.cpu()) for k, s in u.adapters.items()}
                    for u in updates
                }
                if keep_factors:
                    client_factor_history.append(factor_snapshot)
                if keep_snapshot:
                    client_factor_snapshots[r] = factor_snapshot

            ctx = ProbeContext(
                updates,
                aggregate,
                r,
                prev_aggregate,
                prev_client_updates=prev_updates,
                weights=q,
                coalition_ids=tuple(self.cfg.probe_coalition_ids),
                canonical_frames=canonical,
            )
            probe_results = {name: pr.values for name, pr in run_probes(ctx, self.probes).items()}
            discrepancy = (
                {layer: 0.0 for layer in global_adapters}
                if skipped_aggregation
                else deltaw_discrepancy(updates, q)
            )

            # evaluate the aggregated (consensus) model
            self.model.set_adapters(aggregate.next_global)
            if aggregated_head is not None:
                self.model.set_head_state(aggregated_head)
            causal = getattr(data, "test_responses", None) is not None
            image = getattr(data, "modality", "text") == "image"
            task_metrics: dict[str, float] = {}
            if image:
                # THE SAME `evaluate_images` central/baseline.py calls, imported rather
                # than reimplemented, so top1_accuracy means bit-for-bit what it means
                # there — the two runs exist to be overlaid.
                from ..metrics.classification import evaluate_images

                task_metrics.update(
                    evaluate_images(
                        self.model,
                        data.test_images,
                        data.test_labels,
                        data.num_labels,
                        self.device,
                        batch_size=self.cfg.eval_batch_size,
                        limit=self.cfg.eval_examples,
                    )
                )
                # Unlike the causal-LM path, `accuracy` is DEFINED here and is the headline:
                # top-1 IS main-task accuracy. Filling it keeps RunResult.final_accuracy and
                # every analysis that reads it working unchanged on the vision arm.
                acc = float(task_metrics["top1_accuracy"])
                asr = None
                if train_losses:
                    task_metrics["train_loss"] = float(sum(train_losses) / len(train_losses))
            elif causal:
                # Accuracy is undefined for a generative task; the trajectory is
                # perplexity, and it runs the other way (lower is better). The damage
                # sign is looked up from metrics.generation.HIGHER_IS_BETTER rather than
                # re-derived wherever a comparison happens.
                acc, asr = 0.0, None
                task_metrics.update(
                    response_perplexity(
                        self.model,
                        data.test_texts,
                        data.test_responses,
                        self.device,
                        self.cfg.eval_batch_size,
                    )
                )
                if train_losses:
                    task_metrics["train_loss"] = float(sum(train_losses) / len(train_losses))
                last_round = r == self.cfg.num_rounds - 1
                if self.cfg.generation_every > 0 and (
                    last_round or r % self.cfg.generation_every == 0
                ):
                    want = self.cfg.generation_prompts
                    if last_round and self.cfg.generation_final_prompts:
                        want = self.cfg.generation_final_prompts
                    n = min(want, len(data.test_texts))
                    # The SHARED dispatch, the same call central/baseline.py makes. The two
                    # runs exist to be compared, so "decode, extract, execute, compare" is
                    # one body of code rather than two that could drift apart invisibly.
                    gen = outcome_metrics(
                        self.model,
                        data,
                        n,
                        self.device,
                        max_new_tokens=self.cfg.generation_max_new_tokens,
                        batch_size=self.cfg.eval_batch_size,
                        bertscore_model=self.cfg.bertscore_model,
                        db_root=self.cfg.db_root,
                        sql_timeout_s=self.cfg.sql_timeout_s,
                        gold_cache=self._gold_cache,
                    )
                    task_metrics.update({k: v for k, v in gen.items() if v is not None})
                    task_metrics["generation_sample"] = float(n)
            else:
                acc = accuracy(
                    self.model,
                    data.test_texts,
                    data.test_labels,
                    self.device,
                    self.cfg.eval_batch_size,
                )
                asr = attack_success_rate(
                    self.model,
                    data.test_texts,
                    data.test_labels,
                    attack,
                    self.device,
                    self.cfg.eval_batch_size,
                )

            record = {
                "round": r,
                "accuracy": acc,
                "asr": asr,
                **task_metrics,
                "num_clients": len(selected),
                "num_active_attackers": sum(1 for u in updates if u.metadata.get("active")),
                "skipped_aggregation": skipped_aggregation,
                "aggregation_discrepancy": discrepancy,
                "aggregation": {
                    "name": aggregate.strategy,
                    **aggregate.diagnostics,
                },
                # E-B's data source: the defence's per-client STATISTICS (not just its
                # verdicts) in every run, so the paired benign/attacked comparison can
                # assert they are bit-identical.
                "defense": {
                    "name": self.defense.config.name,
                    "client_ids": [u.client_id for u in updates],
                    **defense_result.to_record(),
                },
                # Per-client attack metadata. E-A's control lives here: an update-space
                # attack records its own achieved dW displacement (zero in exact
                # arithmetic, ~eps*cond(T) in floating point), so the "every dW_i is
                # what an honest client would have sent" claim is MEASURED per round
                # rather than assumed.
                "attack": {
                    "name": attack.config.name,
                    "per_client": {str(u.client_id): dict(u.metadata) for u in updates},
                },
                "probes": probe_results,
            }
            if torch.cuda.is_available():
                # Measured, not assumed: this is what decides whether the run fits a 46 GB
                # A40 as well as a 49 GB A6000.
                record["gpu_peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
                torch.cuda.reset_peak_memory_stats()
            if self.logger is not None:
                self.logger.log(record)
            if causal:
                log.info(
                    "round %d | ppl=%.4f | loss=%.4f | disc=%.3e",
                    r,
                    task_metrics.get("eval_perplexity", float("nan")),
                    task_metrics.get("eval_loss", float("nan")),
                    max(discrepancy.values()) if discrepancy else 0.0,
                )
            else:
                log.info(
                    "round %d | acc=%.3f | asr=%s | disc=%.3e",
                    r,
                    acc,
                    f"{asr:.3f}" if asr is not None else "n/a",
                    max(discrepancy.values()) if discrepancy else 0.0,
                )

            rounds.append(r)
            acc_series.append(acc)
            asr_series.append(asr)
            for key, value in task_metrics.items():
                metric_series.setdefault(key, []).append(float(value))
            per_round.append(record)
            global_adapters = aggregate.next_global
            if aggregated_head is not None:
                global_head = aggregated_head
            prev_aggregate = aggregate
            prev_updates = updates
            # Release cached-but-unused blocks between rounds. Semantically a no-op —
            # it frees nothing that is still referenced — but it decides whether this run
            # fits on a 46 GB A40 as well as a 49 GB A6000. Job 10382_5 died on an A40
            # needing 3.94 GB with 3.77 GB free: short by ~180 MB, while 761 MB sat
            # reserved-but-unallocated. Keeping both card types usable matters more than
            # the microseconds this costs: the A6000 node is routinely full while the A40
            # node sits half idle.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if self.cfg.persist_canonical_frames and self.logger is not None and canonical_history:
            torch.save(canonical_history, self.logger.run_dir / "canonical_frames.pt")
        if self.cfg.persist_client_factors and self.logger is not None and client_factor_history:
            torch.save(client_factor_history, self.logger.run_dir / "client_factors.pt")
        if self.logger is not None and client_factor_snapshots:
            torch.save(
                client_factor_snapshots,
                self.logger.run_dir / "client_factor_snapshots.pt",
            )

        curve = PersistenceCurve(
            rounds=rounds,
            asr=[a if a is not None else 0.0 for a in asr_series],
            stop_round=attack.config.attack_stop_round,
        )
        return RunResult(
            metrics=metric_series,
            rounds=rounds,
            accuracy=acc_series,
            asr=asr_series,
            persistence=curve,
            final_accuracy=acc_series[-1] if acc_series else 0.0,
            final_asr=asr_series[-1] if asr_series else None,
            run_dir=str(self.logger.run_dir) if self.logger else None,
            per_round=per_round,
            diverged_at_round=diverged_at,
        )

    @staticmethod
    def initial_global_state(model: LoRAModel) -> GlobalState:
        return GlobalState(round=0, adapters=model.get_adapters())
