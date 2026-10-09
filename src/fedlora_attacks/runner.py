"""Compose a run from task, model, FL, attack, defense, and aggregation configs."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from .attacks.base import Attack, AttackConfig, build_attack
from .attacks.x4_backdoor import X4Config
from .data.base import build_federated_data, build_tokenizer
from .defenses.aggregate_norm_check import AggregateNormCheckConfig
from .defenses.base import Defense, DefenseConfig, build_defense
from .defenses.cosdefense import CosDefenseConfig
from .defenses.factor_broadcast_cosine import FactorBroadcastCosineConfig
from .defenses.factor_broadcast_topk import FactorBroadcastTopKConfig
from .defenses.fltrust import FLTrustConfig, reserve_trusted_root
from .defenses.multi_krum import MultiKrumConfig
from .defenses.norm_bound import NormBoundConfig
from .defenses.spectral_check import SpectralCheckConfig
from .fl.aggregation.base import Aggregation, build_aggregation
from .fl.client import ClientConfig, LoRAClient
from .fl.checkpoint import (
    frozen_base_sha256,
    load_global_checkpoint,
    load_global_checkpoint_context,
    validate_global_checkpoint,
)
from .fl.server import FederatedServer, RunResult, ServerConfig
from .models.lora_wrapper import CLIP_ZEROSHOT_CLS, ModelSpec, build_lora_model
from .probes.base import build_probes
from .utils.config_hash import hash_config
from .utils.logging import RunLogger, get_logger
from .utils.manifest import Manifest
from .utils.seeding import seed_everything

log = get_logger()


def build_attack_from_cfg(a: DictConfig) -> Attack:
    common = dict(
        name=a.name,
        malicious_client_ids=tuple(a.get("malicious_client_ids", []) or []),
        attack_start_round=int(a.get("attack_start_round", 0)),
        attack_stop_round=a.get("attack_stop_round", None),
        active_rounds=(
            None
            if a.get("active_rounds", None) is None
            else tuple(int(round_idx) for round_idx in a.active_rounds)
        ),
    )
    if a.name == "none":
        cfg: AttackConfig = AttackConfig(**common)
    elif a.name == "x4_backdoor":
        cfg = X4Config(
            **common,
            trigger=a.trigger,
            target_label=a.target_label,
            poison_fraction=a.poison_fraction,
            poison_mode=str(a.get("poison_mode", "append")),
        )
    else:
        raise KeyError(f"unknown attack '{a.name}'")
    return build_attack(cfg)


def build_defense_from_cfg(d: DictConfig) -> Defense:
    if d.name == "none":
        return build_defense(DefenseConfig(name="none"))
    if d.name == "norm_bound":
        return build_defense(
            NormBoundConfig(
                threshold_mode=d.get("threshold_mode", "median"),
                gamma=float(d.get("gamma", 1.5)),
                max_delta_norm=float(d.get("max_delta_norm", 1.0)),
                per_layer=bool(d.get("per_layer", False)),
            )
        )
    if d.name == "multi_krum":
        return build_defense(
            MultiKrumConfig(
                num_selected=int(d.get("num_selected", 3)),
                num_byzantine=int(d.get("num_byzantine", 1)),
            )
        )
    if d.name == "aggregate_norm_check":
        return build_defense(
            AggregateNormCheckConfig(
                tolerance=float(d.get("tolerance", 2.0)),
                tolerance_low=float(d.get("tolerance_low", 2.0)),
                baseline_momentum=float(d.get("baseline_momentum", 0.9)),
                baseline_half_life=(
                    None if d.get("baseline_half_life") is None else float(d.baseline_half_life)
                ),
                direction=d.get("direction", "inflation"),
                action=d.get("action", "monitor"),
                warmup_rounds=int(d.get("warmup_rounds", 1)),
            )
        )
    if d.name == "cosdefense":
        return build_defense(
            CosDefenseConfig(
                threshold_mode=d.get("threshold_mode", "median"),
                min_cosine=float(d.get("min_cosine", 0.0)),
                gamma=float(d.get("gamma", 1.5)),
                action=d.get("action", "reject"),
            )
        )
    if d.name == "spectral_check":
        return build_defense(
            SpectralCheckConfig(
                statistic=d.get("statistic", "nuclear_ratio"),
                z_threshold=float(d.get("z_threshold", 3.0)),
                direction=d.get("direction", "two_sided"),
                per_layer=bool(d.get("per_layer", False)),
            )
        )
    if d.name == "fltrust":
        return build_defense(
            FLTrustConfig(
                root_size=int(d.get("root_size", 100)),
                root_seed=int(d.get("root_seed", 0)),
                root_sampling=d.get("root_sampling", "stratified"),
                client_weighting=d.get("client_weighting", "paper_uniform"),
            )
        )
    if d.name == "factor_broadcast_cosine":
        return build_defense(
            FactorBroadcastCosineConfig(
                threshold=float(d.get("threshold", 2.0)),
                action=str(d.get("action", "reject")),
            )
        )
    if d.name == "factor_broadcast_topk":
        return build_defense(
            FactorBroadcastTopKConfig(
                score_mode=str(d.get("score_mode", "cosine")),
                num_reject=int(d.get("num_reject", 2)),
                euclidean_epsilon=float(d.get("euclidean_epsilon", 1e-12)),
                tie_break_seed=int(d.get("tie_break_seed", 0)),
            )
        )
    raise KeyError(f"unknown defense '{d.name}'")


def build_aggregation_from_cfg(a: DictConfig) -> Aggregation:
    """Build an aggregation strategy without silently dropping declared options."""
    if a.name not in ("consensus_frame", "consensus_frame_magnitude"):
        return build_aggregation(a.name)
    kwargs = dict(
        server_rank=(None if a.get("server_rank") is None else int(a.server_rank)),
        coordinate_rule=str(a.get("coordinate_rule", "mean")),
        trim_fraction=float(a.get("trim_fraction", 0.1)),
        geometric_median_max_iterations=int(a.get("geometric_median_max_iterations", 50)),
        geometric_median_tolerance=float(a.get("geometric_median_tolerance", 1e-6)),
    )
    return build_aggregation(a.name, **kwargs)


def _model_spec(m: DictConfig, tokenizer_kind: str) -> ModelSpec:
    return ModelSpec(
        kind=m.kind,
        model_name=m.get("model_name", None),
        hidden_size=m.get("hidden_size", 64),
        num_hidden_layers=m.get("num_hidden_layers", 2),
        num_attention_heads=m.get("num_attention_heads", 2),
        intermediate_size=m.get("intermediate_size", 128),
        max_position_embeddings=m.get("max_position_embeddings", 64),
        lora_r=m.lora_r,
        lora_alpha=m.lora_alpha,
        lora_dropout=m.get("lora_dropout", 0.0),
        target_modules=tuple(m.target_modules),
        tokenizer_kind=tokenizer_kind,
        amp=m.get("amp", None),
        freeze_head=bool(m.get("freeze_head", False)),
        task=m.get("task", "seq_cls"),
        base_dtype=m.get("base_dtype", "fp32"),
        max_prompt_tokens=int(m.get("max_prompt_tokens", 384)),
        image_size=int(m.get("image_size", 224)),
        lora_target_scope=m.get("lora_target_scope", None),
    )


def _spider_db_root(cfg: DictConfig) -> str | None:
    """The Spider database root, or None for a task that executes nothing."""
    if cfg.data.get("name", "") != "spider":
        return None
    from .data.spider import default_db_root

    return str(default_db_root())


def run_experiment(cfg: DictConfig, experiment_name: str = "run") -> RunResult:
    seed = int(cfg.seed)
    seed_everything(seed)
    device = cfg.get("device", "cpu")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg.device = device

    # Torch's default intra-op thread count (~#cores) badly over-subscribes the small
    # per-layer ops of a tiny model; cap it (see configs/config.yaml).
    n_threads = cfg.get("torch_num_threads", None)
    if n_threads:
        torch.set_num_threads(int(n_threads))
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "device=cuda requested but torch reports no CUDA device. Install the CUDA "
            "torch build (see README) or pass device=cpu."
        )

    # -- data + partition -------------------------------------------------
    fed = build_federated_data(
        name=cfg.data.name,
        num_train=cfg.data.get("num_train", None),
        num_test=cfg.data.get("num_test", None),
        num_clients=cfg.fl.num_clients,
        alpha=float(cfg.fl.alpha),
        seed=seed,
        num_labels=cfg.data.get("num_labels", 2),
        # Fixed across the whole sweep so every cell trains on the same examples.
        subsample_seed=cfg.data.get("subsample_seed", None),
        # Spider budgets its schema against THIS number, in the loader, before any
        # tokenizer exists. Threaded through so a federated run builds the IDENTICAL
        # prompts a centralized one does — the two are compared, and a differing budget
        # would change the truncation rate and therefore what is being compared, silently.
        max_prompt_tokens=int(cfg.model.get("max_prompt_tokens", 384)),
        # Read with a default rather than added to configs/fl/default.yaml: putting it in
        # the shared defaults would change the resolved config of EVERY experiment and so
        # every config hash, orphaning every completed run on disk.
        partition_scheme=str(cfg.fl.get("partition", "dirichlet")),
        server_clean_size=int(cfg.data.get("server_clean_size", 0)),
        validation_size=int(cfg.data.get("validation_size", 0)),
    )

    if cfg.attack.name == "x4_backdoor" and fed.data.task != "seq_cls":
        raise ValueError(
            "x4_backdoor currently supports text classification only; "
            f"dataset {cfg.data.name!r} has task {fed.data.task!r}"
        )
    if cfg.attack.name == "x4_backdoor":
        target = int(cfg.attack.target_label)
        if not 0 <= target < fed.data.num_labels:
            raise ValueError(
                f"backdoor target_label={target} outside dataset label range "
                f"[0, {fed.data.num_labels})"
            )

    # FLTrust's clean server data is a separate trust anchor, never another client.
    # Reserve it before constructing clients so no example appears on both sides. This
    # branch is absent for every existing defense, preserving their partitions exactly.
    root_shard = None
    root_record = None
    if cfg.defense.name == "fltrust":
        fed, root_shard = reserve_trusted_root(
            fed,
            size=int(cfg.defense.get("root_size", 100)),
            seed=int(cfg.defense.get("root_seed", 0)),
            sampling=cfg.defense.get("root_sampling", "stratified"),
        )
        root_record = root_shard.to_record(
            sampling=cfg.defense.get("root_sampling", "stratified"),
            seed=int(cfg.defense.get("root_seed", 0)),
        )

    # -- attack (needs to reserve trigger tokens before tokenizer build) --
    attack = build_attack_from_cfg(cfg.attack)
    if fed.data.vocab is not None:
        for tok in attack.extra_vocab():
            if tok not in fed.data.vocab:
                fed.data.vocab.append(tok)

    # -- model/adapter ----------------------------------------------------
    tokenizer = build_tokenizer(cfg.model.tokenizer_kind, fed.data, cfg.model.get("model_name"))
    spec = _model_spec(cfg.model, cfg.model.tokenizer_kind)
    # A zero-shot CLIP classifier has no head until the class names are known — the
    # classifier IS their text embedding. Same construction as central/baseline.py, so the
    # federated and centralized arms cannot end up with different frozen heads.
    extra: dict = {}
    if spec.task == CLIP_ZEROSHOT_CLS:
        from .models.clip_head import zeroshot_prompt_names

        if not fed.data.label_names:
            raise ValueError("a zero-shot CLIP run needs data.label_names to build its head")
        extra["class_names"] = zeroshot_prompt_names(fed.data.label_names)
    model = build_lora_model(
        spec, tokenizer, fed.data.num_labels, max_length=cfg.model.get("max_length", 32), **extra
    ).to(device)
    expected_lora_layers = cfg.model.get("expected_lora_layers", None)
    if expected_lora_layers is not None and model.num_lora_layers() != int(expected_lora_layers):
        raise RuntimeError(
            f"model target scope produced {model.num_lora_layers()} LoRA layers; "
            f"the configuration requires {int(expected_lora_layers)}"
        )
    initial_adapters = None
    initial_head = None
    start_round = 0
    source_checkpoint = cfg.get("initial_checkpoint", None)
    if source_checkpoint:
        saved_round, initial_adapters, initial_head = load_global_checkpoint(source_checkpoint)
        validate_global_checkpoint(model, initial_adapters, initial_head)
        start_round = saved_round + 1

    data_signature = hashlib.sha256(
        json.dumps(
            {"texts": fed.data.train_texts, "labels": fed.data.train_labels,
             "clients": fed.partition.client_indices},
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    checkpoint_context = {
        "seed": seed,
        "model": OmegaConf.to_container(cfg.model, resolve=True),
        "data": OmegaConf.to_container(cfg.data, resolve=True),
        "clients_per_round": cfg.fl.get("clients_per_round", None),
        "partition_sha256": data_signature,
        "frozen_base_sha256": frozen_base_sha256(model),
    }
    if source_checkpoint:
        source_context = load_global_checkpoint_context(source_checkpoint)
        if source_context != checkpoint_context:
            raise ValueError("checkpoint model, data, seed, or client partition differs from run")

    # -- clients ----------------------------------------------------------
    ccfg = ClientConfig(
        local_epochs=cfg.fl.local_epochs,
        lr=cfg.fl.lr,
        batch_size=cfg.fl.batch_size,
        weight_decay=cfg.fl.get("weight_decay", 0.0),
        local_steps=cfg.fl.get("local_steps", None),
    )
    clients = []
    for cid in range(fed.num_clients):
        if fed.data.modality == "image":
            # Pixels, not prompts. `texts` stays empty and the labels ARE the training
            # target — see LoRAClient.images for why presence of the shard, not a flag,
            # is what selects the loss.
            images, labels = fed.client_images(cid)
            clients.append(LoRAClient(cid, [], labels, attack, ccfg, images=images))
            continue
        texts, labels = fed.client_data(cid)
        # None for classification; for instruction tuning these are the generation
        # targets, indexed by the SAME partition indices as `texts`.
        clients.append(
            LoRAClient(cid, texts, labels, attack, ccfg, responses=fed.client_responses(cid))
        )

    trusted_client = None
    if root_shard is not None:
        # The root trainer deliberately uses the same optimizer schedule as a client, as
        # in FLTrust Algorithm 2, but a no-op attack instance and an isolated RNG stream.
        trusted_client = LoRAClient(
            -1,
            root_shard.texts,
            root_shard.labels,
            build_attack(AttackConfig(name="none")),
            ccfg,
            responses=root_shard.responses,
            images=root_shard.images,
        )

    # -- server components ------------------------------------------------
    aggregation = build_aggregation_from_cfg(cfg.aggregation)
    defense = build_defense_from_cfg(cfg.defense)
    probes = build_probes(list(cfg.probes))
    scfg = ServerConfig(
        num_rounds=cfg.fl.num_rounds,
        start_round=start_round,
        evaluation_split=str(cfg.get("evaluation_split", "test")),
        clients_per_round=cfg.fl.get("clients_per_round", None),
        eval_batch_size=cfg.get("eval_batch_size", 64),
        eval_examples=(int(cfg.eval_examples) if cfg.get("eval_examples", None) else None),
        persist_canonical_frames=bool(cfg.get("persist_canonical_frames", False)),
        persist_client_factors=bool(cfg.get("persist_client_factors", False)),
        persist_client_factors_rounds=cfg.get("persist_client_factors_rounds", None),
        persist_client_factors_at_rounds=tuple(
            int(round_idx) for round_idx in (cfg.get("persist_client_factors_at_rounds", []) or [])
        ),
        persist_global_at_rounds=tuple(
            int(round_idx) for round_idx in (cfg.get("persist_global_at_rounds", []) or [])
        ),
        checkpoint_context=checkpoint_context,
        generation_every=int(cfg.get("generation_every", 5)),
        generation_prompts=int(cfg.get("generation_prompts", 150)),
        generation_max_new_tokens=int(cfg.get("generation_max_new_tokens", 128)),
        bertscore_model=str(cfg.get("bertscore_model", "roberta-large")),
        generation_final_prompts=(
            int(cfg.generation_final_prompts) if cfg.get("generation_final_prompts", None) else None
        ),
        # Resolved from the dataset rather than required in every config, so only a SQL run
        # carries SQL settings — but through data/spider.default_db_root, so the location is
        # defined in exactly one place and honours SPIDER_DATA_DIR.
        db_root=_spider_db_root(cfg),
        sql_timeout_s=float(cfg.get("sql_timeout_s", 10.0)),
        probe_coalition_ids=tuple(
            cfg.get("probe_coalition_ids", None) or attack.config.malicious_client_ids
        ),
    )

    # -- run dir + provenance ---------------------------------------------
    chash = hash_config(cfg)
    run_dir = Path(cfg.get("output_dir", "outputs")) / experiment_name / f"{chash}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    Manifest.build(cfg, seed).write(run_dir)
    (run_dir / "partition.json").write_text(json.dumps(fed.partition.to_dict(), indent=2))
    if root_record is not None:
        (run_dir / "trusted_root.json").write_text(json.dumps(root_record, indent=2))
    logger = RunLogger(run_dir)

    server = FederatedServer(
        model,
        aggregation,
        defense,
        probes,
        scfg,
        device,
        logger,
        trusted_client=trusted_client,
    )
    result = server.run(
        clients, fed.data, attack, seed=seed,
        initial_adapters=initial_adapters, initial_head=initial_head,
    )

    # -- per-run summary for cross-seed analysis --------------------------
    summary = {
        "experiment": experiment_name,
        "seed": seed,
        "config_hash": chash,
        "aggregation": cfg.aggregation.name,
        "attack": cfg.attack.name,
        # sweep-cell coordinates: the analysis layer groups runs by (alpha, lora_r)
        "alpha": float(cfg.fl.alpha),
        "lora_r": int(cfg.model.lora_r),
        "num_clients": int(cfg.fl.num_clients),
        "num_rounds": int(cfg.fl.num_rounds),
        "model": cfg.model.get("model_name", cfg.model.kind),
        "dataset": cfg.data.name,
        "amp": cfg.model.get("amp", None),
        "final_accuracy": result.final_accuracy,
        # Task-appropriate finals. For a causal-LM run `final_accuracy` is a placeholder
        # 0.0 and `final_eval_perplexity` is the number that means something; both are
        # written so a reader never has to know which task produced the file to parse it.
        **{f"final_{k}": v[-1] for k, v in result.metrics.items() if v},
        # Taken from the DATA, not inferred from which metrics happen to be present. The
        # inference ("perplexity present => causal_lm, else seq_cls") had no branch for a
        # third modality, so the vision arm was writing `task: seq_cls` — a run on 224x224
        # pixels labelled as BERT-style text classification, in the one file a reader trusts
        # to say what was run.
        "task": fed.data.task,
        "modality": fed.data.modality,
        "num_classes": fed.data.num_labels,
        # WHICH held-out split every number above was computed on. `test` reports,
        # `validation` tunes, and the two are the same size and shape — so without this
        # recorded, a budget selected on validation is indistinguishable from a result.
        **{f"data_{k}": v for k, v in fed.data.info.items()},
        # None = ran the full schedule; an int = the run went non-finite there and
        # stopped. A DISTINCT outcome from "collapsed", and reported as such.
        "diverged_at_round": result.diverged_at_round,
        "rounds_completed": len(result.rounds),
        "initial_checkpoint": str(source_checkpoint) if source_checkpoint else None,
        "start_round": start_round,
        "evaluation_split": scfg.evaluation_split,
        "final_asr": result.final_asr,
        "persistence": result.persistence.to_dict(),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log.info("run complete -> %s", run_dir)
    return result
