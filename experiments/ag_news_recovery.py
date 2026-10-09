"""Train matched AG News arms and select a clean-only recovered checkpoint."""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path

import torch

from fedlora_attacks.config_compose import compose_config
from fedlora_attacks.data.base import build_federated_data, build_tokenizer
from fedlora_attacks.runner import run_experiment
from fedlora_attacks.utils.config_hash import hash_config


def _preflight(cfg) -> dict:
    fed = build_federated_data(
        name=cfg.data.name,
        num_train=int(cfg.data.num_train),
        num_test=int(cfg.data.num_test),
        num_clients=int(cfg.fl.num_clients),
        alpha=float(cfg.fl.alpha),
        seed=int(cfg.seed),
        num_labels=int(cfg.data.num_labels),
        subsample_seed=cfg.data.get("subsample_seed", None),
        partition_scheme=str(cfg.fl.get("partition", "dirichlet")),
        server_clean_size=int(cfg.data.server_clean_size),
        validation_size=int(cfg.data.validation_size),
    )
    tokenizer = build_tokenizer(cfg.model.tokenizer_kind, fed.data, cfg.model.model_name)
    trigger = str(cfg.attack.trigger)
    ids = tokenizer(trigger, add_special_tokens=False)["input_ids"]
    if not ids or getattr(tokenizer, "unk_token_id", None) in ids:
        raise ValueError("trigger tokenizes to an empty or unknown-token sequence")
    pattern = re.compile(rf"(?<!\w){re.escape(trigger)}(?!\w)", re.IGNORECASE)
    clean_texts = (
        fed.data.train_texts
        + (fed.data.server_clean_texts or [])
        + (fed.data.validation_texts or [])
    )
    eligible_by_attacker = {
        str(client_id): sum(
            fed.data.train_labels[i] != int(cfg.attack.target_label)
            for i in fed.partition.client_indices[int(client_id)]
        )
        for client_id in cfg.attack.malicious_client_ids
    }
    if not eligible_by_attacker or not any(eligible_by_attacker.values()):
        raise ValueError("no malicious client has non-target AG News examples to poison")
    return {
        "trigger": trigger,
        "target_label": int(cfg.attack.target_label),
        "malicious_client_ids": list(cfg.attack.malicious_client_ids),
        "poison_fraction_of_non_target_examples": float(cfg.attack.poison_fraction),
        "poison_mode": str(cfg.attack.poison_mode),
        "eligible_non_target_examples_by_attacker": eligible_by_attacker,
        "trigger_token_ids": list(ids),
        "natural_trigger_count": sum(bool(pattern.search(text)) for text in clean_texts),
        "checked_clean_articles": len(clean_texts),
        "train_articles": len(fed.data.train_texts),
        "server_clean_articles": len(fed.data.server_clean_texts or []),
        "validation_articles": len(fed.data.validation_texts or []),
    }


def _partition(run_dir: str) -> dict:
    return json.loads((Path(run_dir) / "partition.json").read_text())


def _attack_effect(clean_validation: dict, poisoned_validation: dict) -> dict:
    """Report the observed attack effect without imposing a dataset-specific cutoff."""
    clean_asr = clean_validation["asr"]
    poisoned_asr = poisoned_validation["asr"]
    if clean_asr is None or poisoned_asr is None:
        raise ValueError("attack comparison requires non-target validation examples")
    asr_gain = float(poisoned_asr) - float(clean_asr)
    accuracy_loss = (
        float(clean_validation["clean_accuracy"])
        - float(poisoned_validation["clean_accuracy"])
    )
    return {
        "asr_gain": asr_gain,
        "clean_accuracy_loss": accuracy_loss,
        "asr_gain_percentage_points": 100 * asr_gain,
        "clean_accuracy_loss_percentage_points": 100 * accuracy_loss,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("overrides", nargs="*", help="Hydra overrides")
    args = parser.parse_args()
    cfg = compose_config(["experiment=ag_news_recovery", *args.overrides])
    if cfg.device == "auto":
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    if cfg.data.name != "ag_news" or cfg.attack.name != "x4_backdoor":
        parser.error("ag_news_recovery requires AG News and x4_backdoor")
    if cfg.attack.poison_mode != "replace":
        parser.error("the controlled AG News study requires replacement poisoning")
    if int(cfg.data.server_clean_size) < 1 or int(cfg.data.validation_size) < 1:
        parser.error("the study needs disjoint server-clean and validation subsets")
    attack_rounds = int(cfg.fl.num_rounds)
    recovery_rounds = int(cfg.recovery.max_rounds)
    if attack_rounds < 1 or recovery_rounds < 1:
        parser.error("attack and recovery rounds must be positive")

    study_dir = Path(cfg.output_dir) / "ag_news_recovery" / f"{hash_config(cfg)}_seed{cfg.seed}"
    study_dir.mkdir(parents=True, exist_ok=True)
    preflight = _preflight(cfg)
    (study_dir / "preflight.json").write_text(json.dumps(preflight, indent=2))

    clean_cfg = copy.deepcopy(cfg)
    clean_cfg.attack.attack_start_round = attack_rounds
    clean_cfg.persist_global_at_rounds = [attack_rounds - 1]
    clean = run_experiment(clean_cfg, experiment_name="ag_news_recovery/clean")

    poisoned_cfg = copy.deepcopy(cfg)
    poisoned_cfg.attack.attack_stop_round = attack_rounds
    poisoned_cfg.persist_global_at_rounds = [attack_rounds - 1]
    poisoned = run_experiment(poisoned_cfg, experiment_name="ag_news_recovery/poisoned")
    if _partition(clean.run_dir) != _partition(poisoned.run_dir):
        raise RuntimeError("clean and poisoned runs have different client partitions")
    if len(clean.per_round) != attack_rounds or len(poisoned.per_round) != attack_rounds:
        raise RuntimeError("clean or poisoned training ended before the final attack round")
    if [record["round"] for record in clean.per_round] != [
        record["round"] for record in poisoned.per_round
    ]:
        raise RuntimeError("clean and poisoned validation rounds are not aligned")

    poisoned_checkpoint = (
        Path(poisoned.run_dir) / "checkpoints" / f"global_round_{attack_rounds - 1:04d}.pt"
    )
    clean_checkpoint = (
        Path(clean.run_dir) / "checkpoints" / f"global_round_{attack_rounds - 1:04d}.pt"
    )
    clean_val = clean.per_round[-1]["validation"]
    poisoned_val = poisoned.per_round[-1]["validation"]
    attack_effect = _attack_effect(clean_val, poisoned_val)
    attack_trajectory = [
        {
            "round": int(clean_record["round"]),
            "clean_validation": clean_record["validation"],
            "poisoned_validation": poisoned_record["validation"],
            "effect": _attack_effect(
                clean_record["validation"], poisoned_record["validation"]
            ),
        }
        for clean_record, poisoned_record in zip(clean.per_round, poisoned.per_round, strict=True)
    ]
    recovery_cfg = copy.deepcopy(cfg)
    recovery_cfg.fl.num_rounds = recovery_rounds
    recovery_cfg.attack.attack_stop_round = attack_rounds
    recovery_cfg.initial_checkpoint = str(poisoned_checkpoint.resolve())
    recovery_cfg.persist_global_at_rounds = list(range(attack_rounds, attack_rounds + recovery_rounds))
    recovered = run_experiment(recovery_cfg, experiment_name="ag_news_recovery/recovery")
    if _partition(poisoned.run_dir) != _partition(recovered.run_dir):
        raise RuntimeError("recovery changed the client partition")

    max_asr = float(clean_val["asr"]) + float(cfg.recovery.asr_tolerance)
    min_acc = float(clean_val["clean_accuracy"]) - float(cfg.recovery.accuracy_tolerance)
    selected = next(
        (
            record for record in recovered.per_round
            if record["validation"]["asr"] <= max_asr
            and record["validation"]["clean_accuracy"] >= min_acc
        ),
        None,
    )
    selected_round = None if selected is None else int(selected["round"])
    selected_checkpoint = (
        None if selected_round is None else
        str(Path(recovered.run_dir) / "checkpoints" / f"global_round_{selected_round:04d}.pt")
    )
    report = {
        "status": "recovered" if selected is not None else "no_eligible_recovery",
        "attack_effect": attack_effect,
        "attack_trajectory": attack_trajectory,
        "selection_rule": {
            "maximum_validation_asr": max_asr,
            "minimum_validation_accuracy": min_acc,
            "first_qualifying_round": True,
        },
        "clean_validation": clean_val,
        "poisoned_validation": poisoned_val,
        "recovery_validation": [record["validation"] for record in recovered.per_round],
        "recovered_validation": None if selected is None else selected["validation"],
        "clean_run": clean.run_dir,
        "poisoned_run": poisoned.run_dir,
        "recovery_run": recovered.run_dir,
        "poisoned_checkpoint": str(poisoned_checkpoint),
        "clean_checkpoint": str(clean_checkpoint),
        "recovered_checkpoint": selected_checkpoint,
        "recovered_round": selected_round,
        "preflight": preflight,
    }
    path = study_dir / "pair.json"
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps({"report": str(path), **report}, indent=2))


if __name__ == "__main__":
    main()
