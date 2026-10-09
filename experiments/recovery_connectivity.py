"""Search clean-loss paths for one AG News poisoned/recovered checkpoint pair."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from fedlora_attacks.data.base import build_federated_data, build_tokenizer
from fedlora_attacks.fl.checkpoint import (
    frozen_base_sha256,
    load_global_checkpoint,
    load_global_checkpoint_context,
)
from fedlora_attacks.models.lora_wrapper import build_lora_model
from fedlora_attacks.metrics.evaluation import backdoor_classification_metrics
from fedlora_attacks.paths.connectivity import (
    ConnectivityPath,
    EmbeddedPath,
    evaluate_path,
    optimize_path,
)
from fedlora_attacks.paths.h0 import sampled_merge_events
from fedlora_attacks.runner import _model_spec, build_attack_from_cfg
from fedlora_attacks.utils.seeding import seed_everything


def _build_model(cfg, tokenizer, num_labels: int, rank: int):
    seed_everything(int(cfg.seed))
    spec = replace(_model_spec(cfg.model, cfg.model.tokenizer_kind), lora_r=rank)
    return build_lora_model(
        spec, tokenizer, num_labels, max_length=int(cfg.model.get("max_length", 32))
    ).to(cfg.device)


def _dense_curve(path, texts, labels, attack, grid: int, batch_size: int) -> list[dict]:
    if grid < 3:
        raise ValueError("path evaluation grid needs at least three points")
    positions = np.linspace(0.0, 1.0, grid).tolist()
    curve = evaluate_path(path, texts, labels, attack, positions, batch_size=batch_size)
    # Refine around the observed peak to reduce the chance that a narrow barrier is missed.
    for _ in range(2):
        peak = max(range(len(curve)), key=lambda i: curve[i]["clean_loss"])
        neighbors = []
        if peak > 0:
            neighbors.append((curve[peak - 1]["t"] + curve[peak]["t"]) / 2)
        if peak + 1 < len(curve):
            neighbors.append((curve[peak]["t"] + curve[peak + 1]["t"]) / 2)
        extra = evaluate_path(path, texts, labels, attack, neighbors, batch_size=batch_size)
        curve = sorted([*curve, *extra], key=lambda record: record["t"])
    return curve


def _summary(curve: list[dict]) -> dict:
    peak = max(curve, key=lambda record: record["clean_loss"])
    end_loss = max(curve[0]["clean_loss"], curve[-1]["clean_loss"])
    return {
        "estimated_merge_loss": peak["clean_loss"],
        "estimated_barrier": max(0.0, peak["clean_loss"] - end_loss),
        "peak_t": peak["t"],
        "start_loss": curve[0]["clean_loss"],
        "end_loss": curve[-1]["clean_loss"],
        "sampled_positions": len(curve),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pair", type=Path, help="pair.json from ag_news_recovery.py")
    args = parser.parse_args()
    pair = json.loads(args.pair.read_text())
    if pair["status"] != "recovered" or not pair["recovered_checkpoint"]:
        parser.error("pair has no validation-qualified recovered checkpoint")
    poisoned_checkpoint = Path(pair["poisoned_checkpoint"])
    recovered_checkpoint = Path(pair["recovered_checkpoint"])
    poisoned_manifest = json.loads((Path(pair["poisoned_run"]) / "manifest.json").read_text())
    cfg = OmegaConf.create(poisoned_manifest["resolved_config"])
    if cfg.data.name != "ag_news" or cfg.attack.name != "x4_backdoor":
        parser.error("connectivity currently supports the AG News text-backdoor study")
    if str(cfg.device).startswith("cuda") and not torch.cuda.is_available():
        parser.error("checkpoint run used CUDA but no CUDA device is available now")
    threads = cfg.get("torch_num_threads", None)
    if threads:
        torch.set_num_threads(int(threads))
    fed = build_federated_data(
        name=cfg.data.name,
        num_train=cfg.data.num_train,
        num_test=cfg.data.num_test,
        num_clients=cfg.fl.num_clients,
        alpha=float(cfg.fl.alpha),
        seed=int(cfg.seed),
        num_labels=int(cfg.data.num_labels),
        subsample_seed=cfg.data.get("subsample_seed", None),
        partition_scheme=str(cfg.fl.get("partition", "dirichlet")),
        server_clean_size=int(cfg.data.server_clean_size),
        validation_size=int(cfg.data.validation_size),
    )
    tokenizer = build_tokenizer(cfg.model.tokenizer_kind, fed.data, cfg.model.model_name)
    attack = build_attack_from_cfg(cfg.attack)
    _, poisoned, poisoned_head = load_global_checkpoint(poisoned_checkpoint)
    _, recovered, recovered_head = load_global_checkpoint(recovered_checkpoint)
    checkpoint_context = load_global_checkpoint_context(poisoned_checkpoint)
    if checkpoint_context != load_global_checkpoint_context(recovered_checkpoint):
        raise ValueError("poisoned and recovered checkpoints have different static contexts")
    source_rank = int(cfg.model.lora_r)
    fit_texts = fed.data.server_clean_texts
    fit_labels = fed.data.server_clean_labels
    val_texts = fed.data.validation_texts
    val_labels = fed.data.validation_labels
    if not fit_texts or not val_texts:
        parser.error("pair run did not reserve server-clean and validation articles")
    search = cfg.path_search
    output = args.pair.parent / "connectivity"
    output.mkdir(parents=True, exist_ok=True)
    candidates: list[dict] = []
    batch_size = int(cfg.eval_batch_size)

    def assess(path, name: str, group: str, *, optimize: bool = False, seed: int = 0):
        history = []
        if optimize:
            history = optimize_path(
                path, fit_texts, fit_labels,
                steps=int(search.steps), batch_size=int(search.batch_size),
                lr=float(search.factor_lr), base_lr=float(search.base_lr), seed=seed,
            )
        curve = _dense_curve(
            path, val_texts, val_labels, attack, int(search.validation_grid), batch_size
        )
        path_file = output / f"{name}.pt"
        if isinstance(path, ConnectivityPath):
            path.save(path_file)
        else:
            path_file = None
        record = {
            "name": name,
            "group": group,
            "path_file": str(path_file) if path_file else None,
            "optimization_last_loss": history[-1] if history else None,
            "validation": _summary(curve),
            "validation_curve": curve,
        }
        candidates.append(record)
        return record

    model_r = _build_model(cfg, tokenizer, fed.data.num_labels, source_rank)
    if frozen_base_sha256(model_r) != checkpoint_context["frozen_base_sha256"]:
        raise ValueError("rebuilt rank-r model has different frozen base weights")
    rank_r_paths = []
    for mode, name in (("rank_r_zero", "rank_r_via_zero"), ("rank_r", "rank_r_factor_linear")):
        path = ConnectivityPath(model_r, poisoned, recovered, poisoned_head, recovered_head, mode=mode)
        record = assess(path, name, "rank_r")
        if name == "rank_r_via_zero":
            expected = (
                pair["poisoned_validation"],
                pair.get("recovered_validation") or pair["recovery_validation"][
                    int(pair["recovered_round"]) - int(cfg.fl.num_rounds)
                ],
            )
            observed = (record["validation_curve"][0], record["validation_curve"][-1])
            for actual, saved in zip(observed, expected, strict=True):
                for key in ("clean_loss", "clean_accuracy", "asr"):
                    if abs(float(actual[key]) - float(saved[key])) > 1e-4:
                        raise ValueError(f"path endpoint {key} differs from checkpoint evaluation")
        rank_r_paths.append((path, record))
    for restart in range(int(search.restarts)):
        name = f"rank_r_curve_{restart}"
        path = ConnectivityPath(
            model_r, poisoned, recovered, poisoned_head, recovered_head,
            mode="rank_r", jitter=0.0 if restart == 0 else 0.05, seed=restart,
        )
        record = assess(path, name, "rank_r", optimize=True, seed=int(cfg.seed) + restart)
        rank_r_paths.append((path, record))
    best_r = min(rank_r_paths, key=lambda item: item[1]["validation"]["estimated_merge_loss"])

    model_2r = _build_model(cfg, tokenizer, fed.data.num_labels, 2 * source_rank)
    if frozen_base_sha256(model_2r) != checkpoint_context["frozen_base_sha256"]:
        raise ValueError("rebuilt expanded-rank model has different frozen base weights")
    embedded = EmbeddedPath(best_r[0], model_2r)
    model_r.model.eval()
    model_2r.model.eval()
    for endpoint_t in (0.0, 1.0):
        with torch.no_grad():
            source_logits = best_r[0].logits(val_texts[:8], endpoint_t)
            embedded_logits = embedded.logits(val_texts[:8], endpoint_t)
            torch.testing.assert_close(source_logits, embedded_logits, atol=1e-4, rtol=1e-4)
    embedded_record = assess(embedded, "rank_r_best_embedded", "rank_2r")
    rank_2r_paths = [(embedded, embedded_record)]
    direct = ConnectivityPath(
        model_2r, poisoned, recovered, poisoned_head, recovered_head, mode="rank_2r_linear"
    )
    direct_record = assess(direct, "rank_2r_effective_line", "rank_2r")
    rank_2r_paths.append((direct, direct_record))
    for restart in range(int(search.restarts)):
        name = f"rank_2r_curve_{restart}"
        path = ConnectivityPath(
            model_2r, poisoned, recovered, poisoned_head, recovered_head,
            mode="rank_2r_curve", jitter=0.0 if restart == 0 else 0.05, seed=restart,
        )
        record = assess(path, name, "rank_2r", optimize=True, seed=int(cfg.seed) + 100 + restart)
        rank_2r_paths.append((path, record))
    best_2r = min(rank_2r_paths, key=lambda item: item[1]["validation"]["estimated_merge_loss"])

    full_paths = [best_2r]
    for restart in range(int(search.restarts)):
        name = f"full_curve_{restart}"
        path = ConnectivityPath(
            model_2r, poisoned, recovered, poisoned_head, recovered_head,
            mode="full_curve", jitter=0.0 if restart == 0 else 0.05, seed=restart,
        )
        record = assess(path, name, "full", optimize=True, seed=int(cfg.seed) + 200 + restart)
        full_paths.append((path, record))
    best_full = min(full_paths, key=lambda item: item[1]["validation"]["estimated_merge_loss"])

    selected = {}
    for group, (path, record) in (
        ("rank_r", best_r), ("rank_2r", best_2r), ("full", best_full)
    ):
        curve = _dense_curve(
            path, fed.data.test_texts, fed.data.test_labels, attack,
            int(search.test_grid), batch_size,
        )
        selected[group] = {
            "candidate": record["name"],
            "validation": record["validation"],
            "test": _summary(curve),
            "test_curve": curve,
        }
    _, clean_adapters, clean_head = load_global_checkpoint(pair["clean_checkpoint"])
    model_r.set_adapters(clean_adapters)
    model_r.set_head_state(clean_head)
    clean_control_test = backdoor_classification_metrics(
        model_r, fed.data.test_texts, fed.data.test_labels, attack,
        cfg.device, batch_size,
    )
    h0 = {}
    for group, allowed in (
        ("rank_r", {"rank_r"}),
        ("rank_2r", {"rank_r", "rank_2r"}),
        ("full", {"rank_r", "rank_2r", "full"}),
    ):
        relevant = [record for record in candidates if record["group"] in allowed]
        endpoint = relevant[0]["validation"]
        h0[group] = sampled_merge_events(
            {"poisoned": endpoint["start_loss"], "recovered": endpoint["end_loss"]},
            [
                {"u": "poisoned", "v": "recovered", "name": record["name"],
                 "maximum_loss": record["validation"]["estimated_merge_loss"]}
                for record in relevant
            ],
        )
    result = {
        "pair": str(args.pair),
        "attack_effect_validation": pair.get("attack_effect"),
        "source_rank": source_rank,
        "head_path": "linear",
        "fit_split": "server_clean",
        "selection_split": "validation",
        "report_split": "test",
        "candidates": candidates,
        "selected": selected,
        "clean_control_test": clean_control_test,
        "sampled_h0_validation": h0,
        "interpretation": (
            "sampled path maxima; failed searches do not establish disconnection; "
            "interpret P as a backdoor endpoint using its observed attack effect"
        ),
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(result, indent=2))
    print(json.dumps({
        "report": str(report_path),
        "selected": {
            group: {"candidate": item["candidate"], "test": item["test"]}
            for group, item in selected.items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
