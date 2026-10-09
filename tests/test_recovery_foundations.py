"""Checks for the AG News recovery study's checkpoint and attack contracts."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from fedlora_attacks.attacks.x4_backdoor import X4BackdoorAttack, X4Config
from fedlora_attacks.config_compose import compose_config
from fedlora_attacks.data import base as data_base
from fedlora_attacks.fl.checkpoint import load_global_checkpoint
from fedlora_attacks.runner import run_experiment
from experiments.ag_news_recovery import _attack_effect
from experiments import ag_news_recovery as study


def test_replacement_poisoning_preserves_shard_size_and_only_changes_non_target():
    attack = X4BackdoorAttack(
        X4Config(
            malicious_client_ids=(0,), trigger="cf", target_label=0,
            poison_fraction=0.5, poison_mode="replace",
        )
    )
    texts = [f"article {i}" for i in range(8)]
    labels = [0, 1, 2, 3, 0, 1, 2, 3]
    poison_rng = np.random.default_rng(4)
    clean_rng = np.random.default_rng(4)
    poisoned_texts, poisoned_labels = attack.poison_dataset(
        texts, labels, 0, 0, poison_rng
    )
    assert len(poisoned_texts) == len(texts)
    assert len(poisoned_labels) == len(labels)
    changed = [i for i, text in enumerate(poisoned_texts) if text != texts[i]]
    assert len(changed) == 3  # half of the six non-target examples
    assert all(labels[i] != 0 for i in changed)
    assert all(poisoned_labels[i] == 0 and poisoned_texts[i] == f"cf {texts[i]}" for i in changed)
    assert all(poisoned_labels[i] == labels[i] for i in range(8) if i not in changed)
    assert poison_rng.integers(2**31) == clean_rng.integers(2**31)


def test_ag_news_reserved_splits_are_disjoint_and_leave_client_budget(monkeypatch):
    def fake_loader(num_train, num_test, seed):
        assert num_train == 20
        texts = [f"unique article {i}" for i in range(num_train)]
        labels = [i % 4 for i in range(num_train)]
        return texts, labels, ["test article"], [1]

    monkeypatch.setattr(data_base, "load_ag_news", fake_loader)
    task = data_base.build_dataset(
        "ag_news", 12, 1, 3, server_clean_size=4, validation_size=4
    )
    assert len(task.train_texts) == 12
    assert len(task.server_clean_texts) == len(task.validation_texts) == 4
    assert len(set(task.train_texts + task.server_clean_texts + task.validation_texts)) == 20


def test_attack_effect_reports_continuous_asr_and_accuracy_changes():
    clean = {"asr": 0.10, "clean_accuracy": 0.88}
    poisoned = {"asr": 0.62, "clean_accuracy": 0.85}
    effect = _attack_effect(clean, poisoned)
    assert effect["asr_gain_percentage_points"] == 52.0
    assert abs(effect["clean_accuracy_loss_percentage_points"] - 3.0) < 1e-10


def test_recovery_runs_without_an_arbitrary_attack_gate(tmp_path: Path, monkeypatch):
    calls = []

    def fake_run(cfg, experiment_name):
        calls.append(experiment_name)
        run_dir = tmp_path / experiment_name.replace("/", "_")
        run_dir.mkdir()
        (run_dir / "partition.json").write_text(json.dumps({"client_indices": [[0], [1]]}))
        if experiment_name.endswith("clean"):
            validation = {"asr": 0.1, "clean_accuracy": 0.8}
        elif experiment_name.endswith("poisoned"):
            validation = {"asr": 0.2, "clean_accuracy": 0.79}
        else:
            validation = {"asr": 0.1, "clean_accuracy": 0.79}
        return SimpleNamespace(
            run_dir=str(run_dir),
            per_round=[
                {"round": i, "validation": validation}
                for i in (range(8, 28) if experiment_name.endswith("recovery") else range(8))
            ],
        )

    monkeypatch.setattr(study, "_preflight", lambda cfg: {})
    monkeypatch.setattr(study, "run_experiment", fake_run)
    monkeypatch.setattr(sys, "argv", [
        "ag_news_recovery.py", f"output_dir={tmp_path}",
    ])
    study.main()
    pair_file = next(tmp_path.rglob("pair.json"))
    report = json.loads(pair_file.read_text())
    assert report["status"] == "recovered"
    assert report["recovered_round"] == 8
    assert abs(report["attack_effect"]["asr_gain_percentage_points"] - 10.0) < 1e-10
    assert len(report["attack_trajectory"]) == 8
    assert report["attack_trajectory"][-1]["round"] == 7
    assert calls == [
        "ag_news_recovery/clean", "ag_news_recovery/poisoned",
        "ag_news_recovery/recovery",
    ]


def test_global_checkpoint_resume_matches_uninterrupted_training(tmp_path: Path):
    base = compose_config(
        [
            "experiment=quickstart", f"output_dir={tmp_path}",
            "fl.local_steps=1", "data.num_train=128", "data.num_test=32",
            "attack.attack_start_round=9",
        ]
    )
    uninterrupted = copy.deepcopy(base)
    uninterrupted.fl.num_rounds = 2
    uninterrupted.persist_global_at_rounds = [1]
    full = run_experiment(uninterrupted, experiment_name="recovery_test/full")

    first_round = copy.deepcopy(base)
    first_round.fl.num_rounds = 1
    first_round.persist_global_at_rounds = [0]
    partial = run_experiment(first_round, experiment_name="recovery_test/partial")
    source = Path(partial.run_dir) / "checkpoints" / "global_round_0000.pt"

    continuation = copy.deepcopy(base)
    continuation.fl.num_rounds = 1
    continuation.initial_checkpoint = str(source)
    continuation.persist_global_at_rounds = [1]
    resumed = run_experiment(continuation, experiment_name="recovery_test/resumed")

    _, full_adapters, full_head = load_global_checkpoint(
        Path(full.run_dir) / "checkpoints" / "global_round_0001.pt"
    )
    resumed_round, resumed_adapters, resumed_head = load_global_checkpoint(
        Path(resumed.run_dir) / "checkpoints" / "global_round_0001.pt"
    )
    assert resumed_round == 1
    for name in full_adapters:
        torch.testing.assert_close(full_adapters[name].delta_w(), resumed_adapters[name].delta_w())
    for name in full_head:
        torch.testing.assert_close(full_head[name], resumed_head[name])
