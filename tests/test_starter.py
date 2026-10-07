"""The quickstart's clean/attacked pair uses the real FL loop."""

import copy
import json
from pathlib import Path

import pytest

from fedlora_attacks.config_compose import compose_config
from fedlora_attacks.runner import (
    build_aggregation_from_cfg,
    build_defense_from_cfg,
    run_experiment,
)


@pytest.mark.parametrize(
    "name",
    [
        "none", "norm_bound", "multi_krum", "aggregate_norm_check", "cosdefense",
        "spectral_check", "fltrust", "factor_broadcast_cosine", "factor_broadcast_topk",
    ],
)
def test_defense_configs_build(name):
    cfg = compose_config([f"defense={name}"])
    assert build_defense_from_cfg(cfg.defense) is not None


@pytest.mark.parametrize(
    "name",
    ["fedit_naive", "exact_deltaw", "canonicalized_fedit", "consensus_frame",
     "consensus_frame_magnitude"],
)
def test_aggregation_configs_build(name):
    cfg = compose_config([f"aggregation={name}"])
    assert build_aggregation_from_cfg(cfg.aggregation) is not None


@pytest.mark.smoke
def test_paired_backdoor_uses_same_partition_and_records_asr(tmp_path):
    attacked = compose_config(
        [
            "experiment=quickstart",
            f"output_dir={tmp_path}",
            "fl.num_rounds=1",
            "fl.local_steps=1",
            "data.num_train=128",
            "data.num_test=32",
        ]
    )
    clean = copy.deepcopy(attacked)
    clean.attack.attack_start_round = 1

    clean_result = run_experiment(clean, experiment_name="test_pair/clean")
    attack_result = run_experiment(attacked, experiment_name="test_pair/attacked")

    clean_partition = json.loads((Path(clean_result.run_dir) / "partition.json").read_text())
    attack_partition = json.loads((Path(attack_result.run_dir) / "partition.json").read_text())
    assert clean_partition == attack_partition
    assert clean_result.final_asr is not None
    assert attack_result.final_asr is not None
    assert len(clean_result.rounds) == len(attack_result.rounds) == 1
