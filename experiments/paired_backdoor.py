"""Run matched clean and triggered training arms; save a small comparison."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from fedlora_attacks.config_compose import compose_config
from fedlora_attacks.runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. defense=norm_bound")
    args = parser.parse_args()

    attacked = compose_config(["experiment=quickstart", *args.overrides])
    if attacked.attack.name != "x4_backdoor":
        parser.error("paired_backdoor requires the x4_backdoor attack config")
    clean = copy.deepcopy(attacked)
    clean.attack.attack_start_round = int(clean.fl.num_rounds)

    clean_result = run_experiment(clean, experiment_name="paired_backdoor/clean")
    attack_result = run_experiment(attacked, experiment_name="paired_backdoor/attacked")
    clean_dir = Path(clean_result.run_dir)
    attack_dir = Path(attack_result.run_dir)
    clean_partition = json.loads((clean_dir / "partition.json").read_text())
    attack_partition = json.loads((attack_dir / "partition.json").read_text())
    if clean_partition != attack_partition:
        raise RuntimeError("paired arms have different client partitions")

    comparison = {
        "seed": int(attacked.seed),
        "dataset": str(attacked.data.name),
        "defense": str(attacked.defense.name),
        "aggregation": str(attacked.aggregation.name),
        "partition_equal": True,
        "clean_accuracy": clean_result.final_accuracy,
        "attacked_accuracy": attack_result.final_accuracy,
        "accuracy_drop": clean_result.final_accuracy - attack_result.final_accuracy,
        "clean_asr": clean_result.final_asr,
        "attacked_asr": attack_result.final_asr,
        "clean_run": str(clean_dir),
        "attacked_run": str(attack_dir),
    }
    path = Path(attacked.output_dir) / "paired_backdoor" / "comparison.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(comparison, indent=2))
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
