"""Hydra entry point: ``python -m fedlora_attacks.cli [overrides...]``."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from .runner import run_experiment


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run_experiment(cfg, experiment_name=cfg.get("experiment_name", "run"))


if __name__ == "__main__":
    main()
