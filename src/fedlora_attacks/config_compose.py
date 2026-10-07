"""Programmatic Hydra composition (used by experiment drivers and tests).

Lets Python code build the same resolved config the CLI would, without spawning a
process, so multi-seed experiment loops and the smoke test can call
``run_experiment`` directly.
"""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs")


def compose_config(overrides: list[str] | None = None, config_name: str = "config") -> DictConfig:
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(config_name=config_name, overrides=overrides or [])
