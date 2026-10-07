"""Deterministic hashing of a resolved config (CLAUDE.md invariant 4).

The config hash goes into ``manifest.json`` and is used to detect whether two runs
share identical (fully-resolved) configuration.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from omegaconf import DictConfig, OmegaConf


def _canonical(obj: Any) -> Any:
    """Recursively sort dict keys so hashing is order-independent."""
    if isinstance(obj, dict):
        return {k: _canonical(obj[k]) for k in sorted(obj)}
    if isinstance(obj, list | tuple):
        return [_canonical(x) for x in obj]
    return obj


def config_to_dict(cfg: DictConfig | dict) -> dict:
    """Fully resolve an OmegaConf config (or pass a plain dict) to a plain dict."""
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    return cfg


def hash_config(cfg: DictConfig | dict) -> str:
    """Return a stable short SHA-256 hex digest (16 chars) of the resolved config."""
    payload = json.dumps(_canonical(config_to_dict(cfg)), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
