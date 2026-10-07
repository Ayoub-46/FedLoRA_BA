"""Run provenance: ``manifest.json`` (CLAUDE.md invariant 4).

Every run writes a manifest capturing everything needed to reproduce it: git
commit SHA (+ dirty flag), config hash, seed, fully-resolved config, package
versions, timestamp, and hardware. Git absence degrades gracefully to
``"uninitialized"`` rather than crashing.
"""

from __future__ import annotations

import importlib.metadata as ilmd
import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config_hash import config_to_dict, hash_config

# Packages whose exact versions we record for reproducibility.
_TRACKED_PACKAGES = (
    "torch",
    "transformers",
    "peft",
    "datasets",
    "accelerate",
    "numpy",
    "scipy",
    "hydra-core",
    "omegaconf",
)


def _git_sha() -> str:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        return sha or "uninitialized"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "uninitialized"


def _git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
        )
        return bool(out.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for pkg in _TRACKED_PACKAGES:
        try:
            versions[pkg] = ilmd.version(pkg)
        except ilmd.PackageNotFoundError:
            versions[pkg] = "not-installed"
    return versions


def _hardware() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
    }
    try:
        import torch

        info["torch_cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_devices"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
    except Exception:  # pragma: no cover - torch always present in this project
        info["torch_cuda_available"] = None
    return info


@dataclass
class Manifest:
    """Serializable run provenance record."""

    seed: int
    config_hash: str
    git_sha: str
    git_dirty: bool
    timestamp_utc: str
    package_versions: dict[str, str]
    hardware: dict[str, Any]
    resolved_config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def build(cls, cfg: Any, seed: int) -> Manifest:
        return cls(
            seed=seed,
            config_hash=hash_config(cfg),
            git_sha=_git_sha(),
            git_dirty=_git_dirty(),
            timestamp_utc=datetime.now(UTC).isoformat(),
            package_versions=_package_versions(),
            hardware=_hardware(),
            resolved_config=config_to_dict(cfg),
        )

    def write(self, run_dir: str | Path) -> Path:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "manifest.json"
        path.write_text(json.dumps(asdict(self), indent=2, default=str))
        return path
