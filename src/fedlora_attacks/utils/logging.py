"""Structured JSON run logging (CLAUDE.md: "structured JSON run logs").

A ``RunLogger`` owns a run directory and appends one JSON object per record to
``metrics.jsonl`` (round-level probe metrics + outcome metrics). A thin adapter
seam is left for optional W&B/MLflow backends, but the default is file-only so runs
are hermetic and diff-able.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

_console = logging.getLogger("fedlora")
if not _console.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    _console.addHandler(_h)
    _console.setLevel(logging.INFO)
    # Don't double-emit when Hydra (or another app) also configures the root logger.
    _console.propagate = False


def get_logger() -> logging.Logger:
    return _console


def _jsonable(obj: Any) -> Any:
    """Best-effort conversion of tensors/arrays/dataclasses to JSON-native types."""
    if hasattr(obj, "tolist"):  # torch.Tensor, np.ndarray
        return obj.tolist()
    if hasattr(obj, "item") and getattr(obj, "ndim", None) == 0:
        return obj.item()
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_jsonable(v) for v in obj]
    return obj


class RunLogger:
    """Append-only JSONL logger scoped to a single run directory."""

    def __init__(self, run_dir: str | Path, filename: str = "metrics.jsonl") -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / filename
        # Truncate any stale file from a previous run at the same path.
        self.path.write_text("")

    def log(self, record: dict[str, Any]) -> None:
        """Append one record. Values are coerced to JSON-native types."""
        line = json.dumps(_jsonable(record), default=str)
        with self.path.open("a") as f:
            f.write(line + "\n")

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]
