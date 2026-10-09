"""Portable global LoRA checkpoints for paired recovery experiments.

The saved state is the post-aggregation model that the server evaluates and broadcasts,
not a client update or the untruncated aggregate used by geometry probes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from .types import AdapterDict, AdapterState


def frozen_base_sha256(model) -> str:
    """Fingerprint unchanged model parameters, including any frozen original head."""
    digest = hashlib.sha256()
    active_head = set(model.get_head_state())
    for name, parameter in model.model.named_parameters():
        if "lora_" in name or name in active_head:
            continue
        digest.update(name.encode())
        digest.update(str(tuple(parameter.shape)).encode())
        raw = parameter.detach().cpu().contiguous().view(torch.uint8).numpy()
        digest.update(raw.tobytes())
    return digest.hexdigest()


def save_global_checkpoint(
    path: str | Path,
    *,
    round_idx: int,
    adapters: AdapterDict,
    head_state: dict[str, torch.Tensor],
    context: dict | None = None,
) -> Path:
    """Save tensor-only state; run metadata lives in the enclosing run manifest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "round": int(round_idx),
        "adapters": {
            name: (state.A.detach().cpu().contiguous(), state.B.detach().cpu().contiguous())
            for name, state in adapters.items()
        },
        "head_state": {name: tensor.detach().cpu().contiguous() for name, tensor in head_state.items()},
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    if context is not None:
        sidecar = path.with_suffix(".json")
        temporary_json = sidecar.with_suffix(".json.tmp")
        temporary_json.write_text(json.dumps(context, indent=2, sort_keys=True))
        temporary_json.replace(sidecar)
    return path


def load_global_checkpoint(
    path: str | Path,
) -> tuple[int, AdapterDict, dict[str, torch.Tensor]]:
    """Load a checkpoint without unpickling application classes."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError(f"unsupported global checkpoint format in {path}")
    adapters = {
        name: AdapterState(A, B)
        for name, (A, B) in payload["adapters"].items()
    }
    return int(payload["round"]), adapters, payload["head_state"]


def load_global_checkpoint_context(path: str | Path) -> dict:
    sidecar = Path(path).with_suffix(".json")
    return json.loads(sidecar.read_text())


def validate_global_checkpoint(model, adapters: AdapterDict, head_state: dict[str, torch.Tensor]) -> None:
    """Reject incomplete or incompatible states before copying any tensors into a model."""
    expected_adapters = model.get_adapters()
    if adapters.keys() != expected_adapters.keys():
        raise ValueError("checkpoint LoRA layer names do not match model")
    for name, state in adapters.items():
        expected = expected_adapters[name]
        if state.A.shape != expected.A.shape or state.B.shape != expected.B.shape:
            raise ValueError(f"checkpoint LoRA shape mismatch for {name}")
        if not torch.isfinite(state.A).all() or not torch.isfinite(state.B).all():
            raise ValueError(f"checkpoint contains non-finite LoRA factors for {name}")
    expected_head = model.get_head_state()
    if head_state.keys() != expected_head.keys():
        raise ValueError("checkpoint head parameter names do not match model")
    for name, tensor in head_state.items():
        if tensor.shape != expected_head[name].shape or not torch.isfinite(tensor).all():
            raise ValueError(f"checkpoint head mismatch for {name}")
