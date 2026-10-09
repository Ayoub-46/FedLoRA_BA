"""Construct and search paths in effective LoRA update space.

Factor paths stay within their declared rank. Expanded paths use two factor blocks to
represent the exact straight line between rank-r effective updates. Full paths may bend
every frozen base parameter while keeping the classifier head on the same linear path.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call

from ..fl.aggregation.exact_deltaw import rank_r_from_factors
from ..fl.types import AdapterDict, AdapterState
from ..metrics.evaluation import backdoor_classification_metrics


def _canonical(state: AdapterState) -> AdapterState:
    return rank_r_from_factors(state.A, state.B, state.rank)


def _on_device(adapters: AdapterDict, device: torch.device) -> AdapterDict:
    return {
        name: AdapterState(state.A.to(device), state.B.to(device))
        for name, state in adapters.items()
    }


class ConnectivityPath:
    """One continuous path, with optional trainable control factors/base offsets."""

    def __init__(
        self,
        model,
        poisoned: AdapterDict,
        recovered: AdapterDict,
        poisoned_head: dict[str, torch.Tensor],
        recovered_head: dict[str, torch.Tensor],
        *,
        mode: str,
        jitter: float = 0.0,
        seed: int = 0,
    ) -> None:
        if mode not in ("rank_r", "rank_r_zero", "rank_2r_linear", "rank_2r_curve", "full_curve"):
            raise ValueError(f"unknown path mode {mode!r}")
        self.model = model
        self.mode = mode
        self.device = next(model.model.parameters()).device
        self.poisoned = _on_device(poisoned, self.device)
        self.recovered = _on_device(recovered, self.device)
        if self.poisoned.keys() != self.recovered.keys():
            raise ValueError("path endpoint layer sets differ")
        if poisoned_head.keys() != recovered_head.keys():
            raise ValueError("path endpoint head sets differ")
        self.poisoned_head = {name: value.to(self.device) for name, value in poisoned_head.items()}
        self.recovered_head = {name: value.to(self.device) for name, value in recovered_head.items()}
        self.start: AdapterDict = {}
        self.end: AdapterDict = {}
        self.control: dict[str, tuple[torch.nn.Parameter, torch.nn.Parameter]] = {}
        generator = torch.Generator(device="cpu").manual_seed(seed)
        for name in self.poisoned:
            p = _canonical(self.poisoned[name])
            r = _canonical(self.recovered[name])
            if mode in ("rank_r", "rank_r_zero"):
                if p.rank != r.rank:
                    raise ValueError("fixed-rank path endpoints differ in factor width")
                # The SVD fixes general GL(r) gauge freedom; orthogonal alignment handles
                # remaining sign/rotation ambiguity in nearly repeated singular values.
                overlap = r.A.T @ p.A + r.B @ p.B.T
                u, _, vh = torch.linalg.svd(overlap)
                q = u @ vh
                start = p
                end = AdapterState(r.A @ q, q.T @ r.B)
            else:
                if p.rank != r.rank:
                    raise ValueError("expanded path endpoints need the same source rank")
                zeros_a = torch.zeros_like(p.A)
                shared_b = torch.cat((p.B, r.B), dim=0)
                start = AdapterState(torch.cat((p.A, zeros_a), dim=1), shared_b)
                end = AdapterState(torch.cat((zeros_a, r.A), dim=1), shared_b)
            self.start[name] = start
            self.end[name] = end
            if mode not in ("rank_r_zero", "rank_2r_linear"):
                mid_a = (start.A + end.A) / 2
                mid_b = (start.B + end.B) / 2
                if jitter:
                    noise_a = torch.randn(mid_a.shape, generator=generator).to(self.device)
                    noise_b = torch.randn(mid_b.shape, generator=generator).to(self.device)
                    mid_a = mid_a + jitter * max(float(mid_a.std()), 1e-8) * noise_a
                    mid_b = mid_b + jitter * max(float(mid_b.std()), 1e-8) * noise_b
                self.control[name] = (
                    torch.nn.Parameter(mid_a.detach().clone()),
                    torch.nn.Parameter(mid_b.detach().clone()),
                )
        actual = model.get_adapters()
        for name, state in self.start.items():
            if actual[name].rank != state.rank:
                raise ValueError(f"model rank does not match path rank at {name}")

        self.base_control: dict[str, torch.nn.Parameter] = {}
        if mode == "full_curve":
            head_names = set(self.poisoned_head)
            for name, parameter in model.model.named_parameters():
                if "lora_" not in name and name not in head_names:
                    self.base_control[name] = torch.nn.Parameter(torch.zeros_like(parameter))

    def adapters_at(self, t: float | torch.Tensor) -> AdapterDict:
        t = torch.as_tensor(t, device=self.device)
        if not 0 <= float(t.detach()) <= 1:
            raise ValueError("path position must lie in [0, 1]")
        result: AdapterDict = {}
        for name, start in self.start.items():
            end = self.end[name]
            if self.mode == "rank_r_zero":
                if float(t.detach()) <= 0.5:
                    A, B = (1 - 2 * t) * start.A, start.B
                else:
                    A, B = (2 * t - 1) * end.A, end.B
            elif self.mode == "rank_2r_linear":
                A = (1 - t) * start.A + t * end.A
                B = start.B
            else:
                middle_a, middle_b = self.control[name]
                A = (1 - t).square() * start.A + 2 * t * (1 - t) * middle_a + t.square() * end.A
                B = (1 - t).square() * start.B + 2 * t * (1 - t) * middle_b + t.square() * end.B
            result[name] = AdapterState(A, B)
        return result

    def head_at(self, t: float) -> dict[str, torch.Tensor]:
        return {
            name: (1 - t) * self.poisoned_head[name] + t * self.recovered_head[name]
            for name in self.poisoned_head
        }

    def overrides(self, t: float) -> dict[str, torch.Tensor]:
        out = dict(self.head_at(t))
        for name, state in self.adapters_at(t).items():
            layer = self.model._lora_layers()[name]
            scaling = float(layer.scaling[self.model.adapter_name])
            suffix = self.model.adapter_name
            out[f"{name}.lora_A.{suffix}.weight"] = state.B
            out[f"{name}.lora_B.{suffix}.weight"] = state.A / scaling
        if self.base_control:
            parameters = dict(self.model.model.named_parameters())
            bend = 2 * t * (1 - t)
            out.update(
                {name: parameters[name] + bend * offset for name, offset in self.base_control.items()}
            )
        return out

    def logits(self, texts: list[str], t: float) -> torch.Tensor:
        batch = self.model.encode(texts, self.device)
        result = functional_call(self.model.model, self.overrides(t), (), batch, strict=False)
        return result.logits

    def parameters_to_optimize(self) -> list[torch.nn.Parameter]:
        return [parameter for pair in self.control.values() for parameter in pair] + list(
            self.base_control.values()
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "mode": self.mode,
                "control": {
                    name: (A.detach().cpu(), B.detach().cpu())
                    for name, (A, B) in self.control.items()
                },
                "base_control": {
                    name: value.detach().cpu() for name, value in self.base_control.items()
                },
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        model,
        poisoned: AdapterDict,
        recovered: AdapterDict,
        poisoned_head: dict[str, torch.Tensor],
        recovered_head: dict[str, torch.Tensor],
    ) -> ConnectivityPath:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        result = cls(
            model, poisoned, recovered, poisoned_head, recovered_head,
            mode=payload["mode"],
        )
        with torch.no_grad():
            if result.control.keys() != payload["control"].keys():
                raise ValueError("saved path control layers differ from model")
            for name, (A, B) in payload["control"].items():
                result.control[name][0].copy_(A.to(result.device))
                result.control[name][1].copy_(B.to(result.device))
            if result.base_control.keys() != payload["base_control"].keys():
                raise ValueError("saved path base controls differ from model")
            for name, value in payload["base_control"].items():
                result.base_control[name].copy_(value.to(result.device))
        return result


class _PathModelView:
    def __init__(self, path: ConnectivityPath, t: float) -> None:
        self.path = path
        self.t = t
        self.model = path.model.model

    def logits(self, texts: list[str], device="cpu") -> torch.Tensor:
        return self.path.logits(texts, self.t)


class EmbeddedPath:
    """Evaluate a lower-rank path in a higher-rank model without changing its dW."""

    def __init__(self, source: ConnectivityPath, model) -> None:
        self.source = source
        self.model = model
        self.device = next(model.model.parameters()).device
        expected = model.get_adapters()
        self.ranks = {name: state.rank for name, state in expected.items()}
        for name, low in source.start.items():
            if expected[name].rank < low.rank:
                raise ValueError("target model has lower rank than source path")

    def adapters_at(self, t: float) -> AdapterDict:
        out = {}
        for name, low in self.source.adapters_at(t).items():
            pad = self.ranks[name] - low.rank
            out[name] = AdapterState(
                F.pad(low.A.to(self.device), (0, pad)),
                F.pad(low.B.to(self.device), (0, 0, 0, pad)),
            )
        return out

    def logits(self, texts: list[str], t: float) -> torch.Tensor:
        overrides = {
            name: tensor.to(self.device)
            for name, tensor in self.source.head_at(t).items()
        }
        for name, state in self.adapters_at(t).items():
            layer = self.model._lora_layers()[name]
            scaling = float(layer.scaling[self.model.adapter_name])
            suffix = self.model.adapter_name
            overrides[f"{name}.lora_A.{suffix}.weight"] = state.B
            overrides[f"{name}.lora_B.{suffix}.weight"] = state.A / scaling
        batch = self.model.encode(texts, self.device)
        return functional_call(self.model.model, overrides, (), batch, strict=False).logits


def evaluate_path(
    path: ConnectivityPath,
    texts: list[str],
    labels: list[int],
    attack,
    positions: list[float],
    *,
    batch_size: int = 64,
) -> list[dict]:
    path.model.model.eval()
    records = []
    for t in positions:
        metrics = backdoor_classification_metrics(
            _PathModelView(path, t), texts, labels, attack,
            path.device, batch_size,
        )
        records.append({"t": float(t), **metrics})
    return records


def optimize_path(
    path: ConnectivityPath,
    texts: list[str],
    labels: list[int],
    *,
    steps: int,
    batch_size: int,
    lr: float,
    base_lr: float | None = None,
    seed: int,
) -> list[float]:
    """Minimize the worst sampled clean loss, keeping both endpoints fixed."""
    parameters = path.parameters_to_optimize()
    if not parameters:
        return []
    if not texts or len(texts) != len(labels):
        raise ValueError("path fitting needs matched, nonempty clean examples")
    if steps < 1 or batch_size < 1 or lr <= 0:
        raise ValueError("steps, batch size, and learning rate must be positive")
    path.model.model.eval()
    if base_lr is not None and base_lr <= 0:
        raise ValueError("base learning rate must be positive")
    factor_parameters = [parameter for pair in path.control.values() for parameter in pair]
    groups = []
    if factor_parameters:
        groups.append({"params": factor_parameters, "lr": lr})
    if path.base_control:
        groups.append({"params": list(path.base_control.values()), "lr": base_lr or lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=0.0)
    rng = np.random.default_rng(seed)
    history = []
    positions = (0.2, 0.4, 0.6, 0.8)
    for _ in range(steps):
        indices = rng.choice(len(texts), size=min(batch_size, len(texts)), replace=False)
        batch_texts = [texts[int(i)] for i in indices]
        target = torch.tensor([labels[int(i)] for i in indices], device=path.device)
        with torch.no_grad():
            losses = [
                float(F.cross_entropy(path.logits(batch_texts, t).float(), target))
                for t in positions
            ]
        worst_t = positions[int(np.argmax(losses))]
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(path.logits(batch_texts, worst_t).float(), target)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
        if not math.isfinite(history[-1]):
            raise FloatingPointError("path optimization produced non-finite clean loss")
    return history
