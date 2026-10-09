"""Path rank, gauge, and replay properties on the hermetic tiny model."""

from __future__ import annotations

import torch

from fedlora_attacks.data.base import build_federated_data, build_tokenizer
from fedlora_attacks.fl.types import gauge_transform
from fedlora_attacks.models.lora_wrapper import ModelSpec, build_lora_model
from fedlora_attacks.paths.connectivity import ConnectivityPath, EmbeddedPath, optimize_path
from fedlora_attacks.paths.h0 import sampled_merge_events
from fedlora_attacks.utils.seeding import seed_everything


def _model(rank, tokenizer):
    seed_everything(7)
    spec = ModelSpec(
        kind="tiny_bert", hidden_size=32, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=64, max_position_embeddings=64,
        lora_r=rank, lora_alpha=16, target_modules=("query", "value"),
        tokenizer_kind="simple",
    )
    model = build_lora_model(spec, tokenizer, 2, max_length=32)
    model.model.eval()
    return model


def test_expanded_line_is_exact_and_replays_lower_rank_path(tmp_path):
    torch.set_num_threads(4)
    data = build_federated_data("synthetic", 128, 16, 2, float("inf"), 7)
    tokenizer = build_tokenizer("simple", data.data)
    low_model = _model(4, tokenizer)
    poisoned = low_model.get_adapters()
    recovered = {name: state.clone() for name, state in poisoned.items()}
    for state in recovered.values():
        state.A.add_(torch.randn_like(state.A) * 0.02)
    head = low_model.get_head_state()
    low_path = ConnectivityPath(
        low_model, poisoned, recovered, head, head, mode="rank_r"
    )
    high_model = _model(8, tokenizer)
    line = ConnectivityPath(
        high_model, poisoned, recovered, head, head, mode="rank_2r_linear"
    )
    embedded = EmbeddedPath(low_path, high_model)
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        for name, state in line.adapters_at(t).items():
            expected = (1 - t) * poisoned[name].delta_w() + t * recovered[name].delta_w()
            torch.testing.assert_close(state.delta_w(), expected, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            low_path.logits(data.data.test_texts[:4], t),
            embedded.logits(data.data.test_texts[:4], t),
            atol=1e-5, rtol=1e-5,
        )
    optimize_path(
        low_path, data.data.train_texts[:24], data.data.train_labels[:24],
        steps=2, batch_size=8, lr=0.001, seed=1,
    )
    path_file = tmp_path / "rank_r.pt"
    low_path.save(path_file)
    reloaded = ConnectivityPath.load(
        path_file, low_model, poisoned, recovered, head, head
    )
    torch.testing.assert_close(
        low_path.logits(data.data.test_texts[:4], 0.4),
        reloaded.logits(data.data.test_texts[:4], 0.4),
    )


def test_effective_line_ignores_endpoint_factor_gauge():
    data = build_federated_data("synthetic", 128, 8, 2, float("inf"), 7)
    tokenizer = build_tokenizer("simple", data.data)
    low_model = _model(4, tokenizer)
    poisoned = low_model.get_adapters()
    recovered = {name: state.clone() for name, state in poisoned.items()}
    for state in recovered.values():
        state.A.add_(torch.randn_like(state.A) * 0.02)
    transformed = {
        name: gauge_transform(state, torch.diag(torch.tensor([2.0, 0.5, 1.5, 0.75])))
        for name, state in recovered.items()
    }
    high_model = _model(8, tokenizer)
    head = low_model.get_head_state()
    original = ConnectivityPath(
        high_model, poisoned, recovered, head, head, mode="rank_2r_linear"
    )
    regauged = ConnectivityPath(
        high_model, poisoned, transformed, head, head, mode="rank_2r_linear"
    )
    for name in poisoned:
        torch.testing.assert_close(
            original.adapters_at(0.37)[name].delta_w(),
            regauged.adapters_at(0.37)[name].delta_w(),
            atol=1e-5, rtol=1e-5,
        )


def test_full_path_bends_base_but_keeps_head_linear():
    data = build_federated_data("synthetic", 64, 8, 2, float("inf"), 7)
    tokenizer = build_tokenizer("simple", data.data)
    source = _model(4, tokenizer)
    model = _model(8, tokenizer)
    endpoint = source.get_adapters()
    head_p = source.get_head_state()
    head_r = {name: value + 0.1 for name, value in head_p.items()}
    path = ConnectivityPath(
        model, endpoint, endpoint, head_p, head_r, mode="full_curve"
    )
    name = next(iter(path.base_control))
    with torch.no_grad():
        path.base_control[name].fill_(0.25)
    original = dict(model.model.named_parameters())[name]
    torch.testing.assert_close(path.overrides(0.0)[name], original)
    torch.testing.assert_close(path.overrides(1.0)[name], original)
    torch.testing.assert_close(path.overrides(0.5)[name], original + 0.125)
    for head_name in head_p:
        torch.testing.assert_close(
            path.overrides(0.5)[head_name], (head_p[head_name] + head_r[head_name]) / 2
        )


def test_sampled_h0_uses_lowest_available_path_threshold():
    events = sampled_merge_events(
        {"P": 0.2, "R": 0.3},
        [
            {"u": "P", "v": "R", "name": "straight", "maximum_loss": 1.0},
            {"u": "P", "v": "R", "name": "curve", "maximum_loss": 0.35},
        ],
    )
    merges = [event for event in events if event["kind"] == "merge"]
    assert len(merges) == 1
    assert merges[0]["threshold"] == 0.35
    assert merges[0]["edge"] == "curve"
