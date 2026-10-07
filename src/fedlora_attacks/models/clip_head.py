"""CLIP zero-shot classification head, and the vision-only LoRA scoping that protects it.

THE LOAD-BEARING DESIGN CONSTRAINT. The 10 class embeddings are built ONCE from CLIP's
text tower, L2-normalised, and frozen — never trained, never averaged, never checkpointed
per round. ``logit_scale`` is frozen for the same reason. This is NOT how CLIP is usually
fine-tuned (most work trains the head or fits a linear probe), and a reviewer will ask.
The answer is the research question: the LoRA factors must be the ONLY trainable and
therefore the only AGGREGATED parameters. A trainable full-rank head would be a second,
larger, un-gauged attack surface sitting in the same federation, and every damage number
would then be a mixture of "what the coalition did to the adapters" and "what it did to a
plain averaged tensor" — with nothing in the output separating them.

THE OTHER HALF OF THAT INVARIANT IS THE LoRA SCOPE, and it is easy to break silently.
CLIP's two towers use the SAME leaf module names::

    vision_model.encoder.layers.{0..11}.self_attn.{q_proj,k_proj,v_proj,out_proj}
    vision_model.encoder.layers.{0..11}.mlp.{fc1,fc2}
    text_model.encoder.layers.{0..11}.self_attn.{q_proj,k_proj,v_proj,out_proj}   # <-- same

PEFT's default matching is by leaf SUFFIX, so ``target_modules=["q_proj","v_proj"]`` adapts
BOTH towers — which un-freezes the classifier through the back door while every explicit
"is the head frozen?" check still passes, because the head tensor itself never moves; the
tower that PRODUCED it does. So the targets are compiled to a regex over the full dotted
path (``vision_target_regex``) and the result is ASSERTED (``assert_vision_only_adapters``),
rather than trusted.

Note the leaf names differ from the Qwen spine: ``out_proj`` not ``o_proj``, ``fc1``/``fc2``
not ``gate_proj``/``up_proj``/``down_proj``. All CLIP vision projections are square 768x768
(attention) or 768x3072 / 3072x768 (mlp) — no grouped-query attention, so unlike Qwen's
``v_proj`` none of the per-layer dW are non-square. Per-layer probes are unaffected either
way; a global stacked-layer SVD remains forbidden by CLAUDE.md.

MODULE COUNT, against the LLM configs: 24 at ``[q_proj, v_proj]``, 72 at all six, versus
Qwen2.5-1.5B's 56 and 3B's 72 at their own defaults.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import torch
from torch import nn

#: CLIP's published EuroSAT prompt templates.
CLIP_EUROSAT_TEMPLATES: tuple[str, ...] = (
    "a centered satellite photo of {}.",
    "a centered satellite photo of a {}.",
    "a centered satellite photo of the {}.",
)

#: Stored CamelCase label -> the natural-language name CLIP's text tower actually scores.
#:
#: This remapping is worth several points of zero-shot accuracy and is NOT cosmetic:
#: "HerbaceousVegetation" is not a phrase the text encoder has a good representation for,
#: while "brushland or shrubland" is. The keys are exactly ``data.eurosat.EUROSAT_CLASSES``
#: (a test pins that), and the zero-shot assert in the prefetch job is what catches anyone
#: quietly reverting to the raw stored names.
EUROSAT_PROMPT_NAMES: dict[str, str] = {
    "AnnualCrop": "annual crop land",
    "Forest": "forest",
    "HerbaceousVegetation": "brushland or shrubland",
    "Highway": "highway or road",
    "Industrial": "industrial buildings or commercial buildings",
    "Pasture": "pasture land",
    "PermanentCrop": "permanent crop land",
    "Residential": "residential buildings or homes or apartments",
    "River": "river",
    "SeaLake": "lake or sea",
}

#: The same names as a tuple, in stored label order — the form ``build_zeroshot_head`` wants.
CLIP_EUROSAT_CLASSNAMES: tuple[str, ...] = tuple(EUROSAT_PROMPT_NAMES.values())


def zeroshot_prompt_names(label_names: Sequence[str]) -> list[str]:
    """Map a dataset's STORED class names to the prompt names, in label order.

    Raises rather than passing an unknown name through. A silent passthrough is exactly the
    failure this indirection exists to prevent: the run would still produce a head, still
    produce accuracies, and simply be several points worse for a reason no metric reports.
    """
    unknown = [n for n in label_names if n not in EUROSAT_PROMPT_NAMES]
    if unknown:
        raise KeyError(
            f"no CLIP prompt name for {unknown}. Only EuroSAT's vocabulary is wired for "
            "zero-shot classification; add the mapping here rather than prompting with the "
            "raw stored names, which costs several points of accuracy silently."
        )
    return [EUROSAT_PROMPT_NAMES[n] for n in label_names]


@torch.no_grad()
def build_zeroshot_head(
    model,
    processor,
    class_names: Sequence[str],
    templates: Sequence[str] = CLIP_EUROSAT_TEMPLATES,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """The frozen ``(C, D)`` classifier: one L2-normalised text embedding per class.

    NORMALISE, AVERAGE, NORMALISE — OpenAI's prompt-ensembling convention, and the one the
    published EuroSAT zero-shot numbers are computed under. Averaging the raw embeddings
    first would weight the templates by their norms rather than equally, which shifts the
    result by a point or two: enough to break the [0.40, 0.58] gate in the prefetch job and
    not enough to look obviously wrong.

    ``processor`` may be a ``CLIPProcessor`` or a bare tokenizer; the text half is taken
    either way. Returned in fp32 regardless of the base dtype, so the head is bit-identical
    across calls and does not carry the base's bf16 mantissa into every logit.
    """
    if not class_names:
        raise ValueError("build_zeroshot_head needs at least one class name")
    if not templates:
        raise ValueError("build_zeroshot_head needs at least one prompt template")
    tokenizer = getattr(processor, "tokenizer", processor)
    was_training = model.training
    model.eval()
    try:
        rows = []
        for name in class_names:
            prompts = [t.format(name) for t in templates]
            batch = tokenizer(prompts, padding=True, return_tensors="pt")
            batch = {k: v.to(device) for k, v in batch.items()}
            emb = model.get_text_features(**batch).float()
            emb = emb / emb.norm(dim=-1, keepdim=True)
            mean = emb.mean(dim=0)
            rows.append(mean / mean.norm())
        return torch.stack(rows)
    finally:
        if was_training:
            model.train()


class CLIPZeroShotClassifier(nn.Module):
    """CLIP image tower + a frozen text-derived head. ``forward(pixel_values) -> logits``.

    Wraps ``CLIPModel`` rather than ``CLIPVisionModel`` because the head is built from the
    text tower and scored in the SHARED projection space — the visual projection is part of
    that path and must be the checkpoint's own.

    Everything except the LoRA factors ends up frozen: the text tower and ``logit_scale``
    explicitly here (so the invariant holds and is testable before PEFT is involved), and
    the rest by ``get_peft_model``. The consequence worth stating: ``get_head_state()``
    returns EMPTY for this model, so there is no non-LoRA tensor for a future server to
    average, which is the whole point.
    """

    def __init__(self, clip, head: torch.Tensor) -> None:
        super().__init__()
        if head.ndim != 2:
            raise ValueError(f"head must be (C, D), got {tuple(head.shape)}")
        self.clip = clip
        # Non-persistent: the head is DERIVED from the frozen text tower, so checkpointing
        # it would create a second copy that can silently disagree with the tower it came
        # from. It is rebuilt, identically, on every load.
        self.register_buffer("head", head.detach().clone().float(), persistent=False)
        self.clip.logit_scale.requires_grad_(False)
        for param in self.clip.text_model.parameters():
            param.requires_grad_(False)
        self.clip.text_projection.requires_grad_(False)

    @property
    def num_classes(self) -> int:
        return int(self.head.shape[0])

    def forward(self, pixel_values: torch.Tensor, **_: object) -> torch.Tensor:
        """``logit_scale.exp() * (normalised image embedding @ head.T)``."""
        # Cast to the patch embedding's dtype rather than assuming the caller matched it:
        # the cache stores float16, the base may be bf16 or fp32, and a mismatch here is a
        # hard error at the first conv rather than anything the run could recover from.
        weight = self.clip.vision_model.embeddings.patch_embedding.weight
        pixel_values = pixel_values.to(dtype=weight.dtype)
        image = self.clip.get_image_features(pixel_values=pixel_values)
        image = image / image.norm(dim=-1, keepdim=True)
        logits = self.clip.logit_scale.exp() * (image @ self.head.to(image.dtype).T)
        return logits.float()


def vision_target_regex(scope: str, target_modules: Sequence[str]) -> str:
    """Compile leaf module names into a full-path regex confined to ``scope``.

    PEFT matches a plain list by leaf suffix, which cannot distinguish the two CLIP towers.
    A regex is matched with ``re.fullmatch`` against the whole dotted path, so anchoring it
    on ``vision_model.`` is what makes "the text tower is never adapted" a property of the
    matcher rather than a property of the names happening not to collide.
    """
    if not target_modules:
        raise ValueError("vision_target_regex needs at least one target module")
    leaves = "|".join(re.escape(m) for m in sorted(set(target_modules)))
    return rf"(.*\.)?{re.escape(scope)}\..*\.({leaves})"


def adapter_paths(peft_model) -> list[str]:
    """Dotted paths of every module PEFT actually attached a LoRA adapter to."""
    from peft.tuners.lora import LoraLayer

    return [name for name, module in peft_model.named_modules() if isinstance(module, LoraLayer)]


def assert_vision_only_adapters(
    peft_model,
    *,
    target_modules: Sequence[str],
    num_layers: int,
    scope: str = "vision_model",
) -> list[str]:
    """Assert the adapters landed on the vision tower, all of it, and nothing else.

    Two separate failures, both silent without this check:

    * an adapter on ``text_model`` un-freezes the classifier through the tower that
      produced it, while the head tensor itself never moves and every direct "is the head
      frozen?" assertion still passes;
    * a target that matched NOTHING (a renamed leaf, a typo) leaves the run training fewer
      modules than the config says, and the accuracy curve is merely a bit worse.

    Returns the adapter paths, so a caller can log what it got rather than what it asked for.
    """
    paths = adapter_paths(peft_model)
    leaked = sorted(p for p in paths if "text_model" in p)
    if leaked:
        raise RuntimeError(
            f"{len(leaked)} LoRA adapter(s) landed on the TEXT tower, e.g. {leaked[:3]}. "
            "The zero-shot head is derived from that tower, so training it makes the "
            "classifier trainable and federated — the one thing this setup exists to "
            f"prevent. Scope the targets to '{scope}.' with vision_target_regex."
        )
    expected = int(num_layers) * len(set(target_modules))
    if len(paths) != expected:
        raise RuntimeError(
            f"expected {expected} adapted modules ({num_layers} layers x "
            f"{len(set(target_modules))} targets), got {len(paths)}. A target that matched "
            "nothing trains a smaller adapter than the config claims and says so nowhere."
        )
    return paths
