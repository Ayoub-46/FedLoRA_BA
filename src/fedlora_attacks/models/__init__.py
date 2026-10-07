"""Models: HF + PEFT LoRA wrapper, per-layer (A, B)/dW extraction, CLIP zero-shot head."""

from .clip_head import (
    CLIPZeroShotClassifier,
    build_zeroshot_head,
    vision_target_regex,
    zeroshot_prompt_names,
)
from .lora_wrapper import LoRAModel, ModelSpec, build_lora_model

__all__ = [
    "LoRAModel",
    "ModelSpec",
    "build_lora_model",
    "CLIPZeroShotClassifier",
    "build_zeroshot_head",
    "vision_target_regex",
    "zeroshot_prompt_names",
]
