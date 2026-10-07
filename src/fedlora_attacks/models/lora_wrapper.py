"""HF + PEFT LoRA wrapper and (A, B)/dW extraction (CLAUDE.md "models/").

Convention bridge (CRITICAL — a wrong sign/transpose here silently corrupts every
probe). PEFT stores, for a Linear(in=m, out=n):
    lora_A.weight : (r, m),  lora_B.weight : (n, r),  scaling = alpha / r,
    delta = scaling * lora_B @ lora_A  (shape n x m).
Plan convention: A in R^{n x r}, B in R^{r x m}, dW = A @ B. We therefore fold the
scaling into A:
    A = scaling * lora_B.weight   (n x r),   B = lora_A.weight   (r x m).
Then A @ B == PEFT's delta exactly (verified in tests/test_lora_extract.py against
``get_delta_weight``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer

from ..fl.types import AdapterDict, AdapterState


@dataclass
class ModelSpec:
    """Model/adapter configuration (mirrors configs/model/*.yaml)."""

    kind: str = "tiny_bert"  # 'tiny_bert' (hermetic) | 'hf'
    model_name: str | None = None  # for kind='hf', e.g. 'bert-base-uncased'
    # tiny_bert dims (ignored for kind='hf')
    hidden_size: int = 64
    num_hidden_layers: int = 2
    num_attention_heads: int = 2
    intermediate_size: int = 128
    max_position_embeddings: int = 64
    # LoRA
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: tuple[str, ...] = ("query", "value")
    tokenizer_kind: str = "simple"  # 'simple' (hermetic) | 'hf'
    amp: str | None = None  # 'bf16' | 'fp16' | None — autocast only, weights stay fp32
    task: str = "seq_cls"  # 'seq_cls' (BERT testbeds) | 'causal_lm' (Dolly / LLaMA)
    base_dtype: str = "fp32"
    """Dtype for the FROZEN base weights. 'bf16' halves the resident model (12 GB -> 6 GB
    at 3B) and is safe *for the base only*: those weights are never trained and never
    probed.

    It is NOT safe for the adapters, and the two are easy to conflate. PEFT creates the
    LoRA factors in the base layer's dtype, so loading a bf16 base would silently make
    (A, B) bf16 — an 8-bit mantissa, ~4e-3 relative noise, sitting exactly where
    invariant 6 says sigma_min lives. ``build_lora_model`` therefore forces every
    ``lora_`` parameter back to fp32 after attachment and ASSERTS it, rather than
    trusting peft's ``autocast_adapter_dtype`` default to stay the default.
    """
    max_prompt_tokens: int = 384
    """Cap on the prompt half of a causal-LM example, so the response always survives.

    Truncating the joined sequence from the right is the obvious thing and it is wrong:
    a long Dolly `context` would consume the whole window and leave zero response tokens,
    giving a batch with no supervised targets and an undefined loss.
    """
    image_size: int = 224
    """Side length the image processor produces, and the vision tower consumes.

    Carried on the spec rather than read off the checkpoint because it is also part of the
    pixel cache key (``data/eurosat.cache_key``): the cache stores PROCESSED pixels, so a
    changed crop size must invalidate it rather than silently feed the model inputs of the
    wrong scale.
    """
    lora_target_scope: str | None = None
    """Dotted-path prefix the LoRA targets are confined to. REQUIRED for CLIP.

    CLIP's two towers use identical leaf names (``q_proj``, ``v_proj``, ...), and PEFT
    matches a plain list by leaf SUFFIX — so an unscoped target list adapts the TEXT tower
    too, which makes the frozen zero-shot head trainable through the tower that produced
    it. Set, the targets are compiled to a full-path regex and the result is asserted; see
    ``models/clip_head.vision_target_regex``. None keeps the historical suffix matching for
    the single-tower spines (BERT, Qwen), where there is nothing to confine.
    """
    freeze_head: bool = False
    """Freeze the classification head: never trained, never federated (ABLATION ONLY).

    The head is the only trainable parameter outside the LoRA bundle, and under a
    defence its weighting is a judgement call (see the server's ``_aggregate_heads``).
    A reviewer will therefore ask whether an observed accuracy drop is head instability
    rather than the attack. With this set, ``get_head_state`` returns nothing, the server
    has no head to aggregate, and every accuracy change is attributable to the adapters —
    which is what experiment/e_a_frozen_head answers.

    Costs accuracy in absolute terms (the head stays at its random init and the adapters
    must adapt the features to it), so it is an ablation arm, never the main setup.
    """


def _build_base_model(spec: ModelSpec, vocab_size: int, num_labels: int):
    if spec.kind == "tiny_bert":
        from transformers import BertConfig, BertForSequenceClassification

        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=spec.hidden_size,
            num_hidden_layers=spec.num_hidden_layers,
            num_attention_heads=spec.num_attention_heads,
            intermediate_size=spec.intermediate_size,
            max_position_embeddings=spec.max_position_embeddings,
            num_labels=num_labels,
        )
        return BertForSequenceClassification(config)
    if spec.kind in ("tiny_llama", "tiny_qwen"):
        # Hermetic causal-LM path: the same module NAMES as the real spine
        # (q_proj/v_proj) and grouped-query attention, so v_proj is NON-SQUARE exactly as
        # it is at 3B. A square-only tiny model would let a shape bug through and surface
        # it only on the cluster.
        #
        # Both families are offered because they differ in one way that touches this
        # code: Qwen2 puts a BIAS on q_proj/v_proj and Llama does not. The bias stays
        # frozen and outside the LoRA bundle either way, but "stays outside" is a claim
        # worth testing on the architecture actually being run rather than inferred from
        # the other one.
        kwargs = dict(
            vocab_size=vocab_size,
            hidden_size=spec.hidden_size,
            intermediate_size=spec.intermediate_size,
            num_hidden_layers=spec.num_hidden_layers,
            num_attention_heads=spec.num_attention_heads,
            num_key_value_heads=max(1, spec.num_attention_heads // 2),
            max_position_embeddings=spec.max_position_embeddings,
        )
        if spec.kind == "tiny_qwen":
            from transformers import Qwen2Config, Qwen2ForCausalLM

            return Qwen2ForCausalLM(Qwen2Config(**kwargs))
        from transformers import LlamaConfig, LlamaForCausalLM

        return LlamaForCausalLM(LlamaConfig(**kwargs))
    if spec.kind == "hf":
        if spec.task == CAUSAL_LM:
            from transformers import AutoModelForCausalLM

            return AutoModelForCausalLM.from_pretrained(
                spec.model_name, torch_dtype=_BASE_DTYPES[spec.base_dtype]
            )
        from transformers import AutoModelForSequenceClassification

        return AutoModelForSequenceClassification.from_pretrained(
            spec.model_name, num_labels=num_labels
        )
    raise KeyError(f"unknown model kind '{spec.kind}' (expected 'tiny_bert', 'tiny_llama' or 'hf')")


def _build_clip(spec: ModelSpec, vocab_size: int):
    """The bare ``CLIPModel``: the real checkpoint, or a tiny random one for the tests.

    ``tiny_clip`` mirrors ``tiny_qwen``'s role — the same MODULE NAMES as the real spine,
    including the collision between the two towers, so the vision-only scoping is exercised
    offline rather than discovered on the cluster. The layer count is what the scoping
    assertion counts against, so it is taken from the spec and both towers get it.
    """
    from transformers import CLIPConfig, CLIPModel, CLIPTextConfig, CLIPVisionConfig

    if spec.kind == "tiny_clip":
        text = CLIPTextConfig(
            vocab_size=max(vocab_size, 8),
            hidden_size=spec.hidden_size,
            intermediate_size=spec.intermediate_size,
            num_hidden_layers=spec.num_hidden_layers,
            num_attention_heads=spec.num_attention_heads,
            max_position_embeddings=spec.max_position_embeddings,
        )
        vision = CLIPVisionConfig(
            hidden_size=spec.hidden_size,
            intermediate_size=spec.intermediate_size,
            num_hidden_layers=spec.num_hidden_layers,
            num_attention_heads=spec.num_attention_heads,
            image_size=spec.image_size,
            patch_size=max(1, spec.image_size // 4),
        )
        config = CLIPConfig.from_text_vision_configs(text, vision, projection_dim=spec.hidden_size)
        return CLIPModel(config)
    if spec.kind == "hf":
        return CLIPModel.from_pretrained(spec.model_name, torch_dtype=_BASE_DTYPES[spec.base_dtype])
    raise KeyError(f"task '{CLIP_ZEROSHOT_CLS}' needs kind 'hf' or 'tiny_clip', got {spec.kind!r}")


def _build_clip_classifier(
    spec: ModelSpec, tokenizer, num_labels: int, class_names, vocab_size: int
):
    """CLIP + the frozen text-derived head, ready for LoRA attachment.

    The head is built BEFORE the adapters exist, from the pretrained text tower, and is then
    frozen — so no adapter can ever have influenced it. Building it afterwards would make
    the classifier a function of the training run, which is the thing this design removes.
    """
    from .clip_head import CLIPZeroShotClassifier, build_zeroshot_head

    if not class_names:
        raise ValueError(
            "a clip_zeroshot_cls model needs class_names: the classifier IS the text "
            "embedding of the class names, so there is no head to build without them"
        )
    if len(class_names) != int(num_labels):
        raise ValueError(f"{len(class_names)} class names for {num_labels} labels")
    clip = _build_clip(spec, vocab_size)
    head = build_zeroshot_head(clip, tokenizer, list(class_names))
    return CLIPZeroShotClassifier(clip, head)


_AMP_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None, None: None}
_BASE_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}

SEQ_CLS = "seq_cls"
CAUSAL_LM = "causal_lm"
CLIP_ZEROSHOT_CLS = "clip_zeroshot_cls"
#: PEFT's task_type string for each of ours. CLIP gets None: the trained object is a plain
#: nn.Module wrapper (image tower + frozen head), not one of PEFT's task heads, and naming a
#: task type would have PEFT add `modules_to_save` — a full-rank trainable tensor, which is
#: precisely the second attack surface the frozen head exists to remove.
_PEFT_TASK = {SEQ_CLS: "SEQ_CLS", CAUSAL_LM: "CAUSAL_LM", CLIP_ZEROSHOT_CLS: None}

#: Label id the HF loss ignores. Prompt tokens carry it, so only response tokens are scored.
IGNORE_INDEX = -100


class LoRAModel:
    """Wraps a PEFT LoRA model, exposing adapter (A, B)/dW get/set and forward.

    Mixed precision (CLAUDE.md "Compute & hardware"): ``amp`` selects an autocast dtype
    for the forward/backward. Deliberately AUTOCAST, not bf16 master weights — bf16 has
    an 8-bit mantissa, so storing the LoRA factors in it would put the noise floor at
    ~4e-3 relative, destroying sigma_min exactly where invariant 6 says the science
    lives. With autocast the parameters stay fp32, so the (A, B) the probes extract are
    fp32 and their fp64 upcast is meaningful; only the matmuls run in bf16.
    """

    def __init__(
        self,
        peft_model,
        tokenizer,
        *,
        adapter_name: str = "default",
        max_length: int = 32,
        amp: str | None = None,
        max_prompt_tokens: int = 384,
    ) -> None:
        self.model = peft_model
        self.tokenizer = tokenizer
        self.adapter_name = adapter_name
        self.max_length = max_length
        self.max_prompt_tokens = min(max_prompt_tokens, max(1, max_length - 1))
        if amp not in _AMP_DTYPES:
            raise KeyError(f"unknown amp mode '{amp}' (expected 'bf16', 'fp16' or 'none')")
        self.amp_dtype = _AMP_DTYPES[amp]

    # -- LoRA layer access ------------------------------------------------
    def _lora_layers(self) -> dict[str, LoraLayer]:
        return {
            name: module
            for name, module in self.model.named_modules()
            if isinstance(module, LoraLayer) and self.adapter_name in module.lora_A
        }

    def get_adapters(self) -> AdapterDict:
        """Extract per-layer (A, B) in plan convention (scaling folded into A)."""
        out: AdapterDict = {}
        for name, module in self._lora_layers().items():
            a_lora = module.lora_A[self.adapter_name].weight  # (r, m)
            b_lora = module.lora_B[self.adapter_name].weight  # (n, r)
            scaling = float(module.scaling[self.adapter_name])
            A = (scaling * b_lora).detach().clone()  # (n, r)
            B = a_lora.detach().clone()  # (r, m)
            out[name] = AdapterState(A, B)
        return out

    def live_adapters(self) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Per-layer (A, B) as LIVE tensors WITH grad — for differentiable penalties.

        Same plan convention as ``get_adapters`` (A = scaling * lora_B, B = lora_A) but
        NOT detached: these are views on the live nn.Parameters, so a loss computed from
        them backprops into the LoRA weights. Used by X2's low-rank steering penalty; the
        detached ``get_adapters`` remains the path for probing and aggregation.
        """
        out = {}
        for name, module in self._lora_layers().items():
            a_lora = module.lora_A[self.adapter_name].weight  # (r, m)
            b_lora = module.lora_B[self.adapter_name].weight  # (n, r)
            scaling = float(module.scaling[self.adapter_name])
            out[name] = (scaling * b_lora, a_lora)  # (A live, B live)
        return out

    def set_adapters(self, adapters: AdapterDict) -> None:
        """Write plan-convention (A, B) back into PEFT factors (unfold scaling)."""
        layers = self._lora_layers()
        with torch.no_grad():
            for name, state in adapters.items():
                module = layers[name]
                scaling = float(module.scaling[self.adapter_name])
                dtype = module.lora_A[self.adapter_name].weight.dtype
                module.lora_A[self.adapter_name].weight.copy_(state.B.to(dtype))
                module.lora_B[self.adapter_name].weight.copy_((state.A / scaling).to(dtype))

    # -- classification head (non-LoRA trainable params) ------------------
    def get_head_state(self) -> dict[str, torch.Tensor]:
        """Trainable non-LoRA parameters (classifier / modules_to_save), federated by
        plain FedAvg in the server. Excluded from all gauge-bundle probes."""
        return {
            name: p.detach().clone()
            for name, p in self.model.named_parameters()
            if p.requires_grad and "lora_" not in name
        }

    def set_head_state(self, state: dict[str, torch.Tensor]) -> None:
        params = dict(self.model.named_parameters())
        with torch.no_grad():
            for name, tensor in state.items():
                params[name].copy_(tensor.to(params[name].dtype))

    def layer_delta_weights(self) -> dict[str, torch.Tensor]:
        """PEFT's own delta weights per layer (ground-truth for the extraction test)."""
        return {
            name: module.get_delta_weight(self.adapter_name).detach()
            for name, module in self._lora_layers().items()
        }

    # -- forward ----------------------------------------------------------
    def encode(self, texts: list[str], device: torch.device | str = "cpu"):
        batch = self.tokenizer(
            texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        return {k: v.to(device) for k, v in batch.items()}

    def logits(self, texts: list[str], device: torch.device | str = "cpu") -> torch.Tensor:
        batch = self.encode(texts, device)
        device_type = torch.device(device).type
        with torch.autocast(
            device_type=device_type,
            dtype=self.amp_dtype or torch.bfloat16,
            enabled=self.amp_dtype is not None,
        ):
            return self.model(**batch).logits

    # -- image classification ---------------------------------------------
    def image_logits(self, pixel_values, device: torch.device | str = "cpu") -> torch.Tensor:
        """``(B, C)`` class logits for a batch of preprocessed pixel values.

        ``pixel_values`` may be a tensor or any array (the EuroSAT cache hands over a
        float16 numpy slice); the classifier casts it to the patch embedding's dtype, so
        the storage dtype of the cache is not a constraint on the model dtype.
        """
        batch = torch.as_tensor(pixel_values).to(device)
        device_type = torch.device(device).type
        with torch.autocast(
            device_type=device_type,
            dtype=self.amp_dtype or torch.bfloat16,
            enabled=self.amp_dtype is not None,
        ):
            return self.model(pixel_values=batch)

    def image_loss(self, pixel_values, labels, device: torch.device | str = "cpu") -> torch.Tensor:
        """Mean cross-entropy over a batch of images. The logits are fp32 by construction."""
        logits = self.image_logits(pixel_values, device)
        target = torch.as_tensor(labels, dtype=torch.long, device=logits.device)
        if logits.shape[0] != target.shape[0]:
            raise ValueError(f"{logits.shape[0]} images vs {target.shape[0]} labels")
        return torch.nn.functional.cross_entropy(logits, target)

    # -- causal LM ---------------------------------------------------------
    def encode_causal(
        self, prompts: list[str], responses: list[str], device: torch.device | str = "cpu"
    ) -> dict[str, torch.Tensor]:
        """(input_ids, attention_mask, labels) with PROMPT TOKENS MASKED to -100.

        The prompt and the response are tokenized SEPARATELY and concatenated, rather
        than joined as a string and split by character offset: for a BPE tokenizer the
        boundary token can span the join, so a character-offset split silently mislabels
        one token per example. Tokenizing the halves makes the boundary exact by
        construction.

        The prompt is capped at ``max_prompt_tokens`` so the response always has room —
        see ModelSpec.max_prompt_tokens for why right-truncating the joined sequence is
        the wrong shape here. Padding is on the RIGHT, and pad positions get both
        ``attention_mask = 0`` and ``labels = -100``, so they contribute to neither the
        attention nor the loss.
        """
        if len(prompts) != len(responses):
            raise ValueError(f"{len(prompts)} prompts vs {len(responses)} responses")
        tok = self.tokenizer
        eos = tok.eos_token_id
        rows_ids, rows_labels = [], []
        room = max(1, self.max_length - 1)
        for prompt, response in zip(prompts, responses, strict=True):
            p_ids = tok(prompt, add_special_tokens=True)["input_ids"][: self.max_prompt_tokens]
            budget = max(1, min(room, self.max_length) - len(p_ids))
            r_ids = tok(response, add_special_tokens=False)["input_ids"][:budget]
            if eos is not None and len(p_ids) + len(r_ids) < self.max_length:
                r_ids = list(r_ids) + [eos]  # teach the model to stop
            ids = list(p_ids) + list(r_ids)
            rows_ids.append(ids)
            rows_labels.append([IGNORE_INDEX] * len(p_ids) + list(r_ids))
        width = max(len(r) for r in rows_ids)
        pad = tok.pad_token_id if tok.pad_token_id is not None else (eos or 0)
        input_ids, attention, labels = [], [], []
        for ids, lab in zip(rows_ids, rows_labels, strict=True):
            gap = width - len(ids)
            input_ids.append(ids + [pad] * gap)
            attention.append([1] * len(ids) + [0] * gap)
            labels.append(lab + [IGNORE_INDEX] * gap)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
            "attention_mask": torch.tensor(attention, dtype=torch.long, device=device),
            "labels": torch.tensor(labels, dtype=torch.long, device=device),
        }

    def lm_loss(
        self, prompts: list[str], responses: list[str], device: torch.device | str = "cpu"
    ) -> torch.Tensor:
        """Mean cross-entropy over RESPONSE tokens only. HF applies the causal shift."""
        batch = self.encode_causal(prompts, responses, device)
        device_type = torch.device(device).type
        with torch.autocast(
            device_type=device_type,
            dtype=self.amp_dtype or torch.bfloat16,
            enabled=self.amp_dtype is not None,
        ):
            return self.model(**batch).loss

    @torch.no_grad()
    def generate(
        self,
        prompts: list[str],
        device: torch.device | str = "cpu",
        max_new_tokens: int = 128,
    ) -> list[str]:
        """Greedy continuations, with only the NEW tokens decoded.

        Batched decoder-only generation needs LEFT padding: with right padding the pads
        sit between the prompt and the first generated token, and the model continues
        from padding instead of from the instruction. The padding side is set for the
        call and restored, so training (which needs right padding) is unaffected.
        """
        tok = self.tokenizer
        previous = getattr(tok, "padding_side", "right")
        tok.padding_side = "left"
        try:
            batch = tok(
                prompts,
                padding=True,
                truncation=True,
                max_length=self.max_prompt_tokens,
                return_tensors="pt",
            )
            batch = {k: v.to(device) for k, v in batch.items()}
            was_training = self.model.training
            self.model.eval()
            out = self.model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
            )
            if was_training:
                self.model.train()
            fresh = out[:, batch["input_ids"].shape[1] :]
            return [tok.decode(row, skip_special_tokens=True).strip() for row in fresh]
        finally:
            tok.padding_side = previous

    def to(self, device) -> LoRAModel:
        self.model.to(device)
        return self

    def num_lora_layers(self) -> int:
        return len(self._lora_layers())


def _force_fp32_adapters(peft_model) -> None:
    """Every ``lora_`` parameter in fp32, asserted rather than assumed.

    With a bf16 base, PEFT creates the factors in the base dtype unless
    ``autocast_adapter_dtype`` intervenes. That default is not something the science
    should rest on: bf16 factors put a ~4e-3 relative noise floor straight through
    sigma_min and the whole spectral probe, and nothing downstream would announce it —
    the numbers would simply be wrong. So cast, then check.
    """
    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if "lora_" in name and param.dtype != torch.float32:
                param.data = param.data.to(torch.float32)
    bad = {
        name: str(p.dtype)
        for name, p in peft_model.named_parameters()
        if "lora_" in name and p.dtype != torch.float32
    }
    if bad:  # pragma: no cover - the cast above makes this unreachable
        raise RuntimeError(f"LoRA factors not fp32 after cast: {bad}")


def build_lora_model(
    spec: ModelSpec,
    tokenizer,
    num_labels: int,
    *,
    max_length: int = 32,
    class_names: list[str] | tuple[str, ...] | None = None,
):
    """Build a tiny/HF base model, attach LoRA, and wrap it.

    ``class_names`` is consumed only by ``clip_zeroshot_cls``, whose classifier IS the text
    embedding of those names. Keyword-only and defaulted, so every existing call site is
    unchanged.
    """
    vocab_size = getattr(tokenizer, "vocab_size", 0)
    if spec.task == CLIP_ZEROSHOT_CLS:
        base = _build_clip_classifier(spec, tokenizer, num_labels, class_names, vocab_size)
    else:
        base = _build_base_model(spec, vocab_size, num_labels)
    if spec.task == CAUSAL_LM and getattr(tokenizer, "pad_token_id", None) is None:
        # LLaMA ships without a pad token. Reusing EOS is the standard choice and is safe
        # here because pad positions are masked out of BOTH attention and the loss, so a
        # pad that happens to equal EOS is never scored.
        tokenizer.pad_token = tokenizer.eos_token
    # A scope turns the leaf list into a full-path regex. Without it PEFT matches by leaf
    # SUFFIX, which on CLIP adapts both towers — see models/clip_head for why that quietly
    # un-freezes the classifier.
    targets: str | list[str] = list(spec.target_modules)
    if spec.lora_target_scope:
        from .clip_head import vision_target_regex

        targets = vision_target_regex(spec.lora_target_scope, spec.target_modules)
    lora_config = LoraConfig(
        r=spec.lora_r,
        lora_alpha=spec.lora_alpha,
        lora_dropout=spec.lora_dropout,
        target_modules=targets,
        bias="none",
        task_type=_PEFT_TASK[spec.task],
    )
    peft_model = get_peft_model(base, lora_config)
    _force_fp32_adapters(peft_model)
    if spec.task == CLIP_ZEROSHOT_CLS:
        if not spec.lora_target_scope:
            raise ValueError(
                "a clip_zeroshot_cls model needs model.lora_target_scope (e.g. "
                "'vision_model'). Unscoped, PEFT's suffix matching adapts the TEXT tower "
                "too and the frozen zero-shot head stops being frozen."
            )
        from .clip_head import assert_vision_only_adapters

        assert_vision_only_adapters(
            peft_model,
            target_modules=spec.target_modules,
            num_layers=base.clip.config.vision_config.num_hidden_layers,
            scope=spec.lora_target_scope,
        )
    if spec.freeze_head:
        # Everything trainable outside the LoRA factors. get_head_state() filters on
        # requires_grad, so this alone removes the head from training, from the optimizer
        # (client._train collects requires_grad params) and from federation.
        for name, p in peft_model.named_parameters():
            if p.requires_grad and "lora_" not in name:
                p.requires_grad_(False)
    return LoRAModel(
        peft_model,
        tokenizer,
        max_length=max_length,
        amp=spec.amp,
        max_prompt_tokens=spec.max_prompt_tokens,
    )
