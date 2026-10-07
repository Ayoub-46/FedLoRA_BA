"""Tokenizers.

Two paths:
  - ``SimpleTokenizer`` — a hermetic whitespace/word-level tokenizer built from a
    fixed vocab. No downloads; used by the tiny-model + synthetic CI path so the
    smoke test and unit tests run offline.
  - HuggingFace ``AutoTokenizer`` — for real runs (bert-base + SST-2), built via
    ``build_hf_tokenizer`` (requires a one-time model download).

Both expose the minimal HF-compatible call surface the clients rely on:
``tok(texts, ...) -> {"input_ids": LongTensor, "attention_mask": LongTensor}`` and a
``vocab_size`` (so the tiny model's embedding matches).
"""

from __future__ import annotations

import torch


class SimpleTokenizer:
    """Deterministic word-level tokenizer over a fixed vocabulary."""

    def __init__(self, vocab: list[str]) -> None:
        specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
        seen = set(specials)
        itos = list(specials)
        for w in vocab:
            if w not in seen:
                itos.append(w)
                seen.add(w)
        self.itos = itos
        self.stoi = {w: i for i, w in enumerate(itos)}
        self.pad_token_id = self.stoi["[PAD]"]
        self.unk_id = self.stoi["[UNK]"]
        self.cls_id = self.stoi["[CLS]"]
        self.sep_id = self.stoi["[SEP]"]
        # Causal-LM surface, so the hermetic path exercises the SAME code as a real run
        # rather than a parallel one. [SEP] doubles as EOS: the tests care that a stop
        # token exists and is masked correctly, not which glyph it is.
        self.eos_token_id = self.sep_id
        self.eos_token = "[SEP]"
        self.pad_token = "[PAD]"
        self.padding_side = "right"

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        specials = {self.pad_token_id, self.cls_id, self.sep_id}
        words = [
            self.itos[int(i)]
            for i in ids
            if not (skip_special_tokens and int(i) in specials) and int(i) < len(self.itos)
        ]
        return " ".join(words)

    def __call__(
        self,
        texts,
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 32,
        return_tensors: str = "pt",
        add_special_tokens: bool = True,
    ) -> dict[str, torch.Tensor] | dict[str, list[int]]:
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        seqs = []
        for t in texts:
            toks = t.split()
            if truncation:
                toks = toks[: max_length - 2]
            body = [self.stoi.get(w, self.unk_id) for w in toks]
            ids = ([self.cls_id] + body + [self.sep_id]) if add_special_tokens else body
            seqs.append(ids)
        if single:
            # HF returns plain lists for a single string; `encode_causal` relies on that
            # to slice prompt and response ids before they are padded into a batch.
            return {"input_ids": seqs[0], "attention_mask": [1] * len(seqs[0])}
        maxlen = max(len(s) for s in seqs) if seqs else 2
        input_ids, attn = [], []
        for s in seqs:
            pad = maxlen - len(s)
            if self.padding_side == "left":
                input_ids.append([self.pad_token_id] * pad + s)
                attn.append([0] * pad + [1] * len(s))
            else:
                input_ids.append(s + [self.pad_token_id] * pad)
                attn.append([1] * len(s) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


def build_hf_tokenizer(model_name: str):
    """Load a HuggingFace tokenizer (real-run path; downloads once)."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name)
