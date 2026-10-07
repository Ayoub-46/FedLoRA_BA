"""FL client: local LoRA training with attack hooks (CLAUDE.md "fl/").

In-process simulation: clients share ONE ``LoRAModel`` (memory-cheap); each client
resets it to the broadcast global adapters before training, so sequential reuse is
safe. The client is where the data-space (poison) and update-space (transform) attack
hooks fire; the objective-space hook (loss_penalty) fires inside the training loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..attacks.base import Attack, AttackContext
from ..models.lora_wrapper import LoRAModel
from .types import AdapterDict, ClientUpdate


@dataclass
class ClientConfig:
    local_epochs: int = 1
    lr: float = 5e-4
    batch_size: int = 16
    weight_decay: float = 0.0
    local_steps: int | None = None
    """Fixed optimizer steps per round; overrides ``local_epochs`` when set.

    Preferred for the heterogeneity sweep. Under ``local_epochs`` the step count is
    proportional to shard size, so a data-rich client takes ~40x more steps than a
    data-poor one and its ||dW|| is correspondingly larger — a confound that would show
    up in the sigma/effective-rank profile as if it were a heterogeneity effect. Fixing
    the step count equalizes the work (and the drift) across clients, so what varies
    with alpha is the DATA distribution alone, which is the intended axis. Small shards
    cycle their data as needed to reach the step count.
    """


class LoRAClient:
    def __init__(
        self,
        client_id: int,
        texts: list[str],
        labels: list[int],
        attack: Attack,
        cfg: ClientConfig,
        responses: list[str] | None = None,
        images=None,
    ) -> None:
        self.client_id = client_id
        self.texts = texts
        self.labels = labels
        self.attack = attack
        self.cfg = cfg
        self.last_train_loss: float = float("nan")
        self.responses = responses
        """Generation targets for instruction tuning; ``None`` for classification.

        Their presence is the ONLY thing that selects the causal-LM loss — one
        discriminator, carried by the data itself, so there is no task flag that could
        disagree with what the client actually holds. For a causal-LM client ``texts``
        are prompts and ``labels`` are Dolly categories (used for the partition and the
        skew index, never as a training target).
        """
        self.images = images
        """This client's image shard (a ``data.eurosat.PixelView``); ``None`` for text.

        Same discriminator convention as ``responses``: its presence — not a task flag —
        selects the image loss, so nothing can disagree with what the client actually holds.
        For an image client ``texts`` is EMPTY and ``labels`` are the class targets.
        """
        if responses is not None and len(responses) != len(texts):
            raise ValueError(
                f"client {client_id}: {len(texts)} prompts vs {len(responses)} responses; "
                "the partition indexed them inconsistently"
            )
        if images is not None:
            if responses is not None:
                raise ValueError(f"client {client_id}: cannot hold both images and responses")
            if len(images) != len(labels):
                raise ValueError(
                    f"client {client_id}: {len(images)} images vs {len(labels)} labels; "
                    "the partition indexed them inconsistently"
                )

    def local_train(
        self,
        model: LoRAModel,
        global_adapters: AdapterDict,
        round_idx: int,
        device="cpu",
        rng: np.random.Generator | None = None,
        global_head: dict | None = None,
    ) -> ClientUpdate:
        rng = rng or np.random.default_rng(round_idx * 1000 + self.client_id)
        model.set_adapters(global_adapters)
        if global_head is not None:
            model.set_head_state(global_head)
        if self.images is not None and "poison_dataset" in self.attack.HOOKS:
            # The data-space hook is text-typed by signature, and an image client's `texts`
            # is empty — so a poisoning attack would silently receive nothing to poison and
            # report success. Paper 1's attacks are update-space only, so this is a guard
            # against a future one being wired here without its image path.
            raise NotImplementedError(
                f"attack {type(self.attack).__name__} declares the poison_dataset hook, "
                "which has no image implementation. Update-space attacks (x1/x5/x6) leave "
                "the dataset identity and are unaffected."
            )
        texts, labels = self.attack.poison_dataset(
            self.texts, self.labels, self.client_id, round_idx, rng
        )
        self._train(model, texts, labels, round_idx, device, rng)
        update = ClientUpdate(
            client_id=self.client_id,
            adapters=model.get_adapters(),
            num_examples=len(labels),
            head_state=model.get_head_state(),
            metadata={
                "malicious": self.attack.is_malicious(self.client_id),
                "active": self.attack.is_active(self.client_id, round_idx),
            },
        )
        # Update-space attack hook (Channel A / B). The context carries the BROADCAST
        # global — legal under the v3 threat model (§1.2) and what a gauge attacker uses
        # to estimate the other clients' factors — and nothing else. Benign clients'
        # data and current-round updates are not reachable from here; see
        # attacks/base.AttackContext and tests/test_attack_contract.py.
        ctx = AttackContext(
            round_idx=round_idx,
            client_id=self.client_id,
            global_adapters=global_adapters,
            rng=rng,
        )
        return self.attack.transform_update(update, ctx)

    def _batches(self, n: int, rng):
        """Yield index batches: fixed step count if configured, else whole epochs."""
        idx = np.arange(n)
        bs = self.cfg.batch_size
        if self.cfg.local_steps is None:
            for _ in range(self.cfg.local_epochs):
                rng.shuffle(idx)
                for start in range(0, n, bs):
                    yield idx[start : start + bs]
            return
        emitted = 0
        while emitted < self.cfg.local_steps:
            rng.shuffle(idx)
            for start in range(0, n, bs):
                if emitted >= self.cfg.local_steps:
                    return
                yield idx[start : start + bs]
                emitted += 1

    def _train(self, model, texts, labels, round_idx, device, rng) -> None:
        """Local SGD. The task changes the LOSS and nothing else.

        Batching, the optimizer, the RNG draw order and both attack hooks are shared
        between the two tasks deliberately. Invariant 5's round-0 bit-identity depends on
        training consuming this client's RNG BEFORE ``transform_update`` runs, so a
        second training loop for instruction tuning would be a second place for that
        ordering to drift out of agreement.
        """
        image = self.images is not None
        # The step budget comes from what the client HOLDS: an image client's `texts` is
        # empty by construction, so keying the early return on it would skip training
        # entirely and report a benign-looking zero update.
        n_local = len(self.images) if image else len(texts)
        if n_local == 0:
            return
        model.model.train()
        params = [p for p in model.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        loss_fn = torch.nn.CrossEntropyLoss()
        causal = self.responses is not None
        if causal and len(texts) != len(self.responses):
            # A data-space attack reordered or resized the prompts, so prompt i and
            # response i are no longer the same example. Silently training on mismatched
            # pairs would look like a working run and poison every number in it.
            raise ValueError(
                f"client {self.client_id}: poison_dataset returned {len(texts)} prompts "
                f"for {len(self.responses)} responses. Data-space hooks are not supported "
                "on the causal-LM path; the update-space attacks (x1/x5/x6) leave the "
                "dataset identity and are unaffected."
            )
        seen, total = 0, 0.0
        for bidx in self._batches(n_local, rng):
            optimizer.zero_grad()
            if image:
                loss = model.image_loss(
                    self.images.take(bidx),
                    [int(labels[int(i)]) for i in bidx],
                    device,
                )
            elif causal:
                b_texts = [texts[int(i)] for i in bidx]
                # `labels` here are Dolly categories and are NOT a training target; the
                # supervised signal is the response, masked to response tokens only.
                loss = model.lm_loss(b_texts, [self.responses[int(i)] for i in bidx], device)
            else:
                b_texts = [texts[int(i)] for i in bidx]
                b_labels = torch.tensor([labels[int(i)] for i in bidx], device=device)
                loss = loss_fn(model.logits(b_texts, device), b_labels)
            penalty = self.attack.loss_penalty(model, self.client_id, round_idx)
            if isinstance(penalty, torch.Tensor) or penalty:
                loss = loss + penalty
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            seen += 1
        # The round's MEAN, not the last batch: a single batch's loss is noisy enough at
        # batch 8 that a per-round trajectory built from it would mostly show batch luck.
        self.last_train_loss = total / seen if seen else float("nan")
