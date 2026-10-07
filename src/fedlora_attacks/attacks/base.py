"""Attack interface + registry (CLAUDE.md "attacks/", plan §3 X1-X4 / v3 §1.2).

An attack can act at several points in a client's round; each hook defaults to a
no-op so a concrete attack overrides only what it needs:
  - ``poison_dataset``   : data-space poisoning (X4 backdoor).
  - ``loss_penalty``     : an added local-objective term (X2 low-rank steering).
  - ``transform_update`` : post-training GL(r) transform of (A, B) (Channel A / B).

PURITY CONTRACT (plan v3 Phase 1.2). Every attack declares the hooks it uses in
``HOOKS`` and the round state it reads in ``SEES``, and tests/test_attack_contract.py
enforces both. This is what makes v3's central claim — "the coalition trains honestly
and only re-expresses what it transmits" — checkable rather than asserted. An
update-space-only attack declares ``HOOKS = {"transform_update"}`` and must leave
``poison_dataset`` and ``loss_penalty`` at the ``BaseAttack`` identity, so no poisoned
data and no modified objective can enter through a side door.

The run-level bit-identity claim only holds at the injection round (from round 2 the
attacked run's global has moved, so coalition updates legitimately diverge from the
benign twin's). The unit test is what holds every round.

Persistence (plan §3 outcome variable) is driven by ``attack_stop_round``: a
malicious client injects only while ``is_active`` is True; afterwards it trains
benignly and we watch ASR decay. ``apply_trigger``/``target_label`` let the metrics
layer build the triggered ASR evaluation set. v3 attacks define no target label, so
ASR is None and persistence is inert for them without any code change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import torch

from ..fl.types import AdapterDict, ClientUpdate
from ..utils.registry import Registry

#: The hooks an attack may override. ``HOOKS`` declarations are checked against these.
ATTACK_HOOKS = ("poison_dataset", "loss_penalty", "transform_update")

#: The round state an attack may read, i.e. the fields of ``AttackContext``. Everything
#: here is threat-model-legal (plan v3 §1.2): the adversary knows the broadcast global
#: model and its own clients' state. Benign clients' data and current-round updates are
#: NOT here and never will be — that is server compromise, an explicit non-capability.
ATTACK_VISIBLE = ("global_adapters", "round_idx", "client_id")


@dataclass
class AttackContext:
    """What a client's attack hooks may see at ``transform_update`` time.

    Deliberately narrow. ``global_adapters`` is the model BROADCAST at the start of this
    round — legal under §1.2 ("knows the previous global model, it is broadcast") and
    required by Channel A, whose attacker estimates other clients' factors from it.

    What is absent is the point: no other client's update, no other client's data. One
    ``Attack`` instance is shared by every client and clients train sequentially, so an
    attack COULD cache a benign client's update as it goes past and read it when a later
    coalition member runs. That would be server compromise. ``SEES`` declares the
    intent and tests/test_attack_contract.py tests the behaviour, by perturbing a
    non-coalition client's data and asserting the coalition's transmitted factors do not
    move at the injection round.
    """

    round_idx: int
    client_id: int
    global_adapters: AdapterDict
    rng: np.random.Generator


@dataclass
class AttackConfig:
    name: str = "none"
    malicious_client_ids: tuple[int, ...] = ()
    attack_start_round: int = 0
    """First round the coalition injects. 0 (the default) reproduces the pre-E-A3
    behaviour exactly, so every earlier result is unaffected.

    With ``attack_stop_round`` this makes the injection WINDOW expressible, which E-A3
    Stage A needs: a coalition that joins an ongoing federation is both the weaker threat
    model and the one a reviewer expects, and holding the window length fixed while
    varying its start is what separates "damage per round of attack" from "more rounds of
    attack".
    """
    attack_stop_round: int | None = None  # inject while round < stop (None = never stop)
    active_rounds: tuple[int, ...] | None = None
    """Optional explicit zero-based rounds on which the attack is active.

    ``None`` preserves the historical contiguous start/stop window. A tuple intersects
    that window with a pre-registered sporadic schedule. The schedule is shared by the
    coalition because it belongs to the attack instance, not to individual clients.
    """

    def __post_init__(self) -> None:
        if self.active_rounds is None:
            return
        rounds = tuple(int(round_idx) for round_idx in self.active_rounds)
        if len(rounds) != len(set(rounds)):
            raise ValueError("active_rounds contains duplicate rounds")
        if any(round_idx < 0 for round_idx in rounds):
            raise ValueError("active_rounds must contain non-negative round indices")
        self.active_rounds = tuple(sorted(rounds))


@runtime_checkable
class Attack(Protocol):
    config: AttackConfig
    HOOKS: frozenset[str]
    SEES: frozenset[str]

    def is_malicious(self, client_id: int) -> bool: ...
    def is_active(self, client_id: int, round_idx: int) -> bool: ...
    def poison_dataset(
        self,
        texts: list[str],
        labels: list[int],
        client_id: int,
        round_idx: int,
        rng: np.random.Generator,
    ) -> tuple[list[str], list[int]]: ...
    def loss_penalty(self, model, client_id: int, round_idx: int) -> torch.Tensor | float: ...
    def transform_update(self, update: ClientUpdate, ctx: AttackContext) -> ClientUpdate: ...
    def apply_trigger(self, text: str) -> str: ...
    @property
    def target_label(self) -> int | None: ...


ATTACKS: Registry[type[Attack]] = Registry("attack")


class BaseAttack:
    """Default no-op implementation; concrete attacks subclass and override hooks.

    Subclasses MUST declare ``HOOKS`` (which of ``ATTACK_HOOKS`` they override) and, if
    they override ``transform_update``, ``SEES`` (which ``AttackContext`` fields they
    read). Both are enforced by tests/test_attack_contract.py in both directions: an
    undeclared override fails, and a declared-but-unused hook fails.
    """

    HOOKS: frozenset[str] = frozenset()
    SEES: frozenset[str] = frozenset()

    def __init__(self, config: AttackConfig) -> None:
        self.config = config

    def is_malicious(self, client_id: int) -> bool:
        return client_id in self.config.malicious_client_ids

    def is_active(self, client_id: int, round_idx: int) -> bool:
        if not self.is_malicious(client_id):
            return False
        if round_idx < self.config.attack_start_round:
            return False
        stop = self.config.attack_stop_round
        if stop is not None and round_idx >= stop:
            return False
        rounds = self.config.active_rounds
        return rounds is None or round_idx in rounds

    def poison_dataset(self, texts, labels, client_id, round_idx, rng):
        return texts, labels

    def loss_penalty(self, model, client_id, round_idx):
        return 0.0

    def transform_update(self, update: ClientUpdate, ctx: AttackContext) -> ClientUpdate:
        return update

    def apply_trigger(self, text: str) -> str:
        return text

    @property
    def target_label(self) -> int | None:
        return None

    # tokens the tokenizer must reserve (e.g. the trigger); empty by default
    def extra_vocab(self) -> list[str]:
        return []


@ATTACKS.register("none")
class NoAttack(BaseAttack):
    """Benign baseline: no poisoning, no transform, no ASR target."""

    HOOKS = frozenset()
    SEES = frozenset()

    def is_active(self, client_id: int, round_idx: int) -> bool:
        return False


def build_attack(config: AttackConfig) -> Attack:
    return ATTACKS.get(config.name)(config)
