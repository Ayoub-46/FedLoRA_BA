"""Decorator-based registries.

Extensibility spine: attacks, defenses, probes, and aggregation strategies all
register themselves via ``@registry.register("name")`` and are selected by config
name. Adding a component touches only its own module + a config — never the core
loop (CLAUDE.md, "Extensibility conventions").
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """A named registry of objects (classes or factory callables).

    Example::

        attacks: Registry[type[Attack]] = Registry("attack")

        @attacks.register("x4_backdoor")
        class BackdoorAttack: ...

        cls = attacks.get("x4_backdoor")
    """

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._entries: dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        """Return a decorator that registers the decorated object under ``name``."""

        def deco(obj: T) -> T:
            if name in self._entries:
                raise KeyError(f"{self.kind} '{name}' is already registered")
            self._entries[name] = obj
            return obj

        return deco

    def get(self, name: str) -> T:
        try:
            return self._entries[name]
        except KeyError:
            raise KeyError(
                f"unknown {self.kind} '{name}'. Registered: {sorted(self._entries)}"
            ) from None

    def names(self) -> Iterable[str]:
        return sorted(self._entries)

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def __len__(self) -> int:
        return len(self._entries)
