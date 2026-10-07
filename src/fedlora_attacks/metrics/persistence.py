"""Behavioral persistence (plan §3 / Lifespan notion): ASR after the attacker stops.

The persistence curve is the ASR time series with the attacker's stop round marked.
``persistence_half_life`` reports how many post-stop rounds until ASR decays to half
of its value at the stop round — the headline persistence statistic E2 studies.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PersistenceCurve:
    rounds: list[int]
    asr: list[float]
    stop_round: int | None

    def post_stop(self) -> tuple[list[int], list[float]]:
        if self.stop_round is None:
            return [], []
        zipped = zip(self.rounds, self.asr, strict=False)
        pairs = [(r, a) for r, a in zipped if r >= self.stop_round]
        rs = [r for r, _ in pairs]
        as_ = [a for _, a in pairs]
        return rs, as_

    def asr_at_stop(self) -> float | None:
        rs, as_ = self.post_stop()
        return as_[0] if as_ else None

    def half_life(self) -> float | None:
        """Rounds after stop until ASR first falls to <= 0.5 * ASR-at-stop.

        None if the attacker never stops, no post-stop data, or ASR never halves
        within the observed horizon.
        """
        rs, as_ = self.post_stop()
        if not as_:
            return None
        base = as_[0]
        if base <= 0:
            return 0.0
        threshold = 0.5 * base
        for r, a in zip(rs, as_, strict=False):
            if a <= threshold:
                return float(r - rs[0])
        return None  # did not halve within the horizon

    def to_dict(self) -> dict:
        return {
            "rounds": self.rounds,
            "asr": self.asr,
            "stop_round": self.stop_round,
            "asr_at_stop": self.asr_at_stop(),
            "half_life": self.half_life(),
        }
