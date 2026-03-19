"""Shared scoring-system identifiers and score-shaping metadata."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

ESPN_ROUND_VALUES = (10, 20, 40, 80, 160, 320)
NO_SEED_BONUS_ROUNDS = (False, False, False, False, False, False)


class ScoringSystemKey(StrEnum):
    """Supported scoring system identifiers."""

    ESPN = "1-2-4-8-16-32"
    LINEAR = "1-2-3-4-5-6"
    FIBONACCI = "2-3-5-8-13-21"
    ROUND_PLUS_SEED = "round+seed"
    ROUND_OF_64_FLAT = "round-of-64-flat"
    ROUND_OF_64_SEED = "round-of-64-seed"


@dataclass(frozen=True)
class ScoringSpec:
    """Resolved round and seed-bonus scoring behavior for one pool."""

    round_values: tuple[int, int, int, int, int, int]
    seed_bonus_rounds: tuple[bool, bool, bool, bool, bool, bool] = NO_SEED_BONUS_ROUNDS


_SCORING_SPECS: dict[ScoringSystemKey, ScoringSpec] = {
    ScoringSystemKey.ESPN: ScoringSpec(round_values=ESPN_ROUND_VALUES),
    ScoringSystemKey.LINEAR: ScoringSpec(round_values=(1, 2, 3, 4, 5, 6)),
    ScoringSystemKey.FIBONACCI: ScoringSpec(round_values=(2, 3, 5, 8, 13, 21)),
    ScoringSystemKey.ROUND_PLUS_SEED: ScoringSpec(
        round_values=(1, 2, 3, 4, 5, 6),
        seed_bonus_rounds=(True, True, True, True, True, True),
    ),
    ScoringSystemKey.ROUND_OF_64_FLAT: ScoringSpec(round_values=(1, 0, 0, 0, 0, 0)),
    ScoringSystemKey.ROUND_OF_64_SEED: ScoringSpec(
        round_values=(0, 0, 0, 0, 0, 0),
        seed_bonus_rounds=(True, False, False, False, False, False),
    ),
}


def resolve_scoring_spec(scoring_system: ScoringSystemKey) -> ScoringSpec:
    """Return the round and seed-bonus rules for one scoring-system key."""

    return _SCORING_SPECS[scoring_system]
