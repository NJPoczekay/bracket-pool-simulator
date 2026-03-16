from __future__ import annotations

import pytest
from pydantic import ValidationError

from bracket_sim.domain.product_models import (
    BracketEditPick,
    EditableBracket,
    PoolSettings,
    ScoringSystem,
    ScoringSystemKey,
)


def test_bracket_edit_pick_requires_winner_when_locked() -> None:
    with pytest.raises(ValidationError, match="locked picks require winner_team_id"):
        BracketEditPick(game_id="g001", locked=True)


def test_editable_bracket_rejects_duplicate_game_ids() -> None:
    with pytest.raises(ValidationError, match="Duplicate editable pick"):
        EditableBracket(
            picks=[
                BracketEditPick(game_id="g001", winner_team_id="team-a"),
                BracketEditPick(game_id="g001", winner_team_id="team-b"),
            ]
        )


def test_pool_settings_requires_positive_pool_size() -> None:
    with pytest.raises(ValidationError):
        PoolSettings(pool_size=0)


def test_scoring_system_requires_positive_round_values() -> None:
    with pytest.raises(ValidationError, match="round_values must all be positive"):
        ScoringSystem(
            key=ScoringSystemKey.ESPN,
            label="Broken",
            round_values=(1, 2, 4, 8, 0, 32),
            description="invalid",
        )
