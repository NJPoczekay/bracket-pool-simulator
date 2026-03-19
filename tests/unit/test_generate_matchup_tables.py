from __future__ import annotations

from pathlib import Path

from bracket_sim.application.generate_matchup_tables import generate_matchup_tables


def test_generate_matchup_tables_defaults_to_round_one(
    prepared_bracket_lab_dir: Path,
) -> None:
    result = generate_matchup_tables(input_dir=prepared_bracket_lab_dir)

    assert result.round_filter == 1
    assert len(result.matchup_rows) == 64
    assert len(result.value_rows) == 64

    first = result.matchup_rows[0]
    assert first.game_id == "g001"
    assert first.game_label == "East Team 1 vs East Team 16"
    assert first.team_id == "east-01"
    assert abs(first.win_probability - 0.8122892350505024) < 1e-12
    assert abs(first.public_pick_rate - 0.826) < 1e-12
    assert abs((first.value or 0.0) - 0.9834010109567826) < 1e-12


def test_generate_matchup_tables_can_include_all_rounds(
    prepared_bracket_lab_dir: Path,
) -> None:
    result = generate_matchup_tables(
        input_dir=prepared_bracket_lab_dir,
        round_filter=None,
    )

    assert result.round_filter is None
    assert len(result.matchup_rows) == 384
    assert len(result.value_rows) == 384
    assert result.value_rows[0].value is not None
    assert result.value_rows[0].value >= result.value_rows[1].value
