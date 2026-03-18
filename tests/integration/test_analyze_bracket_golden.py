from __future__ import annotations

import json
from pathlib import Path

from bracket_sim.application.analyze_bracket import BracketLabService
from bracket_sim.domain.product_models import (
    AnalyzeBracketRequest,
    BracketEditPick,
    EditableBracket,
    PoolSettings,
    ScoringSystemKey,
)


def test_analyze_bracket_matches_golden_output(
    prepared_bracket_lab_dir: Path,
    synthetic_input_dir: Path,
) -> None:
    service = BracketLabService(prepared_bracket_lab_dir)
    analysis = service.analyze_bracket(
        AnalyzeBracketRequest(
            bracket=_editable_bracket_from_fixture(synthetic_input_dir),
            pool_settings=PoolSettings(
                pool_size=18,
                scoring_system=ScoringSystemKey.ROUND_PLUS_SEED,
            ),
        )
    )

    expected_path = (
        Path(__file__).resolve().parents[1]
        / "expected"
        / "bracket_lab_analysis_round_plus_seed_pool18.json"
    )
    expected = json.loads(expected_path.read_text(encoding="utf-8"))

    assert analysis.model_dump(mode="json") == expected


def _editable_bracket_from_fixture(synthetic_input_dir: Path) -> EditableBracket:
    entries = json.loads((synthetic_input_dir / "entries.json").read_text(encoding="utf-8"))
    return EditableBracket(
        picks=[
            BracketEditPick(game_id=game_id, winner_team_id=winner_team_id)
            for game_id, winner_team_id in sorted(entries[0]["picks"].items())
        ]
    )
