from __future__ import annotations

import json
from pathlib import Path

import pytest

from bracket_sim.application.analyze_bracket import (
    BracketLabService,
    _editable_bracket_to_entry,
    _sample_public_opponents,
    _weighted_slot_rating,
)
from bracket_sim.domain.bracket_lab_models import PlayInCandidate, PlayInSlot
from bracket_sim.domain.product_models import (
    AnalyzeBracketRequest,
    BracketEditPick,
    CompletionMode,
    EditableBracket,
    PickDiagnosticTag,
    PoolSettings,
    ScoringSystemKey,
)
from bracket_sim.domain.scoring import validate_entries


def test_weighted_slot_rating_uses_advancement_probabilities() -> None:
    slot = PlayInSlot(
        game_id="g001",
        placeholder_team_id="placeholder-team",
        placeholder_team_name="Team A/Team B",
        seed=11,
        region="East",
        candidates=[
            PlayInCandidate(
                team_id="team-a",
                team_name="Team A",
                rank=1,
                rating=24.0,
                tempo=68.0,
                advancement_probability=0.75,
            ),
            PlayInCandidate(
                team_id="team-b",
                team_name="Team B",
                rank=2,
                rating=12.0,
                tempo=66.0,
                advancement_probability=0.25,
            ),
        ],
    )

    assert _weighted_slot_rating(slot) == 21.0


def test_service_bootstrap_exposes_dataset_and_graph(prepared_bracket_lab_dir: Path) -> None:
    service = BracketLabService(prepared_bracket_lab_dir)

    bootstrap = service.build_bootstrap()

    assert bootstrap.completion_mode == CompletionMode.MANUAL
    assert bootstrap.default_pool_settings.pool_size == 10
    assert len(bootstrap.teams) == 64
    assert len(bootstrap.games) == 63
    assert bootstrap.dataset_hash == service.dataset_hash


def test_sample_public_opponents_is_deterministic_and_legal(
    prepared_bracket_lab_dir: Path,
) -> None:
    service = BracketLabService(prepared_bracket_lab_dir)

    first = _sample_public_opponents(runtime=service._runtime, opponent_count=4, seed=99)
    second = _sample_public_opponents(runtime=service._runtime, opponent_count=4, seed=99)

    assert first == second
    validate_entries(entries=first, graph=service._runtime.graph)


def test_analyze_bracket_is_deterministic_for_fixed_request(
    prepared_bracket_lab_dir: Path,
    synthetic_input_dir: Path,
) -> None:
    service = BracketLabService(prepared_bracket_lab_dir)
    request = AnalyzeBracketRequest(
        bracket=_editable_bracket_from_fixture(synthetic_input_dir),
        pool_settings=PoolSettings(pool_size=25, scoring_system=ScoringSystemKey.LINEAR),
    )

    first = service.analyze_bracket(request)
    second = service.analyze_bracket(request)

    assert first == second
    assert first.public_percentile is None
    assert len(first.pick_diagnostics) == 63


def test_analyze_bracket_cache_key_changes_when_bracket_changes(
    prepared_bracket_lab_dir: Path,
    synthetic_input_dir: Path,
) -> None:
    service = BracketLabService(prepared_bracket_lab_dir)
    bracket = _editable_bracket_from_fixture(synthetic_input_dir)
    original = service.analyze_bracket(
        AnalyzeBracketRequest(
            bracket=bracket,
            pool_settings=PoolSettings(pool_size=15, scoring_system=ScoringSystemKey.ESPN),
        )
    )
    changed_picks = list(bracket.picks)
    changed_picks[-1] = changed_picks[-1].model_copy(
        update={
            "winner_team_id": _alternate_game_winner(changed_picks[-1], synthetic_input_dir),
        }
    )
    changed = service.analyze_bracket(
        AnalyzeBracketRequest(
            bracket=EditableBracket(picks=changed_picks),
            pool_settings=PoolSettings(pool_size=15, scoring_system=ScoringSystemKey.ESPN),
        )
    )

    assert original.cache_key != changed.cache_key


def test_analyze_bracket_rejects_non_manual_completion_mode(
    prepared_bracket_lab_dir: Path,
    synthetic_input_dir: Path,
) -> None:
    service = BracketLabService(prepared_bracket_lab_dir)

    with pytest.raises(ValueError, match="completion_mode='manual'"):
        service.analyze_bracket(
            AnalyzeBracketRequest(
                bracket=_editable_bracket_from_fixture(synthetic_input_dir),
                pool_settings=PoolSettings(pool_size=10, scoring_system=ScoringSystemKey.ESPN),
                completion_mode=CompletionMode.POPULAR_PICKS,
            )
        )


def test_analyze_bracket_produces_expected_pick_tags(
    prepared_bracket_lab_dir: Path,
    synthetic_input_dir: Path,
) -> None:
    service = BracketLabService(prepared_bracket_lab_dir)
    analysis = service.analyze_bracket(
        AnalyzeBracketRequest(
            bracket=_editable_bracket_from_fixture(synthetic_input_dir),
            pool_settings=PoolSettings(
                pool_size=20,
                scoring_system=ScoringSystemKey.ROUND_PLUS_SEED,
            ),
        )
    )

    tag_counts = {
        tag: sum(tag in diagnostic.tags for diagnostic in analysis.pick_diagnostics)
        for tag in PickDiagnosticTag
    }
    assert tag_counts[PickDiagnosticTag.BEST_PICK] == 1
    assert tag_counts[PickDiagnosticTag.WORST_PICK] == 1
    assert tag_counts[PickDiagnosticTag.MOST_IMPORTANT] == 1
    assert all(
        0.0 <= diagnostic.survival_probability <= 1.0
        for diagnostic in analysis.pick_diagnostics
    )


def test_editable_bracket_to_entry_rejects_incomplete_bracket(
    prepared_bracket_lab_dir: Path,
    synthetic_input_dir: Path,
) -> None:
    service = BracketLabService(prepared_bracket_lab_dir)
    bracket = _editable_bracket_from_fixture(synthetic_input_dir)
    incomplete = EditableBracket(
        picks=[
            bracket.picks[0].model_copy(update={"winner_team_id": None}),
            *bracket.picks[1:],
        ]
    )

    with pytest.raises(ValueError, match="Bracket is incomplete"):
        _editable_bracket_to_entry(bracket=incomplete, graph=service._runtime.graph)


def _editable_bracket_from_fixture(synthetic_input_dir: Path) -> EditableBracket:
    entries = json.loads((synthetic_input_dir / "entries.json").read_text(encoding="utf-8"))
    picks = entries[0]["picks"]
    return EditableBracket(
        picks=[
            BracketEditPick(game_id=game_id, winner_team_id=winner_team_id)
            for game_id, winner_team_id in sorted(picks.items())
        ]
    )


def _alternate_game_winner(pick: BracketEditPick, synthetic_input_dir: Path) -> str:
    entries = json.loads((synthetic_input_dir / "entries.json").read_text(encoding="utf-8"))
    games = json.loads((synthetic_input_dir / "games.json").read_text(encoding="utf-8"))
    entry_picks = entries[0]["picks"]
    game = next(game for game in games if game["game_id"] == pick.game_id)
    if game["round"] == 1:
        candidates = [str(game["left_team_id"]), str(game["right_team_id"])]
    else:
        candidates = [
            str(entry_picks[game["left_game_id"]]),
            str(entry_picks[game["right_game_id"]]),
        ]
    winner = next(team_id for team_id in candidates if team_id != pick.winner_team_id)
    return str(winner)
