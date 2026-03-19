from __future__ import annotations

import json
from pathlib import Path

import pytest

from bracket_sim.application.analyze_bracket import BracketLabService
from bracket_sim.domain.bracket_graph import BracketGraph
from bracket_sim.domain.product_models import (
    AnalyzeBracketRequest,
    BracketEditPick,
    CompletionMode,
    EditableBracket,
    OptimizeBracketRequest,
    PickFourSelection,
    PoolSettings,
    ScoringSystemKey,
)


def test_optimize_bracket_is_deterministic_and_returns_diverse_alternatives(
    prepared_bracket_lab_dir: Path,
    synthetic_input_dir: Path,
) -> None:
    service = BracketLabService(prepared_bracket_lab_dir)
    request = OptimizeBracketRequest(
        bracket=_editable_bracket_from_fixture(synthetic_input_dir, entry_index=0),
        pool_settings=PoolSettings(pool_size=25, scoring_system=ScoringSystemKey.ESPN),
    )

    first = service.optimize_bracket(request)
    second = service.optimize_bracket(request)

    assert first == second
    assert first.cache_key.startswith("optimization-")
    assert first.changed_pick_count == len(first.changed_picks)
    assert first.summary is not None

    region_tuples = [
        _region_winner_tuple(service, first.recommended_bracket),
        *[
            _region_winner_tuple(service, alternative.bracket)
            for alternative in first.alternatives
        ],
    ]
    assert len(region_tuples) == len(set(region_tuples))
    assert all(
        alternative.changed_pick_count == len(alternative.changed_picks)
        for alternative in first.alternatives
    )
    assert all(
        _region_winner_tuple(service, alternative.bracket)
        != _region_winner_tuple(service, first.recommended_bracket)
        for alternative in first.alternatives
    )


def test_optimize_bracket_rejects_incomplete_bracket(
    prepared_bracket_lab_dir: Path,
) -> None:
    service = BracketLabService(prepared_bracket_lab_dir)

    with pytest.raises(ValueError, match="Optimizer requires a complete bracket"):
        service.optimize_bracket(
            OptimizeBracketRequest(
                bracket=EditableBracket(picks=[]),
                pool_settings=PoolSettings(pool_size=25, scoring_system=ScoringSystemKey.ESPN),
            )
        )


def test_optimize_bracket_preserves_locked_paths_and_pick_four_constraints(
    prepared_bracket_lab_dir: Path,
    synthetic_input_dir: Path,
) -> None:
    service = BracketLabService(prepared_bracket_lab_dir)
    bracket = _editable_bracket_from_fixture(synthetic_input_dir, entry_index=0)
    champion_pick = bracket.picks[-1].model_copy(update={"locked": True})
    locked_bracket = EditableBracket(
        picks=[
            champion_pick if pick.game_id == champion_pick.game_id else pick
            for pick in bracket.picks
        ]
    )
    request = OptimizeBracketRequest(
        bracket=locked_bracket,
        pool_settings=PoolSettings(pool_size=25, scoring_system=ScoringSystemKey.ESPN),
        completion_mode=CompletionMode.MANUAL,
        pick_four=PickFourSelection(
            regional_winner_seeds={
                "east": 1,
                "west": 1,
                "south": 1,
                "midwest": 6,
            }
        ),
    )

    result = service.optimize_bracket(request)
    picks_by_game_id = {
        pick.game_id: pick.winner_team_id for pick in result.recommended_bracket.picks
    }
    path_game_ids = _winner_path_game_ids(
        graph=service._runtime.graph,
        game_id=service._runtime.graph.championship_game_id,
        winner_team_id=str(champion_pick.winner_team_id),
    )

    assert all(
        picks_by_game_id[game_id] == champion_pick.winner_team_id
        for game_id in path_game_ids
    )
    assert picks_by_game_id[service._runtime.region_champion_game_ids["east"]] == "east-01"
    assert picks_by_game_id[service._runtime.region_champion_game_ids["west"]] == "west-01"
    assert picks_by_game_id[service._runtime.region_champion_game_ids["south"]] == "south-01"
    assert picks_by_game_id[service._runtime.region_champion_game_ids["midwest"]] == "midwest-06"


@pytest.mark.parametrize(
    "scoring_system",
    [ScoringSystemKey.ESPN, ScoringSystemKey.ROUND_PLUS_SEED],
)
def test_optimize_bracket_improves_over_chalk_baseline(
    prepared_bracket_lab_dir: Path,
    synthetic_input_dir: Path,
    scoring_system: ScoringSystemKey,
) -> None:
    service = BracketLabService(prepared_bracket_lab_dir)
    bracket = _editable_bracket_from_fixture(synthetic_input_dir, entry_index=0)
    analysis = service.analyze_bracket(
        AnalyzeBracketRequest(
            bracket=bracket,
            pool_settings=PoolSettings(pool_size=25, scoring_system=scoring_system),
        )
    )
    optimized = service.optimize_bracket(
        OptimizeBracketRequest(
            bracket=bracket,
            pool_settings=PoolSettings(pool_size=25, scoring_system=scoring_system),
        )
    )

    assert optimized.projected_win_probability > analysis.win_probability
    assert optimized.changed_pick_count > 0


@pytest.mark.parametrize(
    "scoring_system",
    [
        ScoringSystemKey.ROUND_OF_64_FLAT,
        ScoringSystemKey.ROUND_OF_64_SEED,
    ],
)
def test_optimize_bracket_supports_round_of_64_scoring_systems(
    prepared_bracket_lab_dir: Path,
    synthetic_input_dir: Path,
    scoring_system: ScoringSystemKey,
) -> None:
    service = BracketLabService(prepared_bracket_lab_dir)
    bracket = _editable_bracket_from_fixture(synthetic_input_dir, entry_index=0)
    baseline = service.analyze_bracket(
        AnalyzeBracketRequest(
            bracket=bracket,
            pool_settings=PoolSettings(pool_size=25, scoring_system=scoring_system),
        )
    )
    optimized = service.optimize_bracket(
        OptimizeBracketRequest(
            bracket=bracket,
            pool_settings=PoolSettings(pool_size=25, scoring_system=scoring_system),
        )
    )
    optimized_analysis = service.analyze_bracket(
        AnalyzeBracketRequest(
            bracket=optimized.recommended_bracket,
            pool_settings=PoolSettings(pool_size=25, scoring_system=scoring_system),
        )
    )

    assert optimized.cache_key.startswith("optimization-")
    assert optimized.changed_pick_count > 0
    assert optimized_analysis.win_probability >= baseline.win_probability


def _editable_bracket_from_fixture(
    synthetic_input_dir: Path,
    *,
    entry_index: int,
) -> EditableBracket:
    entries = json.loads((synthetic_input_dir / "entries.json").read_text(encoding="utf-8"))
    picks = entries[entry_index]["picks"]
    return EditableBracket(
        picks=[
            BracketEditPick(game_id=game_id, winner_team_id=winner_team_id)
            for game_id, winner_team_id in sorted(picks.items())
        ]
    )


def _region_winner_tuple(service: BracketLabService, bracket: EditableBracket) -> tuple[str, ...]:
    picks_by_game_id = {pick.game_id: pick.winner_team_id or "" for pick in bracket.picks}
    return tuple(
        picks_by_game_id[game_id]
        for _, game_id in sorted(service._runtime.region_champion_game_ids.items())
    )


def _winner_path_game_ids(
    *,
    graph: BracketGraph,
    game_id: str,
    winner_team_id: str,
) -> list[str]:
    game = graph.games_by_id[game_id]
    if game.round == 1:
        return [game_id]

    left_child_id, right_child_id = graph.children_by_game_id[game_id]
    if winner_team_id in graph.possible_teams_by_game_id[left_child_id]:
        return _winner_path_game_ids(
            graph=graph,
            game_id=left_child_id,
            winner_team_id=winner_team_id,
        ) + [game_id]

    return _winner_path_game_ids(
        graph=graph,
        game_id=right_child_id,
        winner_team_id=winner_team_id,
    ) + [game_id]
