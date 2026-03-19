from __future__ import annotations

import json
from pathlib import Path

import pytest

from bracket_sim.application.analyze_bracket import BracketLabService
from bracket_sim.application.bracket_completion import (
    canonicalize_bracket,
    classify_bracket_state,
    editable_bracket_to_entry,
    select_game_winner,
)
from bracket_sim.domain.bracket_graph import BracketGraph
from bracket_sim.domain.models import Game, RatingRecord, Team
from bracket_sim.domain.product_models import (
    AnalyzeBracketRequest,
    BracketCompletionState,
    BracketEditPick,
    CompleteBracketRequest,
    CompletionMode,
    EditableBracket,
    PickFourSelection,
    PoolSettings,
)


def test_canonicalize_bracket_expands_sparse_drafts(
    prepared_bracket_lab_dir: Path,
    synthetic_input_dir: Path,
) -> None:
    service = BracketLabService(prepared_bracket_lab_dir)
    fixture = _editable_bracket_from_fixture(synthetic_input_dir)
    sparse = EditableBracket(picks=fixture.picks[:3])

    canonical = canonicalize_bracket(
        bracket=sparse,
        graph=service._runtime.graph,
        constraints_by_game_id=service._runtime.constraints_by_game_id,
    )

    assert len(canonical.picks) == 63
    assert canonical.picks[0].winner_team_id == fixture.picks[0].winner_team_id
    assert sum(pick.winner_team_id is None for pick in canonical.picks) == 60


def test_classify_bracket_state_distinguishes_manual_and_auto_completed() -> None:
    incomplete = EditableBracket(picks=[BracketEditPick(game_id="g001")])
    complete = EditableBracket(
        picks=[BracketEditPick(game_id="g001", winner_team_id="team-a", locked=True)]
    )
    auto_completed = EditableBracket(
        picks=[BracketEditPick(game_id="g001", winner_team_id="team-a", locked=False)]
    )

    assert classify_bracket_state(incomplete) == BracketCompletionState.INCOMPLETE
    assert classify_bracket_state(complete) == BracketCompletionState.COMPLETE
    assert classify_bracket_state(auto_completed) == BracketCompletionState.AUTO_COMPLETED


def test_complete_bracket_preserves_locked_picks_and_fills_remaining_games(
    prepared_bracket_lab_dir: Path,
    synthetic_input_dir: Path,
) -> None:
    service = BracketLabService(prepared_bracket_lab_dir)
    fixture = _editable_bracket_from_fixture(synthetic_input_dir)
    request = CompleteBracketRequest(
        bracket=EditableBracket(
            picks=[
                fixture.picks[0].model_copy(update={"locked": True}),
                fixture.picks[1].model_copy(update={"locked": True}),
            ]
        ),
        completion_mode=CompletionMode.TOURNAMENT_SEEDS,
    )

    result = service.complete_bracket(request)

    assert result.state == BracketCompletionState.AUTO_COMPLETED
    assert result.preserved_locked_pick_count == 2
    assert result.auto_filled_pick_count == 61
    assert result.completed_bracket.picks[0].locked is True
    assert result.completed_bracket.picks[1].locked is True
    editable_bracket_to_entry(bracket=result.completed_bracket, graph=service._runtime.graph)


def test_complete_bracket_propagates_locked_late_round_winner(
    prepared_bracket_lab_dir: Path,
    synthetic_input_dir: Path,
) -> None:
    service = BracketLabService(prepared_bracket_lab_dir)
    fixture = _editable_bracket_from_fixture(synthetic_input_dir)
    champion_pick = fixture.picks[-1].model_copy(update={"locked": True})
    path_game_ids = _winner_path_game_ids(
        graph=service._runtime.graph,
        game_id=service._runtime.graph.championship_game_id,
        winner_team_id=str(champion_pick.winner_team_id),
    )

    result = service.complete_bracket(
        CompleteBracketRequest(
            bracket=EditableBracket(picks=[champion_pick]),
            completion_mode=CompletionMode.POPULAR_PICKS,
        )
    )
    picks_by_game_id = {
        pick.game_id: pick.winner_team_id for pick in result.completed_bracket.picks
    }

    assert all(
        picks_by_game_id[game_id] == champion_pick.winner_team_id
        for game_id in path_game_ids
    )


def test_complete_bracket_rejects_conflicting_pick_four_seed(
    prepared_bracket_lab_dir: Path,
    synthetic_input_dir: Path,
) -> None:
    service = BracketLabService(prepared_bracket_lab_dir)
    fixture = _editable_bracket_from_fixture(synthetic_input_dir)
    east_regional_final = service._runtime.region_champion_game_ids["east"]
    locked_pick = next(
        pick
        for pick in fixture.picks
        if pick.game_id == east_regional_final and pick.winner_team_id == "east-01"
    ).model_copy(update={"locked": True})

    with pytest.raises(ValueError, match="Conflicting forced winners"):
        service.complete_bracket(
            CompleteBracketRequest(
                bracket=EditableBracket(picks=[locked_pick]),
                completion_mode=CompletionMode.KENPOM,
                pick_four=PickFourSelection(
                    regional_winner_seeds={
                        "east": 2,
                        "west": 1,
                        "south": 1,
                        "midwest": 1,
                    }
                ),
            )
        )


@pytest.mark.parametrize(
    ("mode", "expected_runtime_mode"),
    [
        (CompletionMode.TOURNAMENT_SEEDS, CompletionMode.TOURNAMENT_SEEDS),
        (CompletionMode.POPULAR_PICKS, CompletionMode.POPULAR_PICKS),
        (CompletionMode.KENPOM, CompletionMode.KENPOM),
        (CompletionMode.INTERNAL_MODEL_RANK, CompletionMode.KENPOM),
    ],
)
def test_complete_bracket_is_deterministic_for_each_mode(
    prepared_bracket_lab_dir: Path,
    mode: CompletionMode,
    expected_runtime_mode: CompletionMode,
) -> None:
    service = BracketLabService(prepared_bracket_lab_dir)
    request = CompleteBracketRequest(
        bracket=EditableBracket(picks=[]),
        completion_mode=mode,
    )

    first = service.complete_bracket(request)
    second = service.complete_bracket(request)

    assert first == second
    assert first.completion_mode == mode
    if mode == CompletionMode.INTERNAL_MODEL_RANK:
        kenpom = service.complete_bracket(
            request.model_copy(update={"completion_mode": CompletionMode.KENPOM})
        )
        assert first.completed_bracket == kenpom.completed_bracket
        assert expected_runtime_mode == CompletionMode.KENPOM


def test_select_game_winner_uses_weighted_placeholder_ratings_for_kenpom() -> None:
    graph = BracketGraph(
        teams_by_id={
            "placeholder-east-11": Team(
                team_id="placeholder-east-11",
                name="Team A/Team B",
                seed=11,
                region="east",
            ),
            "east-06": Team(
                team_id="east-06",
                name="East Team 6",
                seed=6,
                region="east",
            ),
        },
        games_by_id={
            "g001": Game(
                game_id="g001",
                round=1,
                left_team_id="placeholder-east-11",
                right_team_id="east-06",
            )
        },
        topological_game_ids=["g001"],
        championship_game_id="g001",
        parents_by_game_id={"g001": []},
        children_by_game_id={"g001": []},
        possible_teams_by_game_id={"g001": {"placeholder-east-11", "east-06"}},
    )

    winner = select_game_winner(
        game_id="g001",
        graph=graph,
        winners_by_game_id={"g001": None},
        completion_mode=CompletionMode.KENPOM,
        public_pick_weights_by_game={},
        rating_records_by_team_id={
            "placeholder-east-11": RatingRecord(
                team_id="placeholder-east-11",
                rating=23.0,
                tempo=68.0,
            ),
            "east-06": RatingRecord(team_id="east-06", rating=20.0, tempo=67.0),
        },
        team_rank_by_team_id={"placeholder-east-11": 18, "east-06": 22},
    )

    assert winner == "placeholder-east-11"


def test_analyze_bracket_accepts_completed_auto_filled_brackets(
    prepared_bracket_lab_dir: Path,
) -> None:
    service = BracketLabService(prepared_bracket_lab_dir)
    completed = service.complete_bracket(
        CompleteBracketRequest(
            bracket=EditableBracket(picks=[]),
            completion_mode=CompletionMode.KENPOM,
        )
    )

    analysis = service.analyze_bracket(
        AnalyzeBracketRequest(
            bracket=completed.completed_bracket,
            pool_settings=PoolSettings(pool_size=12),
            completion_mode=CompletionMode.KENPOM,
        )
    )

    assert analysis.completion_mode == CompletionMode.KENPOM
    assert len(analysis.pick_diagnostics) == 63


def _editable_bracket_from_fixture(synthetic_input_dir: Path) -> EditableBracket:
    entries = json.loads((synthetic_input_dir / "entries.json").read_text(encoding="utf-8"))
    picks = entries[0]["picks"]
    return EditableBracket(
        picks=[
            BracketEditPick(game_id=game_id, winner_team_id=winner_team_id)
            for game_id, winner_team_id in sorted(picks.items())
        ]
    )


def _winner_path_game_ids(*, graph: BracketGraph, game_id: str, winner_team_id: str) -> list[str]:
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
