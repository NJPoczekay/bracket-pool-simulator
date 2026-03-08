from __future__ import annotations

from pathlib import Path

import pytest

from bracket_sim.domain.bracket_graph import build_bracket_graph
from bracket_sim.domain.models import Team
from bracket_sim.infrastructure.storage.normalized_loader import (
    NormalizedInput,
    load_normalized_input,
)


def test_valid_graph_has_expected_shape(synthetic_input_dir: Path) -> None:
    normalized = load_normalized_input(synthetic_input_dir)
    graph = build_bracket_graph(teams=normalized.teams, games=normalized.games)

    assert len(graph.games_by_id) == 63
    assert len(graph.topological_game_ids) == 63
    assert graph.games_by_id[graph.championship_game_id].round == 6


def test_duplicate_game_id_rejected(normalized_input: NormalizedInput) -> None:
    duplicate_games = list(normalized_input.games)
    duplicate_games[1] = duplicate_games[1].model_copy(
        update={"game_id": duplicate_games[0].game_id}
    )

    with pytest.raises(ValueError, match="Duplicate game id"):
        build_bracket_graph(teams=normalized_input.teams, games=duplicate_games)


def test_cycle_rejected(normalized_input: NormalizedInput) -> None:
    games = list(normalized_input.games)
    replacement = list(games)

    round_two_index = next(idx for idx, game in enumerate(replacement) if game.round == 2)
    round_two_game = replacement[round_two_index]
    replacement[round_two_index] = round_two_game.model_copy(
        update={
            "left_game_id": round_two_game.game_id,
            "right_game_id": round_two_game.right_game_id,
        }
    )

    with pytest.raises(ValueError):
        build_bracket_graph(teams=normalized_input.teams, games=replacement)


def test_unknown_round_one_team_rejected(normalized_input: NormalizedInput) -> None:
    games = list(normalized_input.games)
    first = games[0].model_copy(update={"left_team_id": "missing-team"})
    mutated = [first, *games[1:]]

    with pytest.raises(ValueError, match="unknown team"):
        build_bracket_graph(teams=normalized_input.teams, games=mutated)


def test_non_64_team_bracket_rejected(normalized_input: NormalizedInput) -> None:
    reduced_teams: list[Team] = list(normalized_input.teams[:-1])
    with pytest.raises(ValueError, match="64 teams"):
        build_bracket_graph(teams=reduced_teams, games=normalized_input.games)
