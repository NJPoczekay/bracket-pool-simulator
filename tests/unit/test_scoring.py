from __future__ import annotations

import numpy as np
import pytest

from bracket_sim.domain.bracket_graph import BracketGraph
from bracket_sim.domain.models import EntryPick
from bracket_sim.domain.scoring import (
    aggregate_win_shares,
    build_predicted_wins_matrix,
    score_entries,
    validate_entries,
)
from bracket_sim.infrastructure.storage.normalized_loader import NormalizedInput


def test_score_entries_matches_expected_toy_values() -> None:
    predicted = np.array([[2, 0], [1, 1]], dtype=np.int16)
    actual = np.array([[2, 0], [0, 2]], dtype=np.int16)

    scores = score_entries(predicted_wins=predicted, actual_wins=actual)
    expected = np.array([[3, 0], [1, 1]], dtype=np.int32)

    assert np.array_equal(scores, expected)


def test_score_entries_respects_exponential_round_values() -> None:
    predicted = np.array([[6], [3], [0]], dtype=np.int16)
    actual = np.array([[6], [3], [0]], dtype=np.int16)

    scores = score_entries(predicted_wins=predicted, actual_wins=actual)
    expected = np.array(
        [
            [63, 7, 0],
            [7, 7, 0],
            [0, 0, 0],
        ],
        dtype=np.int32,
    )

    assert np.array_equal(scores, expected)


def test_score_entries_supports_linear_round_values() -> None:
    predicted = np.array([[6], [3], [0]], dtype=np.int16)
    actual = np.array([[6], [3], [0]], dtype=np.int16)

    scores = score_entries(
        predicted_wins=predicted,
        actual_wins=actual,
        round_values=(1, 2, 3, 4, 5, 6),
    )

    expected = np.array(
        [
            [21, 6, 0],
            [6, 6, 0],
            [0, 0, 0],
        ],
        dtype=np.int32,
    )
    assert np.array_equal(scores, expected)


def test_score_entries_supports_round_plus_seed_bonus() -> None:
    predicted = np.array([[2, 1]], dtype=np.int16)
    actual = np.array([[2, 1]], dtype=np.int16)
    team_seeds = np.array([12, 4], dtype=np.int16)

    scores = score_entries(
        predicted_wins=predicted,
        actual_wins=actual,
        round_values=(1, 2, 3, 4, 5, 6),
        team_seeds=team_seeds,
        seed_bonus=True,
    )

    assert np.array_equal(scores, np.array([[32]], dtype=np.int32))


def test_tie_split_aggregation() -> None:
    scores = np.array(
        [
            [10, 8, 7],
            [10, 8, 7],
            [5, 9, 7],
        ],
        dtype=np.int32,
    )

    shares = aggregate_win_shares(scores)
    assert np.allclose(shares, np.array([0.27777778, 0.27777778, 0.44444444]), atol=1e-8)


def test_entry_validation_and_predicted_wins_matrix(
    normalized_input: NormalizedInput,
    graph: BracketGraph,
) -> None:
    validate_entries(entries=normalized_input.entries, graph=graph)

    entry_ids, team_ids, predicted_wins = build_predicted_wins_matrix(
        entries=normalized_input.entries,
        graph=graph,
    )

    assert len(entry_ids) == len(normalized_input.entries)
    assert len(team_ids) == 64
    assert predicted_wins.shape == (len(normalized_input.entries), 64)
    assert int(np.max(predicted_wins)) <= 6


def test_entry_validation_rejects_inconsistent_upstream_pick(
    normalized_input: NormalizedInput,
    graph: BracketGraph,
) -> None:
    entry = normalized_input.entries[0]
    pick_map = {pick.game_id: pick.winner_team_id for pick in entry.picks}

    round_two_game_id = sorted(
        game.game_id for game in graph.games_by_id.values() if game.round == 2
    )[0]
    left_child_id, right_child_id = graph.children_by_game_id[round_two_game_id]
    upstream_winners = {pick_map[left_child_id], pick_map[right_child_id]}
    invalid_options = sorted(graph.possible_teams_by_game_id[round_two_game_id] - upstream_winners)
    assert invalid_options

    pick_map[round_two_game_id] = invalid_options[0]
    invalid_entry = entry.model_copy(
        update={
            "picks": [
                EntryPick(game_id=game_id, winner_team_id=winner_team_id)
                for game_id, winner_team_id in sorted(pick_map.items())
            ]
        }
    )
    entries = [invalid_entry, *normalized_input.entries[1:]]

    with pytest.raises(ValueError, match="inconsistent with its own upstream picks"):
        validate_entries(entries=entries, graph=graph)
