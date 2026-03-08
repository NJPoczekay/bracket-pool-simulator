from __future__ import annotations

import numpy as np

from bracket_sim.domain.bracket_graph import BracketGraph
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
