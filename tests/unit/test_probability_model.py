from __future__ import annotations

import numpy as np

from bracket_sim.domain.probability_model import expected_point_differential, kenpom_win_probability


def test_expected_point_differential_matches_reddit_formula() -> None:
    point_diff = expected_point_differential(
        left_ratings=np.array([24.31], dtype=np.float64),
        right_ratings=np.array([15.83], dtype=np.float64),
        left_tempos=np.array([69.5], dtype=np.float64),
        right_tempos=np.array([65.8], dtype=np.float64),
    )

    assert abs(point_diff[0] - 5.73672) < 1e-9


def test_kenpom_win_probability_matches_reddit_example() -> None:
    probability = kenpom_win_probability(
        left_ratings=np.array([24.31], dtype=np.float64),
        right_ratings=np.array([15.83], dtype=np.float64),
        left_tempos=np.array([69.5], dtype=np.float64),
        right_tempos=np.array([65.8], dtype=np.float64),
        point_spread_std_dev=11.0,
    )

    assert abs(probability[0] - 0.699) < 0.002
