"""KenPom-style win-probability model utilities."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

_POINT_DIFF_POSSESSION_SCALE = 200.0
_NORMAL_CDF_P = 0.2316419
_NORMAL_CDF_B1 = 0.319381530
_NORMAL_CDF_B2 = -0.356563782
_NORMAL_CDF_B3 = 1.781477937
_NORMAL_CDF_B4 = -1.821255978
_NORMAL_CDF_B5 = 1.330274429
_INV_SQRT_2PI = 0.3989422804014327


def expected_point_differential(
    left_ratings: npt.NDArray[np.float64],
    right_ratings: npt.NDArray[np.float64],
    left_tempos: npt.NDArray[np.float64],
    right_tempos: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Return the expected scoring margin for the left team."""

    return (left_ratings - right_ratings) * (left_tempos + right_tempos) / (
        _POINT_DIFF_POSSESSION_SCALE
    )


def standard_normal_cdf(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Approximate the standard normal CDF with a vectorized polynomial."""

    abs_values = np.abs(values)
    t = 1.0 / (1.0 + (_NORMAL_CDF_P * abs_values))
    polynomial = (
        ((((_NORMAL_CDF_B5 * t) + _NORMAL_CDF_B4) * t + _NORMAL_CDF_B3) * t + _NORMAL_CDF_B2)
        * t
        + _NORMAL_CDF_B1
    ) * t
    positive_cdf = 1.0 - (_INV_SQRT_2PI * np.exp(-0.5 * abs_values * abs_values) * polynomial)
    return np.where(values >= 0.0, positive_cdf, 1.0 - positive_cdf)


def kenpom_win_probability(
    left_ratings: npt.NDArray[np.float64],
    right_ratings: npt.NDArray[np.float64],
    left_tempos: npt.NDArray[np.float64],
    right_tempos: npt.NDArray[np.float64],
    point_spread_std_dev: float,
) -> npt.NDArray[np.float64]:
    """Return P(left wins) from tempo-adjusted KenPom ratings."""

    point_differentials = expected_point_differential(
        left_ratings=left_ratings,
        right_ratings=right_ratings,
        left_tempos=left_tempos,
        right_tempos=right_tempos,
    )
    return standard_normal_cdf(point_differentials / point_spread_std_dev)
