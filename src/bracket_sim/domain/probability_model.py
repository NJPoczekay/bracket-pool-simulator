"""Win-probability model utilities."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def logistic_win_probability(
    left_ratings: npt.NDArray[np.float64],
    right_ratings: npt.NDArray[np.float64],
    rating_scale: float,
) -> npt.NDArray[np.float64]:
    """Return P(left wins) via logistic transform of rating differences."""

    deltas = (left_ratings - right_ratings) / rating_scale
    return 1.0 / (1.0 + np.exp(-deltas))
