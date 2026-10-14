#!/usr/bin/env python3
"""Author-local quadratic assignment expectations, not common vector-axis tests."""

from itertools import permutations

import numpy as np
import pytest

from grain_research.construction import math_criterion_components as C
from grain_research.construction import math_output_observability as M
from scripts.analyze_grain_output_author_local_hm284 import diagonal_mean


def test_author_local_orbit_mean_matches_exhaustive_assignment():
    y, groups = np.array([0, 0, 1, 1]), np.zeros(4, int)
    a = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [2, 1, 0, 0]], float)
    b = np.array([[0, 1, 1, 0], [1, 0, 0, 1], [0, 1, 2, 0]], float)
    diagonal = np.array([2, -1, 3], float)
    observed = []
    for ix in permutations(range(4)):
        e = y[list(ix)] - 0.5
        observed.append(
            np.mean(M.centered_scores(a @ e) * M.centered_scores(b @ e) * diagonal)
        )
    assert diagonal_mean(
        a, b, C.orbit_covariance(y, groups), diagonal
    ) == pytest.approx(np.mean(observed), abs=1e-14)
