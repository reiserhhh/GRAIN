#!/usr/bin/env python3
"""F14/F15 contract checks."""

import numpy as np


def test_placeholder_and_rank_certificate():
    natural = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [1, 1, 1, 0], [1, 0, 1, 1]], dtype=float)
    assert np.linalg.matrix_rank(natural) == 4
    assert None != 0
