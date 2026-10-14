#!/usr/bin/env python3
"""Independent HM150 formula readback without model or data dependencies."""

import numpy as np

from grain_research.construction.math_semantic_anchor import tensor_features
from scripts.verify_grain_joint_persona_style_hm150 import (
    seed_identity,
    tensor_by_coordinates,
)


def test_coordinate_product_matches_existing_tensor_order():
    values = np.random.default_rng(2).normal(size=(7, 4, 4))
    np.testing.assert_allclose(tensor_by_coordinates(values), tensor_features(values))


def test_explicit_seed_pairs_preserve_negative_support_and_identity():
    support, mean, spread = seed_identity(np.array([[1.0, 0], [-1.0, 0], [0.0, 0]]))
    assert support == -1 / 3
    assert mean == 0
    np.testing.assert_allclose(mean, support + spread)
