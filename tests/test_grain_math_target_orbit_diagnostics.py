#!/usr/bin/env python3
"""Synthetic checks for the no-forward HM10B2 diagnostic companion."""

import numpy as np
import pytest
from scipy.linalg import hadamard

from scripts.diagnose_grain_math_target_orbits import (
    anchored,
    component,
    family_intervals,
    score,
    term_name,
)


def test_address_terms_partition_without_data_selection():
    """All 31 nonconstant characters belong to exactly one declared group."""
    groups = [component(t) for t in range(1, 32)]
    assert groups.count("main") == 5
    assert groups.count("matched_owner_binding") == 2
    assert groups.count("other_order_interaction") == 15
    assert groups.count("other_nonorder_interaction") == 9
    assert term_name(5) == "A:U"
    assert term_name(10) == "T:V"
    with pytest.raises(ValueError, match="constant"):
        component(0)


def test_full_completion_retains_anchor_and_all_known_displacements():
    """Constant gauge terms cannot alter the cold anchor."""
    rng = np.random.default_rng(33)
    coeff = rng.normal(size=(32, 3))
    source = hadamard(32) @ coeff
    cold = source + np.array([0.4, -0.3, 0.8])
    recovered = anchored(cold[0], coeff, list(range(1, 32)))
    np.testing.assert_allclose(recovered, cold, atol=1e-13)
    np.testing.assert_allclose(recovered[0], cold[0], atol=1e-13)
    np.testing.assert_allclose(
        score(cold + 12, np.arange(3)), score(cold, np.arange(3))
    )


def test_secondary_family_interval_can_find_zero_or_nonzero_interaction():
    """Known finite worlds verify uncertainty mechanics, not human measurement."""
    coeff = np.zeros((32, 2))
    coeff[1] = [-0.2, 0.2]
    coeff[3] = [-0.3, 0.3]
    source = np.repeat(coeff[None], 8, axis=0)
    app = np.repeat((hadamard(32) @ coeff)[None], 4, axis=0)
    actual = score(app, np.array([0.0, 1.0]))
    ci = family_intervals(source, app, actual, np.array([0.0, 1.0]), draws=12)
    assert np.all(ci > 0)
    coeff[3] = 0
    source = np.repeat(coeff[None], 8, axis=0)
    app = np.repeat((hadamard(32) @ coeff)[None], 4, axis=0)
    ci = family_intervals(source, app, score(app, [0.0, 1.0]), [0.0, 1.0], draws=12)
    np.testing.assert_allclose(ci, 0, atol=1e-15)
