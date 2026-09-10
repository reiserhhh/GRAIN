#!/usr/bin/env python3
"""Counterexample and exact signed account for adding a nested reader mode."""

import numpy as np

from grain_research.vector_mechanisms.position_reader import (
    contractions,
    errors,
    project_field,
)
from scripts.analyze_grain_position_modes import addition_account


def test_better_field_projection_can_worsen_the_measured_action():
    j = np.diag([3.0, 2.0, 1.0])
    d = np.diag([1 / 3, -0.5, 1.0])[None]
    a = project_field(j, np.eye(3), 1)
    b = project_field(j, np.eye(3), 2)
    qa, qb = errors(j, a, d), errors(j, b, d)
    assert qb["field_error_ratio"] < qa["field_error_ratio"]
    assert qa["predicted"][0] == qa["actual"][0] == 1
    assert abs(qb["predicted"][0]) < 1e-12
    result = addition_account(
        contractions(j, d), contractions(a, d), contractions(b, d)
    )
    assert result["old_action_error"] == 0
    assert result["new_action_error"] == 1
    assert result["identity_error"] == 0


def test_action_addition_account_is_exact_for_correlated_terms():
    rng = np.random.default_rng(3)
    y, a, b = rng.normal(size=(3, 16))
    result = addition_account(y, a, b)
    assert abs(result["identity_error"]) < 1e-12
