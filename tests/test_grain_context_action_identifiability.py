#!/usr/bin/env python3
"""Existence and absence of source-equivalent application function changes."""

import numpy as np

from scripts.audit_grain_context_action_identifiability_hm139 import extension_geometry


def test_unobserved_application_direction():
    a = np.array([[1.0, 0, 0], [0, 1, 0]])
    b = np.array([[0.0, 0, 1]])
    r, v = extension_geometry(a, b, np.ones(3))
    assert r["unidentified_parameters"] == 1
    assert r["application_action_unseen_energy_fraction"] == 1
    np.testing.assert_allclose(a @ v["coefficient_difference_raw"], 0)
    np.testing.assert_allclose(b @ v["coefficient_difference_raw"], 1)


def test_no_new_identifiability_problem_when_inside_source():
    a = np.array([[1.0, 0, 0], [0, 1, 0]])
    b = np.array([[3.0, 2, 0]])
    r, _ = extension_geometry(a, b, np.ones(3))
    assert r["unidentified_parameters"] == 1
    assert r["application_unit_coefficient_witness_max_change"] == 0
    r, _ = extension_geometry(np.eye(3), b, np.ones(3))
    assert r["unidentified_parameters"] == 0
