#!/usr/bin/env python3
"""Geometry tests do not call a neural observer."""

import numpy as np

from grain_research.construction import math_readout_quotient as Q


def test_exact_gauge_null_and_head_action():
    head = np.random.default_rng(2).normal(size=(10, 40))
    quotient = Q.quotient_basis(head)
    assert quotient["rank"] == 9
    split = Q.split_displacement(np.arange(40), quotient)
    assert split["null_action_max"] < 1e-10
    assert split["energy_closure_error"] < 1e-8
    np.testing.assert_allclose(
        split["observable"] + split["head_null"], np.arange(40), atol=1e-12
    )


def test_rotation_covariance_with_transported_head():
    rng = np.random.default_rng(3)
    head, delta = rng.normal(size=(5, 20)), rng.normal(size=20)
    rotation, _ = np.linalg.qr(rng.normal(size=(20, 20)))
    original = Q.split_displacement(delta, Q.quotient_basis(head))
    moved = Q.split_displacement(rotation @ delta, Q.quotient_basis(head @ rotation.T))
    np.testing.assert_allclose(
        moved["observable"], rotation @ original["observable"], atol=1e-12
    )
    assert abs(original["observable_energy"] - moved["observable_energy"]) < 1e-12


def test_null_direction_is_not_universal_noise():
    first = np.array([[1, 0, 0], [0, 0, 0]], float)
    second = np.array([[0, 1, 0], [0, 0, 0]], float)
    delta = np.array([0, 1, 0], float)
    split = Q.split_displacement(delta, Q.quotient_basis(first))
    assert split["observable_energy"] == 0
    assert np.linalg.norm(second @ split["head_null"]) > 0
