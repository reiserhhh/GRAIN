#!/usr/bin/env python3
"""Exact finite witnesses for the next conditional sensitivity-field construction."""

import numpy as np
import pytest


def test_group_constant_gradient_cancels_but_state_dependent_field_does_not():
    h = np.array([[1.0, 3.0], [3.0, -1.0], [2.0, 2.0]])
    d = h - h.mean(0)
    constant = np.broadcast_to(np.array([2.0, -4.0]), h.shape)
    assert np.sum(d * constant) == pytest.approx(0)
    matrix = np.diag([2.0, -0.5])
    # F(H)=1/2 sum h' A h gives the exact integrated gradient A(m+d/2).
    gradient = (h.mean(0) + d / 2) @ matrix
    effect = 0.5 * np.sum(d * (d @ matrix))
    assert np.sum(d * gradient) == pytest.approx(effect)
    assert np.sum(d * (gradient - gradient.mean(0))) == pytest.approx(effect)
    assert np.sum(d * gradient) != pytest.approx(0)


def test_covariance_trace_and_token_count_reparameterization_are_exact():
    rng = np.random.default_rng(87)
    h, g = rng.normal(size=(11, 7)), rng.normal(size=(11, 7))
    d = h - h.mean(0)
    cross_cov = d.T @ (g - g.mean(0)) / len(h)
    gamma = len(h) * (g - g.mean(0))
    assert np.sum(d * g) == pytest.approx(len(h) * np.trace(cross_cov))
    assert np.sum(d * gamma) / len(h) == pytest.approx(np.sum(d * g))
