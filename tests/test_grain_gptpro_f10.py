#!/usr/bin/env python3
"""F10 action metric checks."""

import numpy as np

from grain_research.gptpro_foundation_v3.f10 import common_metric, edit_signature, quadratic_response, svd_rank


def test_finite_signature_and_metric():
    base = np.zeros(4); edited = np.ones(4)
    np.testing.assert_allclose(edit_signature(base, edited), np.ones(4))
    j = np.eye(4)
    np.testing.assert_allclose(common_metric([j]), np.eye(4))
    assert svd_rank(j) == 4


def test_quadratic_is_not_linear():
    h = np.array([.5, -.2]); u = np.array([.1, .2]); b = np.eye(2)
    assert quadratic_response(h, u, b) != float(u @ h)
