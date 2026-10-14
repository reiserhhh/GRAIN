#!/usr/bin/env python3
"""F11 MTMM checks."""

import numpy as np

from grain_research.gptpro_foundation_v3.f11 import covariance_matrix, fit_moments


def test_mtmm_covariance_is_psd_and_anchor_required():
    matrix = covariance_matrix()
    assert np.linalg.eigvalsh(matrix).min() >= -1e-10
    assert fit_moments(matrix)["anchor_required"] is True
