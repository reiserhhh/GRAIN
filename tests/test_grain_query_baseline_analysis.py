#!/usr/bin/env python3
"""Geometry and constant-correlation checks for source derivative reports."""

import numpy as np
import pytest

from scripts.analyze_grain_query_baseline import corr, field_geometry


def test_raw_constant_is_not_roundoff_correlation():
    assert np.isnan(corr(np.arange(20), np.full(20, 0.1)))


def test_common_gradient_field():
    x = np.tile(np.arange(1, 10, dtype=float), (4, 1))
    g = field_geometry(x)
    assert g["rank95"] == 1
    assert g["effective_rank"] == pytest.approx(1)
    assert g["common_field_energy_ratio"] == pytest.approx(1)
    assert g["cosine_mean"] == pytest.approx(1)


def test_cancelling_sources_have_shared_energy_not_mean_field():
    x = np.array([[1.0, 2.0], [-1.0, -2.0]])
    g = field_geometry(x)
    assert g["top1_energy_ratio"] == pytest.approx(1)
    assert g["common_field_energy_ratio"] == pytest.approx(0)
    assert g["cosine_mean"] == pytest.approx(-1)
