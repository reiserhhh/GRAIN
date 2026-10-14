#!/usr/bin/env python3
"""Finite tests of the author operator and inference unit, without project data."""

import numpy as np
import pytest

from grain_research.construction import math_author_pair_criterion as M
from grain_research.construction import math_source_event_coupling as P


def test_kernel_weights_equal_existing_profile_laws():
    """Source-author weights recover the original F/J/R, not a new estimator."""
    rng = np.random.default_rng(277)
    logs = rng.normal(size=(3, 64, 2, 22))
    codes = np.tile(np.arange(32), 2)
    relink = np.tile(np.roll(np.arange(22), 1), (64, 1))
    old = P.profile_readings(logs, codes, relink)
    weights = M.source_author_weights(logs, relink)
    np.testing.assert_allclose(
        weights @ P.BITS[codes], old["targets"][:, M.WEIGHT_LAWS], atol=1e-14
    )


def test_arithmetic_aggregation_keeps_all_windows():
    """Repeated pairs do not create overconfident product evidence."""
    p = np.zeros((22, 4, 32))
    p[:11, :, 0], p[11:, :, 1] = 1, 1
    out = M.aggregate_profiles(p)
    assert out.shape == (3, 4, 32)
    np.testing.assert_array_equal(out[2, :, :2], np.full((4, 2), 0.5))
    with pytest.raises(ValueError):
        M.aggregate_profiles(p[:-1])


def test_metric_bank_offset_does_not_create_primary_link():
    """Metric centering is distinct from modifying the text representation."""
    b = np.repeat((0, 1), 4)
    x = np.array([0, 1, 0, 1, 10, 11, 10, 11.0])[:, None]
    y = np.array([0, 0, 1, 1, 10, 10, 11, 11.0])[:, None]
    assert M.corr(x, y)[0] > 0.9
    assert abs(M.corr(x, y, b)[0]) < 1e-14
    assert x[4, 0] == 10


def test_whole_profile_permutation_and_crossfit_guard():
    """A complete recomputation preserves label dependencies and forbids self sources."""
    b = np.repeat((0, 1), 32)
    g = np.tile(np.arange(32), 2)
    rng = np.random.default_rng(4)
    d = rng.normal(size=(2, 64, 64))
    d[:, b[:, None] == b[None, :]] = 0
    out = M.permutation_inference(d, g, b, 9, 27)
    assert out[0].shape == (2, 5)
    assert np.all(out[3] >= out[2])
    for perm in out[4]:
        for k in (0, 1):
            np.testing.assert_array_equal(np.sort(perm[b == k]), np.arange(32))
    d[0, 0, 0] = 1
    with pytest.raises(ValueError, match="Own-bank"):
        M.permutation_inference(d, g, b, 9, 27)


def test_matching_is_without_labels_and_excludes_self():
    """Matching uses condition counts only, not learned score similarity."""
    conditions = np.zeros((8, 24), int)
    banks = np.repeat((0, 1), 4)
    match, _, dist = M.matched_strangers(conditions, banks)
    assert np.all(match != np.arange(8))
    np.testing.assert_array_equal(banks[match], banks)
    assert np.all(dist == 0)


def test_cold_scoring_and_components():
    """All signed differences retain their mathematical reconstruction."""
    rng = np.random.default_rng(7)
    bundle = {
        "prototypes": rng.normal(size=(32, 22, 2, 4)),
        "codes": np.arange(32),
        "relink": np.tile(np.arange(22), (32, 1)),
        "projection": np.eye(4),
        "variance": 1,
    }
    out = M.cold_author(bundle, rng.normal(size=(22, 2, 4)), "GRAIN15")
    series = out["series"]
    assert series.shape == (3, 8, 5)
    np.testing.assert_allclose(series[:, 4] + series[:, 5], series[:, 6], atol=1e-14)
