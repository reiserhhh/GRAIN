#!/usr/bin/env python3
"""Synthetic tests for prospective evidence, without real texts or model calls."""

import numpy as np
import pytest

from grain_research.construction import math_prospective_evidence as M


def bank():
    """Provide all32 source profiles with an identifiable A direction."""
    codes = np.tile(np.arange(32), 2)
    p = np.random.default_rng(218).normal(0, 0.1, (64, 2, 3))
    p[:, :, 0] += (((codes >> 2) & 1) * 2 - 1)[:, None]
    return {"source_means": p, "codes": codes, "variance": 0.5}


def test_source_shape_and_support():
    """An incomplete target catalogue cannot silently acquire zero mass."""
    b = bank()
    M.validate(b)
    with pytest.raises(ValueError):
        M.validate(b | {"codes": np.zeros(64, int)})
    with pytest.raises(ValueError):
        M.validate(b | {"variance": 0})


def test_score_additivity_and_batch_independence():
    """Cold scores do not depend on accompanying test authors."""
    x = np.random.default_rng(13).normal(size=(4, 3))
    v = M.accounts(bank(), x)
    np.testing.assert_allclose(v[:, 0], v[:, 1] + v[:, 2], atol=1e-15)
    np.testing.assert_allclose(
        v, np.concatenate([M.accounts(bank(), a[None]) for a in x])
    )


def test_equal_anchors_remove_departure():
    """Equal actual conditional laws yield no manufactured A departure."""
    b = bank()
    b["source_means"][4::8] = b["source_means"][0::8]
    b["source_means"][5::8] = b["source_means"][1::8]
    b["source_means"][6::8] = b["source_means"][2::8]
    b["source_means"][7::8] = b["source_means"][3::8]
    np.testing.assert_allclose(M.accounts(b, np.zeros((2, 3)))[:, 2, 2], 0, atol=1e-14)


def test_matched_source_only_selection():
    """Removal pairs are disjoint and exactly condition-count matched."""
    x = np.random.default_rng(12).normal(size=(12, 3))
    r = M.select(bank(), x, np.arange(12) % 2, np.full(12, 20, int))
    i, j = r["support"], r["counter"]
    assert not set(r["subsets"][i]) & set(r["subsets"][j])
    assert r["condition_counts"][i] == r["condition_counts"][j]
    assert r["accounts"][j, 2, 2] >= r["accounts"][i, 2, 2]
    assert r["subsets"].shape == (495, 4)
    np.testing.assert_allclose(r["points"][i], np.delete(x, r["subsets"][i], 0).mean(0))


def test_no_matching_relaxation():
    """Unsupported operations fail rather than changing the frozen design."""
    x = np.zeros((12, 3))
    with pytest.raises(ValueError, match="No supported"):
        M.select(bank(), x, np.zeros(12), 2 ** np.arange(12), absolute=0, relative=0)


def test_native_generic_and_specific_are_separate():
    """A global reader shift does not manufacture target-specific evidence."""
    x = np.zeros((3, 3, 5, 10))
    x[:, 2, :, 5:] = 0.3
    y, raw, specific = M.native_accounts(x)
    np.testing.assert_allclose(raw, 0.3)
    np.testing.assert_allclose(specific, 0, atol=1e-15)
    assert y.shape == (3, 3, 5)
    with pytest.raises(ValueError):
        M.native_accounts(x[:, :2])


def test_author_bootstrap_deterministic():
    """Intervals resample authors with a fixed seed and finite bounds."""
    x = np.arange(20).reshape(4, 5)
    a = M.bootstrap(x, 99, 12)
    np.testing.assert_array_equal(a, M.bootstrap(x, 99, 12))
    assert a.shape == (5, 2)
