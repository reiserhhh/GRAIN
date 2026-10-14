#!/usr/bin/env python3
"""Synthetic finite sampling levels and conditional-contrast account tests."""

import numpy as np
import pytest

from grain_research.construction import math_source_law_update as M


@pytest.mark.parametrize("mode", ["author", "evidence", "joint", "null"])
def test_sampling_levels(mode):
    """Keep both levels and their signed interaction instead of a single residual."""
    a = np.array([0.25, 0.75])
    b = np.array([[0.4, 0.6], [0.7, 0.3]])
    aa = np.array([0.5, 0.5]) if mode in ("author", "joint") else a
    bb = b[:, ::-1] if mode in ("evidence", "joint") else b
    p, h = a[:, None] * b, aa[:, None] * bb
    laws, i = M.four_laws(p, h)
    np.testing.assert_allclose(h - p, laws[1] - p + laws[2] - p + i, atol=1e-15)
    if mode in ("author", "null"):
        np.testing.assert_allclose(laws[2], p)
    if mode in ("evidence", "null"):
        np.testing.assert_allclose(laws[1], p)


@pytest.mark.parametrize(
    "p",
    [
        np.array([[-0.1, 0.6], [0.2, 0.3]]),
        np.array([[0.0, 0.0], [0.5, 0.5]]),
        np.ones((2, 2)),
    ],
)
def test_invalid_law(p):
    """Reject impossible measures rather than clipping or adding author support."""
    with pytest.raises(ValueError):
        M.disintegrate(p)


def test_marginal_change_unequal_prevalence():
    """A signed-label moment alone is not the normalized high-low derivative."""
    y = np.array([[int(i >> q & 1) for q in range(5)] for i in range(32)])
    y = np.concatenate((y, y[y[:, 0] == 1]))
    rng = np.random.default_rng(183)
    x, d = rng.normal(size=(48, 3)), rng.normal(size=48)
    for profile in np.unique(y, axis=0):
        mask = np.all(y == profile, axis=1)
        d[mask] -= d[mask].mean()
    signs = np.column_stack((np.ones(48), 2 * y - 1))
    w = signs.T @ (d[:, None] * x)
    direct = np.array(
        [
            (d[y[:, q] == 1] @ x[y[:, q] == 1]) / y[:, q].mean()
            - (d[y[:, q] == 0] @ x[y[:, q] == 0]) / (1 - y[:, q].mean())
            for q in range(5)
        ]
    )
    np.testing.assert_allclose(M.contrast_change(w, y.mean(0)), direct, atol=1e-14)


def test_profile_preserving_readers():
    """Source effective features retain the exact four-law target-change identity."""
    y = np.repeat([[int(i >> q & 1) for q in range(5)] for i in range(32)], 2, axis=0)
    rng = np.random.default_rng(184)
    phi = rng.normal(size=(64, 3, 4))
    p = np.full((64, 3), 1 / 192)
    h = p.copy()
    h[::2] *= 1.1
    h[1::2] *= 0.9
    laws, _ = M.four_laws(p, h)
    f = M.readers_from_features(M.effective_features(phi, laws), y)
    delta = f[3] - f[0]
    account = f[1] - f[0] + f[2] - f[0] + f[3] - f[1] - f[2] + f[0]
    np.testing.assert_allclose(delta, account, atol=1e-15)


def test_cold_labels_and_missing():
    """No application scale,author type or missing-role neutral trait is inferred."""
    f = np.zeros((2, 6, 4, 5, 4, 4096))
    row = {"frame_key": "opaque", "view": 0, "events": []}
    value, n = M.cold_record(f, row)
    assert n == 0 and np.max(np.abs(value)) == 0
    with pytest.raises(ValueError):
        M.cold_record(f, row | {"targets": [0] * 5})
