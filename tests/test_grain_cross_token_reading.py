#!/usr/bin/env python3
"""Region intervention, byte invariance and finite contraction witnesses."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.cross_token_reading import (
    intervention,
    region_account,
)


def toy():
    """Make a gapped BODY so non-BODY invariance is also exercised."""
    rng = np.random.default_rng(67)
    return (
        rng.normal(size=(12, 5)).astype(np.float32),
        np.array([1, 3, 4, 7, 9]),
        rng.normal(size=(4, 5, 5)),
    )


def test_fixed_regions_and_equal_factor_scale_invariance():
    x, pos, ds = toy()
    p, m, meta = intervention(x, pos, ds)
    assert p.dtype == m.dtype == x.dtype
    assert p[meta["unchanged_mask"]].tobytes() == x[meta["unchanged_mask"]].tobytes()
    assert m[meta["unchanged_mask"]].tobytes() == x[meta["unchanged_mask"]].tobytes()
    assert np.linalg.norm(meta["unit_direction"]) == pytest.approx(1)
    assert float(
        meta["realized_plus_norm"] / np.linalg.norm(x[pos[:2]])
    ) == pytest.approx(0.01, rel=1e-5)
    p2, m2, _ = intervention(x, pos, ds * np.arange(1, 5)[:, None, None])
    np.testing.assert_array_equal(p, p2)
    np.testing.assert_array_equal(m, m2)


def test_zero_and_cancelling_directions_are_not_replaced():
    x, pos, ds = toy()
    ds[0] = 0
    with pytest.raises(ValueError, match="vanishes"):
        intervention(x, pos, ds)
    ds[:] = 1
    ds[2:] = -1
    with pytest.raises(ValueError, match="cancels"):
        intervention(x, pos, ds)
    with pytest.raises(ValueError, match="ordered"):
        intervention(x, np.array([1, 1, 4, 7, 9]), ds)


def test_finite_response_uses_fixed_directions_and_sign_reversal():
    x, pos, ds = toy()
    g = x[pos].astype(float)
    gp, gm = g + 0.1, g - 0.2
    rows, acts = region_account(g, gp, gm, ds, 2)
    flipped, opposite = region_account(g, gm, gp, ds, 2)
    for region, sl in (("A", slice(None, 2)), ("B", slice(2, None))):
        np.testing.assert_allclose(
            acts[region]["delta"], 0.15 * ds[:, sl].sum(axis=(1, 2))
        )
        np.testing.assert_allclose(acts[region]["delta"], -opposite[region]["delta"])
        assert rows[region] == flipped[region]


def test_local_sum_zero_vs_normalized_pooling_nonzero():
    """Cross-region response can exist while every token in the observed region is fixed."""
    x, pos, ds = toy()
    p, m, meta = intervention(x, pos, ds)
    cut = int(meta["cut"])
    rows, _ = region_account(2 * x[pos], 2 * p[pos], 2 * m[pos], ds, cut)
    assert rows["B"]["delta_norm"] == 0

    def gradient(state):
        z = state[pos].astype(float)
        weights = np.exp(z[:, 0] - max(z[:, 0]))
        weights /= weights.sum()
        f = weights @ z[:, 1]
        g = np.zeros_like(z)
        g[:, 1] = weights
        g[:, 0] = weights * (z[:, 1] - f)
        return g

    rows, _ = region_account(gradient(x), gradient(p), gradient(m), ds, cut)
    assert rows["B"]["delta_norm"] > 1e-7
