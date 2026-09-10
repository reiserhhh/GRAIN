#!/usr/bin/env python3
"""Source clamp, BODY-only motion, constant lower bound and complete aggregate panels."""

import numpy as np
import pytest

from grain_research.vector_mechanisms import reader_material_dependence as f


def test_clamp_and_body_intervention():
    rng = np.random.default_rng(81)
    x = rng.normal(size=(7, 9)).astype(np.float32)
    data = {"state": x, "positions": np.array([1, 2, 4])}
    last = rng.normal(size=9).astype(np.float32)
    c = f.clamp(data, last)
    np.testing.assert_array_equal(c["state"][:-1], x[:-1])
    np.testing.assert_array_equal(c["state"][-1], last)
    d = np.zeros_like(x)
    d[data["positions"]] = 1
    p = f.perturb(c, d, 0.01)
    np.testing.assert_array_equal(p["state"][[0, 3, 5, 6]], c["state"][[0, 3, 5, 6]])
    assert np.any(p["state"][data["positions"]] != c["state"][data["positions"]])
    d[-1] = 1
    with pytest.raises(ValueError):
        f.perturb(c, d, 0.01)


def test_best_constant_identity_independent_lstsq():
    rng = np.random.default_rng(82)
    x = rng.normal(size=(4, 9))
    a = f.dispersion(x)
    fitted = np.ones((4, 1)) @ np.linalg.lstsq(np.ones((4, 1)), x, rcond=None)[0]
    np.testing.assert_allclose(a["constant_floor_sse"], np.sum((x - fitted) ** 2))
    np.testing.assert_allclose(a["energy"], a["constant_floor_sse"] + a["mean_energy"])
    assert f.dispersion(np.ones((4, 9)))["constant_floor_sse"] == 0


def test_secant_sign_and_query_order():
    rng = np.random.default_rng(83)
    r, j = rng.normal(size=(2, 2, 9))
    eta = 0.01
    x = np.stack((r - eta * j, r + eta * j), axis=1)
    s = f.query_slopes(x, eta)
    np.testing.assert_allclose(s[:2], j, atol=1e-12)
    np.testing.assert_allclose(s[2], j[0] - j[1], atol=1e-12)


def test_output_range_does_not_bound_projected_action():
    basis = np.eye(3)[:1]
    x = np.array([[0.0, 1.0, 0.0]])
    assert f.energy_outside(x, basis) == 1
    assert float((x @ np.array([1.0, 0.0, 0.0])).item()) == 0
    assert f.energy_outside(np.zeros((4, 3)), basis) == 0


def test_scalar_sign_counts_keep_authors():
    a = f.summarize_scalars(np.array([-1.0, 0.0, 1.0, 2.0]))
    assert (
        a["authors"],
        a["positive_authors"],
        a["negative_authors"],
        a["zero_authors"],
    ) == (4, 2, 1, 1)


def test_fixed_panel_shapes_and_clamp_floor():
    from scripts.run_grain_reader_material_dependence import COUNTS, tables

    rng = np.random.default_rng(84)
    packet = {
        "original": rng.normal(size=(4, 3, 9)),
        "fixed": rng.normal(size=(4, 3, 9)),
        "slopes": rng.normal(size=(4, 4, 3, 9)),
        "primal": rng.normal(size=(4, 2, 5, 9)),
        "vectors": rng.normal(size=(4, 2, 5, 3, 4, 9)),
        "original_last": rng.normal(size=(4, 2, 9)),
        "last": np.ones((4, 2, 9)),
        "body_count": np.ones((4, 2)),
    }
    ts = tables(packet, np.eye(9)[:3], np.eye(9)[:7])
    assert [len(t) for t in ts] == COUNTS
    assert not ts[-1].last_max_difference.any()
    assert set(ts[2].factor) == {"F01", "F02", "F03", "F04"}
    assert all((t.authors == 4).all() for t in ts)
