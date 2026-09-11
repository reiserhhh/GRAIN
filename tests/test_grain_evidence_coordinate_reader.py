#!/usr/bin/env python3
"""Independent constraint geometry, moving-frame calculus and source binding."""

import numpy as np
import pandas as pd
import pytest

from grain_research.vector_mechanisms import evidence_coordinate_reader as f
from grain_research.vector_mechanisms import observable_reader_teaching as oldfit


def toy():
    """Finite mixed source observations with a low-rank ambient target."""
    rng = np.random.default_rng(850)
    x, v = rng.normal(size=(2, 9, 4))
    y = np.zeros((9, 11))
    y[:, :3] = rng.normal(size=(9, 3))
    groups = np.repeat(np.arange(3), 3)
    m = f.base.fit(
        x, v, groups > 0, y, groups, rng.normal(size=11), rng.normal(size=11)
    )
    m.update(target=y, groups=groups)
    return rng, m


def test_minimum_change_and_coordinate_units():
    rng = np.random.default_rng(85)
    r = rng.normal(size=(5, 11))
    db, d0 = rng.normal(size=(2, 11))
    c = rng.normal(size=5)
    d = f.lift_value(db, r, c, d0)
    delta = np.linalg.lstsq(r, c - r @ (db - d0), rcond=None)[0]
    np.testing.assert_allclose(d, db + delta, atol=1e-11)
    u = f.right_inverse(r)
    np.testing.assert_allclose((np.eye(11) - u @ r) @ (d - db), 0, atol=1e-11)
    scale = np.arange(1, 6)
    np.testing.assert_allclose(
        f.lift_value(db, r / scale[:, None], c / scale, d0), d, atol=1e-11
    )
    np.testing.assert_allclose(f.lift_value(db, r, r @ (db - d0), d0), db, atol=1e-12)
    with pytest.raises(ValueError):
        f.right_inverse(np.ones((5, 11)))


def test_inverse_derivative_and_full_moving_lift():
    rng = np.random.default_rng(851)
    r, dr = rng.normal(size=(2, 5, 11))
    db, ddb, d0 = rng.normal(size=(3, 11))
    c, dc = rng.normal(size=(2, 5))
    eps = 1e-5
    u = f.right_inverse(r)
    np.testing.assert_allclose(
        f.inverse_tangent(r, dr[None], u)[0],
        (f.right_inverse(r + eps * dr) - f.right_inverse(r - eps * dr)) / (2 * eps),
        atol=1e-8,
    )
    d, dd, checks = f.lift(db, ddb[None], r, dr[None], c, dc[None], d0)
    plus = f.lift_value(db + eps * ddb, r + eps * dr, c + eps * dc, d0)
    minus = f.lift_value(db - eps * ddb, r - eps * dr, c - eps * dc, d0)
    np.testing.assert_allclose(dd[0], (plus - minus) / (2 * eps), atol=1e-8)
    np.testing.assert_allclose(r @ (d - d0), c, atol=1e-10)
    assert max(checks) < 1e-10


def test_product_targets_bind_body_to_clamp_and_context_to_original():
    rng = np.random.default_rng(852)
    y = rng.normal(size=(28, 11))
    r, dr = rng.normal(size=(2, 28, 5, 11))
    dr[:8] = 0
    groups = np.repeat(np.arange(3), [8, 16, 4])
    result = f.observation_coordinates(y, r, dr, groups)
    expected = np.stack([r[i] @ y[i] for i in range(28)])
    for i in range(4):
        for j in range(8 + 4 * i, 12 + 4 * i):
            expected[j] += dr[j] @ y[2 * i + 1]
        expected[24 + i] += dr[24 + i] @ y[2 * i]
    np.testing.assert_allclose(result, expected, atol=1e-12)
    with pytest.raises(AssertionError):
        f.observation_coordinates(y, r, dr, groups[::-1])


def test_coordinate_solve_against_whitened_augmented_lstsq():
    rng, raw = toy()
    y = rng.normal(size=(9, 5))
    m = f.fit_coordinates(raw, y)
    k = f.base.gram(raw["x"], raw["v"], raw["kinds"], raw["x"], raw["v"], raw["kinds"])
    eig, vec = np.linalg.eigh(k)
    a = vec * np.sqrt(eig)
    w = m["weights"][:, None]
    design = np.concatenate((w * a, np.sqrt(raw["regularizer"]) * np.eye(9)))
    target = np.concatenate((w * y, np.zeros((9, 5))))
    theta = np.linalg.lstsq(design, target, rcond=None)[0]
    np.testing.assert_allclose(k @ m["beta"], a @ theta, atol=1e-9)
    np.testing.assert_allclose(
        m["regularization_penalty"], raw["regularizer"] * np.sum(theta**2), atol=1e-9
    )
    assert m["regularizer"] == raw["regularizer"]


def test_frame_extracts_same_composition_and_four_actions():
    rng = np.random.default_rng(853)
    p, v = rng.normal(size=(2, 4, 11)), rng.normal(size=(2, 4, 3, 4, 11))
    r = f.frame(p, v)
    np.testing.assert_array_equal(r[0], p[:, 3].mean(axis=0))
    np.testing.assert_array_equal(r[1:], v[:, 3, 2].mean(axis=0))
    with pytest.raises(ValueError):
        f.frame(p[:1], v)


def test_twelve_field_summary_keeps_old_slots_and_native_targets():
    from scripts.run_grain_evidence_coordinate_reader import (
        COUNTS,
        p82,
        previous,
        summarize,
    )

    rng, raw = toy()
    restricted = [
        oldfit.fit(raw, rng.normal(size=(9, 5, 11)), flag) for flag in (False, True)
    ]
    frame = rng.normal(size=(5, 11))
    models = [
        f.fit_coordinates(raw, rng.normal(size=(9, 5)))
        | {"fixed_frame": frame, "moving": bool(i), "source_lift_checks": np.zeros(3)}
        for i in range(2)
    ]
    teaching = {
        "original_frames": rng.normal(size=(4, 5, 11)),
        "clamped_frames": rng.normal(size=(4, 5, 11)),
        "fd_abs": np.zeros((4, 5, 5)),
        "fd_relative": np.zeros((4, 5, 5)),
    }
    rows = []
    for _ in range(8):
        rr = rng.normal(size=(12, 2, 11))
        for vi, m in enumerate(restricted, 6):
            d = m["d0"] + rng.normal(size=3) @ m["output_basis"].T
            rr[vi] = np.stack((m["m"] + d / 2, m["m"] - d / 2))
        dr = rng.normal(size=(12, 2, 4, 11))
        p, v = rng.normal(size=(2, 4, 11)), rng.normal(size=(2, 4, 3, 4, 11))
        packets = [p82.projections(p, v, a, b) for a, b in zip(rr, dr, strict=True)]
        rows.append(
            {k: np.stack([x[k] for x in packets]) for k in packets[0]}
            | {
                "readers": rr,
                "reader_derivatives": dr,
                "frame": rng.normal(size=(5, 11)),
                "coordinate_values": rng.normal(size=(2, 5)),
                "lift_checks": np.zeros((2, 3)),
                "native_r": rng.normal(size=(2, 11)),
                "native_primal": rng.normal(size=(2, 11)),
                "native_vectors": rng.normal(size=(2, 3, 4, 11)),
                "target_scalar": rng.normal(size=(2, 2)),
                "target_actions": rng.normal(size=(2, 3, 2, 4)),
            }
        )
    placeholders = [
        pd.DataFrame({"variant": ["old"] * n, "authors": [8] * n})
        for n in previous.COUNTS
    ]
    tables = summarize(rows, models, raw, teaching, placeholders, restricted)
    assert [len(t) for t in tables] == COUNTS
    assert set(tables[2].tail(48)["variant"]) == set(f.VARIANTS)
    assert set(tables[14]["coordinate"]) == {
        "scalar",
        "total_F01",
        "total_F02",
        "total_F03",
        "total_F04",
    }
    for a, b in zip(placeholders, tables, strict=False):
        pd.testing.assert_frame_equal(
            a, b.iloc[: len(a)][list(a)].reset_index(drop=True)
        )
