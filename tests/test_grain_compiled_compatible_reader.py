#!/usr/bin/env python3
"""Synthetic target identities, matched teaching and moving-frame calculus."""

import numpy as np
import pytest

from grain_research.vector_mechanisms import compiled_compatible_reader as f


def toy():
    """Fixed synthetic paired evidence and a28-row source contract."""
    rng = np.random.default_rng(870)
    current, native = rng.normal(size=(2, 4, 2, 5, 9))
    readers = rng.normal(size=(4, 2, 9))
    common, origin = rng.normal(size=(2, 9))
    units = np.arange(1, 6, dtype=float)
    x = rng.normal(size=(28, 7))
    old = {
        "x": x,
        "v": np.zeros_like(x),
        "kinds": np.zeros(28, bool),
        "weights": np.linspace(0.5, 2, 28),
        "regularizer": 0.02,
        "fixed_frame": rng.normal(size=(5, 9)),
        "units": units,
    }
    t = f.teaching(current, native, readers, common, origin, units)
    return rng, old, t, (current, native, readers, common, origin, units)


def test_compatible_identity_recovers_all_five_contrasts_after_lift():
    rng, _, t, (cur, _, _, common, origin, units) = toy()
    for i in range(4):
        db = rng.normal(size=9)
        d = f.coord.lift_value(
            db, t["frame"][i], t["compatible_coordinates"][i], origin
        )
        result = (cur[i, 0] - cur[i, 1]) @ common + cur[i].mean(axis=0) @ d
        np.testing.assert_allclose(result, t["contrast"][i], atol=1e-10)
        change = d - db
        np.testing.assert_allclose(
            change,
            f.coord.right_inverse(t["frame"][i]) @ (t["frame"][i] @ change),
            atol=1e-10,
        )
    np.testing.assert_allclose(
        t["compatible_coordinates"],
        t["native_coordinates"] - t["other_error"] / units,
        atol=1e-10,
    )


def test_native_evidence_limit_also_requires_matching_common_reader():
    _, _, _, (cur, _, readers, common, origin, units) = toy()
    d = readers[:, 0] - readers[:, 1]
    same_common = np.stack((common + d / 2, common - d / 2), axis=1)
    t = f.teaching(cur, cur, same_common, common, origin, units)
    np.testing.assert_allclose(
        t["compatible_coordinates"], t["native_coordinates"], atol=1e-10
    )
    mismatched = f.teaching(cur, cur, readers, common, origin, units)
    assert not np.allclose(
        mismatched["compatible_coordinates"], mismatched["native_coordinates"]
    )


def test_fixed_subset_weights_lambda_and_target_linearity():
    _, old, t, _ = toy()
    models = [
        f.fit(old, t[k]) for k in ("native_coordinates", "compatible_coordinates")
    ]
    for m in models:
        for k in ("x", "v", "kinds", "weights"):
            np.testing.assert_array_equal(m[k], old[k][f.ORIGINAL])
        assert m["regularizer"] == old["regularizer"]
        assert m["coefficients"] == 20
        k = f.base.gram(m["x"], m["v"], m["kinds"], m["x"], m["v"], m["kinds"])
        np.testing.assert_allclose(
            (k + np.diag(m["regularizer"] / m["weights"] ** 2)) @ m["beta"],
            m["target"],
            atol=1e-10,
        )
    scaled = f.fit(old, 3 * t["compatible_coordinates"])
    np.testing.assert_allclose(scaled["beta"], 3 * models[1]["beta"], atol=1e-10)


def test_coherent_function_derivative_and_null_complement():
    rng, old, t, _ = toy()
    x, dx = rng.normal(size=(2, 7))
    r, dr = rng.normal(size=(2, 5, 9))
    db, ddb, origin = rng.normal(size=(3, 9))
    eps = 1e-5
    for key in ("native_coordinates", "compatible_coordinates"):
        m = f.fit(old, t[key])
        c, dc = f.base.field(m, x, dx)
        _, dd, checks = f.coord.lift(db, ddb[None], r, dr[None], c, dc, origin)
        values = [
            f.coord.lift_value(
                db + s * eps * ddb,
                r + s * eps * dr,
                f.base.field(m, x + s * eps * dx),
                origin,
            )
            for s in (-1, 1)
        ]
        np.testing.assert_allclose(
            dd[0], (values[1] - values[0]) / (2 * eps), atol=1e-7
        )
        assert max(checks) < 1e-10


def test_invalid_teaching_and_subset_fail_closed():
    _, old, t, args = toy()
    with pytest.raises(ValueError):
        f.teaching(*args[:-1], np.zeros(5))
    with pytest.raises(ValueError):
        f.fit(old, t["native_coordinates"][:3])
    changed = old | {"kinds": np.ones(28, bool)}
    with pytest.raises(ValueError):
        f.fit(changed, t["native_coordinates"])


def test_summary_inventory_and_targets_are_independently_reconstructed():
    import pandas as pd

    from scripts.run_grain_compiled_compatible_reader import COUNTS, ep, p82, summarize

    rng, old, t, _ = toy()
    ms = [f.fit(old, t[k]) for k in ("native_coordinates", "compatible_coordinates")]
    baseline = {"m": rng.normal(size=9), "d0": rng.normal(size=9)}
    rows, programs = [], []
    for _ in range(8):
        p = rng.normal(size=(2, 4, 9))
        v = rng.normal(size=(2, 4, 3, 4, 9))
        d = rng.normal(size=(2, 9))
        rr = np.stack((baseline["m"] + d / 2, baseline["m"] - d / 2), axis=1)
        dd = rng.normal(size=(2, 2, 4, 9))
        projected = [p82.projections(p, v, a, b) for a, b in zip(rr, dd, strict=True)]
        nr = rng.normal(size=(2, 9))
        np_ = rng.normal(size=(2, 9))
        nv = rng.normal(size=(2, 3, 4, 9))
        rows.append(
            {k: np.stack([a[k] for a in projected]) for k in projected[0]}
            | {
                "readers": rr,
                "native_r": nr,
                "native_primal": np_,
                "native_vectors": nv,
                "frame": f.coord.frame(p, v) / old["units"][:, None],
                "coordinate_values": rng.normal(size=(2, 5)),
                "target_scalar": np.einsum("qh,qmh->qm", np_, ep.fn.readers(nr)),
                "target_actions": np.einsum("qcfh,qmh->qcmf", nv, ep.fn.readers(nr)),
            }
        )
        programs.append({"primal": p, "vectors": v})
    prior = [
        pd.DataFrame({"variant": ["prior"] * n, "authors": [8] * n})
        for n in (432, 1296, 5184, 234, 18, 1728)
    ]
    tables = summarize(rows, ms, baseline, prior, programs)
    assert [len(a) for a in tables] == COUNTS
    assert set(tables[8].target) == {"native_coordinates", "compatible_coordinates"}
    for before, after in zip(prior, tables, strict=False):
        pd.testing.assert_frame_equal(
            before, after.iloc[: len(before)][list(before)].reset_index(drop=True)
        )


def test_stage_guard_blocks_recipient_targets(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    code = """
import sys
from grain_research.vector_mechanisms.stage_file_guard import install
p = sys.argv[1]
denied = install([], [], [p], [])
assert len(denied) == 1
try:
    open(p, 'rb')
except PermissionError:
    pass
else:
    raise AssertionError('target read escaped guard')
assert len(denied) == 2
"""
    subprocess.run(
        [sys.executable, "-c", code, str(tmp_path / "native_targets.npz")],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )


def test_float32_capture_is_promoted_before_reader_subtraction():
    _, _, _, args = toy()
    cur, native, readers, common, origin, units = args
    rr = readers.astype(np.float32)
    t = f.teaching(cur, native, rr, common, origin, units)
    expected = np.einsum(
        "nkh,nh->nk",
        t["frame"],
        rr.astype(float)[:, 0] - rr.astype(float)[:, 1] - origin,
    )
    np.testing.assert_array_equal(t["native_coordinates"], expected)
