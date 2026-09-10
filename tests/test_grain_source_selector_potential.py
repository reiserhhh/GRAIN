#!/usr/bin/env python3
"""Matched scalar/tangent, gauge, source-fit and data-boundary tests."""

import subprocess
import sys

import numpy as np
import pytest

from grain_research.vector_mechanisms import source_selector_potential as f
from grain_research.vector_mechanisms.source_query_selection import key_actions


def toy(seed=74):
    """Small fixed-layout GQA material with one forbidden attention token."""
    r = np.random.default_rng(seed)
    t, h, hd, kv, rank = 13, 8, 4, 1, 2
    phase = r.normal(size=(t, hd))
    data = {
        "state": r.normal(size=(t, h)),
        "cos": np.cos(phase),
        "sin": np.sin(phase),
        "mask": np.r_[-np.inf, np.zeros(t - 1)],
    }
    d = r.normal(size=(4, t, h)) * 0.1
    w = {
        "k": r.normal(size=(kv * hd, h)),
        "k_bias": r.normal(size=kv * hd),
        "gamma": r.uniform(0.5, 1.5, size=h),
        "epsilon": np.asarray(1e-6),
        "head_dim": np.asarray(hd),
        "scale": np.asarray(hd**-0.5),
    }
    width = 2 * rank + 3
    base = {
        "mean": r.normal(size=h),
        "basis": r.normal(size=(h, rank)),
        "center": r.normal(size=width),
        "scale": r.uniform(0.5, 1.5, size=width),
        "mode": np.asarray("causal_context"),
        "coef": r.normal(size=(width, 2)) * 0.01,
        "anchor": r.normal(size=(2, hd)),
        "members": np.array([0, 1, 2]),
        "alpha": np.asarray(0.01),
        "rank": np.asarray(rank),
    }
    base["coef"][-3] = 0
    v = {"v": r.normal(size=(t, kv, hd)), "g": r.normal(size=(2, 2, hd))}
    return data, d, w, base, v


def test_scalar_and_grain_derivative_same_function():
    data, d, w, model, v = toy()
    ell, dell = f.potential(data, d, w, model)
    actual = f.readout(ell, dell, data["mask"], v)
    for j in range(4):
        values, ls = [], []
        for eps in (-1e-5, 1e-5):
            moved = dict(data, state=data["state"] + eps * d[j])
            l, dl = f.potential(moved, d, w, model)
            ls.append(l)
            values.append(f.readout(l, dl, data["mask"], v)["scalar"])
        np.testing.assert_allclose((ls[1] - ls[0]) / 2e-5, dell[j], atol=1e-8)
        np.testing.assert_allclose(
            (values[1] - values[0]) / 2e-5, actual["actions"][:, j], atol=1e-8
        )


def test_teacher_equivalence_gauge_and_mask():
    data, d, w, model, v = toy()
    k, dk = f.identity_keys(data, d, w)
    ell, dell = f.logits(model["anchor"], k, dk, data, float(w["scale"]))
    result = f.readout(ell, dell, data["mask"], v)
    e = (
        v
        | {n: data[n] for n in ("cos", "sin", "mask")}
        | {"k": k, "dk": dk, "scale": w["scale"]}
    )
    np.testing.assert_allclose(
        result["actions"], key_actions(model["anchor"][None], e)[0], atol=1e-12
    )
    shift = f.readout(
        ell + [40, -20], dell + np.arange(8).reshape(4, 1, 2), data["mask"], v
    )
    for key in ("actions", "attention", "scalar"):
        np.testing.assert_allclose(shift[key], result[key], atol=1e-12)
    np.testing.assert_array_equal(result["attention"][0], 0)
    with pytest.raises(ValueError, match="Only value"):
        f.readout(ell, dell, data["mask"], v | {"k": k})
    with pytest.raises(ValueError, match="Current-query"):
        f.identity_keys(data, d, w | {"q": np.ones((2, 8))})


def source_items():
    """Teacher key targets for exactly three permitted authors and a held poison."""
    items = []
    for index in range(3):
        data, d, w, _base, _ = toy(74 + index)
        k0, dk0 = f.identity_keys(data, d, w)
        items.append(
            {
                "index": index,
                "query": "neuroticism",
                "data": data,
                "d": d,
                "k0": k0,
                "dk0": dk0,
                "k": k0 * 1.3,
                "dk": dk0 * 1.4,
            }
        )
    return items


def test_source_anchor_consistency_zero_gauge_and_exclusions():
    _, _, w, base, _ = toy()
    items = source_items()
    a, tr = f.fit(
        items, [0, 1, 2], "neuroticism", base, base["anchor"], float(w["scale"])
    )
    # A held author's fields are deliberately unreadable: filtering must precede access.
    b, _ = f.fit(
        items + [{"index": 3, "query": "neuroticism"}],
        [0, 1, 2],
        "neuroticism",
        base,
        base["anchor"],
        float(w["scale"]),
    )
    for name in f.OBJECTIVES:
        np.testing.assert_array_equal(a[name]["coef"], b[name]["coef"])
        np.testing.assert_array_equal(a[name]["coef"][-3], 0)
        np.testing.assert_array_equal(a[name]["anchor"], base["anchor"])
        assert tr[name]["normal_error"] < 1e-10
    assert tr["logit_only"]["x_loss"] < 1
    assert tr["joint"]["x_loss"] + tr["joint"]["d_loss"] < 2
    with pytest.raises(ValueError, match="Exact permitted"):
        f.fit(items, [0, 1], "neuroticism", base, base["anchor"], float(w["scale"]))


def test_gauge_design_derivative_matches_masked_sequence_centering():
    data, d, _, base, _ = toy()
    z, dz = f.design(data, d, base)
    np.testing.assert_allclose(z[1:].mean(0), 0, atol=1e-15)
    np.testing.assert_allclose(dz[:, 1:].mean(1), 0, atol=1e-15)
    np.testing.assert_array_equal(z[:, -3], 0)
    for j in range(4):
        plus = f.design(dict(data, state=data["state"] + 1e-5 * d[j]), d, base)[0]
        minus = f.design(dict(data, state=data["state"] - 1e-5 * d[j]), d, base)[0]
        np.testing.assert_allclose((plus - minus) / 2e-5, dz[j], atol=1e-9)


def test_target_guard_in_fresh_process(tmp_path):
    target = tmp_path / "native_keys.npz"
    target.write_bytes(b"forbidden")
    code = f"""from grain_research.vector_mechanisms.stage_file_guard import install
denied=install([], [], [{str(target)!r}], [])
assert len(denied)==1
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
