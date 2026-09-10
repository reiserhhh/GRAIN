#!/usr/bin/env python3
"""Source projection geometry, split exclusion and compatible response tests."""

import numpy as np
import pytest

from grain_research.vector_mechanisms import source_response_coordinates as fn
from grain_research.vector_mechanisms import source_selector_potential as scalar
from tests.test_grain_source_selector_potential import source_items, toy


def materials():
    """Paired synthetic queries with no held-out payload dependency."""
    return source_items() + [
        dict(i, query="confirmation_seeking", k=i["k"] * 1.1, dk=i["dk"] * 0.9)
        for i in source_items()
    ]


def test_source_capacity_matches_explicit_least_squares():
    r = np.random.default_rng(75)
    a, y = r.normal(size=(50, 7)), r.normal(size=(50, 3))
    a[:, -1], a[:, -2] = 0, a[:, 0]
    s = {
        "gd": a.T @ a / 50,
        "cd": a.T @ y / 50,
        "rd": np.mean(y * y),
        "heads": 3,
        "gx": np.eye(7),
        "cx": np.zeros((7, 3)),
        "rx": 1.0,
    }
    c = fn.capacity(s, np.zeros((7, 3)))
    explicit = np.linalg.lstsq(a, y, rcond=1e-5)[0]
    ratio = np.sum((y - a @ explicit) ** 2) / np.sum(y * y)
    assert c["derivative_rank"] == 5
    assert c["derivative_minimum_residual_fraction"] == pytest.approx(ratio, abs=1e-12)
    assert c["joint_derivative_excess"] == pytest.approx(1 - ratio)
    assert c["minimum_coef_norm"] == pytest.approx(np.linalg.norm(explicit))


def test_supervised_basis_source_only_and_full_to_reduced_derivative():
    data, d, w, base, _ = toy()
    anchors = {
        q: base["anchor"] * (1 if q == "neuroticism" else 0.8) for q in fn.QUERIES
    }
    items = materials()
    b, account = fn.build_basis(items, [0, 1, 2], anchors, float(w["scale"]), rank=2)
    extra = [{"index": 7, "query": q} for q in fn.QUERIES]
    bp, ap = fn.build_basis(
        items + extra, [0, 1, 2], anchors, float(w["scale"]), rank=2
    )
    for key in b:
        np.testing.assert_array_equal(b[key], bp[key])
    assert ap == account
    u = b["basis"] * b["input_scale"][:, None]
    np.testing.assert_allclose(u.T @ u, np.eye(2), atol=1e-12)
    phi, dp = scalar.design(data, d, b)
    assert phi.shape == (13, 7)
    for f in range(4):
        plus = scalar.design(dict(data, state=data["state"] + 1e-5 * d[f]), d, b)[0]
        minus = scalar.design(dict(data, state=data["state"] - 1e-5 * d[f]), d, b)[0]
        np.testing.assert_allclose((plus - minus) / 2e-5, dp[f], atol=1e-9)


def test_new_joint_map_scalar_derivative_and_fixed_source_losses():
    data, d, w, old, v = toy()
    anchors = {q: old["anchor"] for q in fn.QUERIES}
    items = materials()
    b, _ = fn.build_basis(items, [0, 1, 2], anchors, float(w["scale"]), rank=2)
    model, tr = fn.fit_joint(
        items, [0, 1, 2], "neuroticism", b, anchors["neuroticism"], float(w["scale"])
    )
    assert tr["normal_error"] < 1e-10
    np.testing.assert_array_equal(model["coef"][-3], 0)
    ell, dell = scalar.potential(data, d, w, model)
    action = scalar.readout(ell, dell, data["mask"], v)["actions"]
    for f in range(4):
        values = []
        for eps in (-1e-5, 1e-5):
            el, dl = scalar.potential(
                dict(data, state=data["state"] + eps * d[f]), d, w, model
            )
            values.append(scalar.readout(el, dl, data["mask"], v)["scalar"])
        np.testing.assert_allclose(
            (values[1] - values[0]) / 2e-5, action[:, f], atol=1e-8
        )


def test_unsupported_rank_and_incomplete_pair_rejected():
    _, _, w, b, _ = toy()
    anchors = {q: b["anchor"] for q in fn.QUERIES}
    with pytest.raises(ValueError, match="unsupported"):
        fn.build_basis(materials(), [0, 1, 2], anchors, float(w["scale"]), rank=32)
    with pytest.raises(ValueError, match="Exact source-query"):
        fn.build_basis(materials()[:-1], [0, 1, 2], anchors, float(w["scale"]), rank=2)
