#!/usr/bin/env python3
"""Analytic Gaussian operators, shared BODY features and anchored/diagonal calculus."""

import numpy as np

from grain_research.vector_mechanisms import material_reader_field as f


def test_gaussian_first_and_mixed_derivatives():
    rng = np.random.default_rng(820)
    x, z, v, w = rng.normal(size=(4, 1, 5))
    val = lambda a, b: f.gram(
        a, np.zeros_like(a), [False], b, np.zeros_like(b), [False]
    )
    eps = 1e-4
    dx = (val(x + eps * v, z) - val(x - eps * v, z)) / (2 * eps)
    dz = (val(x, z + eps * w) - val(x, z - eps * w)) / (2 * eps)
    mix = (
        val(x + eps * v, z + eps * w)
        - val(x + eps * v, z - eps * w)
        - val(x - eps * v, z + eps * w)
        + val(x - eps * v, z - eps * w)
    ) / (4 * eps**2)
    np.testing.assert_allclose(dx, f.gram(x, v, [True], z, w, [False]), atol=1e-7)
    np.testing.assert_allclose(dz, f.gram(x, v, [False], z, w, [True]), atol=1e-7)
    np.testing.assert_allclose(mix, f.gram(x, v, [True], z, w, [True]), atol=1e-7)


def toy():
    rng = np.random.default_rng(821)
    x, v = rng.normal(size=(2, 9, 5))
    kinds = np.array([False] * 3 + [True] * 6)
    groups = np.repeat(np.arange(3), 3)
    y = rng.normal(size=(9, 7))
    model = f.fit(x, v, kinds, y, groups, np.zeros(7), rng.normal(size=7))
    return rng, model


def test_gram_psd_and_field_derivative():
    rng, m = toy()
    k = f.gram(m["x"], m["v"], m["kinds"], m["x"], m["v"], m["kinds"])
    assert np.linalg.eigvalsh(k).min() > -1e-10
    x, dx = rng.normal(size=(2, 5))
    d, dd = f.field(m, x, dx)
    eps = 1e-5
    np.testing.assert_allclose(
        (f.field(m, x + eps * dx) - f.field(m, x - eps * dx)) / (2 * eps),
        dd[0],
        atol=1e-8,
    )
    p, dp = rng.normal(size=(2, 7))
    anchored = (d @ (p + eps * dp) - d @ (p - eps * dp)) / (2 * eps)
    diagonal = (
        f.field(m, x + eps * dx) @ (p + eps * dp)
        - f.field(m, x - eps * dx) @ (p - eps * dp)
    ) / (2 * eps)
    np.testing.assert_allclose(anchored, d @ dp, atol=1e-8)
    np.testing.assert_allclose(diagonal, d @ dp + dd[0] @ p, atol=1e-8)


def test_zero_operator_and_group_weighting():
    rng, m = toy()
    v = m["v"].copy()
    v[3:6] = 0
    y = rng.normal(size=(9, 7))
    groups = np.repeat(np.arange(3), 3)
    a = f.fit(m["x"], v, m["kinds"], y, groups, m["d0"], m["m"])
    assert a["zero_operator_rows"] == 3
    np.testing.assert_allclose(a["group_loss"][1], 1)
    w, _ = f.observation_weights(y, groups)
    for gi in range(3):
        np.testing.assert_allclose(
            np.sum((w[groups == gi, None] * y[groups == gi]) ** 2), 1
        )


def test_feature_derivatives_and_nested_zero_body():
    rng = np.random.default_rng(822)
    h, g = rng.normal(size=(7, 5)), rng.normal(size=(4, 4))
    pos = np.array([0, 1, 3, 4])
    data = [{"state": h.copy(), "positions": pos, "grain": g.copy()} for _ in range(2)]
    data[1]["state"][-1] += 1
    directions = [np.zeros((2, 7, 5)) for _ in range(2)]
    d = rng.normal(size=(2, 4, 5))
    for q in range(2):
        directions[q][:, pos] = d
    _b, db = f.features(data, directions)
    eps = 1e-5
    changed = [
        [
            dict(x, state=x["state"] + e * dx[0])
            for x, dx in zip(data, directions, strict=True)
        ]
        for e in (-eps, eps)
    ]
    for ai, bi, di in zip(
        f.features(changed[0]), f.features(changed[1]), db, strict=True
    ):
        np.testing.assert_allclose((bi - ai) / (2 * eps), di[0], atol=1e-9)
    assert not f.flatten(db, np.ones(6), "context_field").any()
    assert f.flatten(db, np.ones(6), "grain_field").any()


def test_full_comparison_inventory_and_diagonal_identity():
    from scripts.run_grain_material_reader_field import COUNTS, projections, summarize

    rng = np.random.default_rng(823)
    rows = []
    for _ in range(8):
        primal = rng.normal(size=(2, 4, 7))
        vectors = rng.normal(size=(2, 4, 3, 4, 7))
        readers = rng.normal(size=(6, 2, 7))
        dr = rng.normal(size=(6, 2, 4, 7))
        dr[:3] = 0
        ps = [projections(primal, vectors, r, d) for r, d in zip(readers, dr, strict=True)]
        p = {k: np.stack([v[k] for v in ps]) for k in ps[0]}
        p.update(
            readers=readers,
            reader_derivatives=dr,
            native_r=rng.normal(size=(2, 7)),
            native_primal=rng.normal(size=(2, 7)),
            native_vectors=rng.normal(size=(2, 3, 4, 7)),
            target_scalar=rng.normal(size=(2, 2)),
            target_actions=rng.normal(size=(2, 3, 2, 4)),
            support=np.ones((3, 3)),
        )
        rows.append(p)
    _, m = toy()
    m.update(groups=np.repeat(np.arange(3), 3), scale=np.ones(6))
    tables = summarize(rows, [m] * 3)
    assert [len(t) for t in tables] == COUNTS
    assert (tables[0].role == "source_training").all()
    assert (tables[7].query == "N_minus_confirmation").any()
