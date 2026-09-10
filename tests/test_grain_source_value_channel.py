#!/usr/bin/env python3
"""Value transport, complete derivatives, source exclusion and metric inventory."""

import numpy as np
import pytest

from grain_research.vector_mechanisms import source_value_channel as f
from tests.test_grain_source_selector_potential import toy


def setup(seed=76):
    """Finite grouped-head data and a shared source feature design."""
    data, d, w, sm, value = toy(seed)
    r = np.random.default_rng(seed + 900)
    w.update(v=r.normal(size=w["k"].shape), v_bias=r.normal(size=w["k_bias"].shape))
    sm["center"][-3], sm["scale"][-3] = 0.0, 1.0
    vm = dict(
        sm,
        coef=r.normal(size=(len(sm["center"]), w["v"].shape[0])) * 0.03,
        query=np.asarray("neuroticism"),
    )
    return data, d, w, sm, vm, {"g": value["g"]}


def test_value_projection_and_value_model_derivative():
    data, d, w, _, vm, _ = setup()
    v, dv = f.apply(data, d, w, vm)
    for j in range(4):
        values = [
            f.apply(dict(data, state=data["state"] + e * d[j]), d, w, vm)[0]
            for e in (-1e-5, 1e-5)
        ]
        np.testing.assert_allclose((values[1] - values[0]) / 2e-5, dv[:, j], atol=1e-8)
    assert v.shape[:2] == (2, 13)
    with pytest.raises(ValueError, match="Current query"):
        f.values(data["state"], d, w | {"q": w["k"]})


def test_four_functions_total_derivative_and_bias():
    data, d, w, sm, vm, g = setup()
    p = f.predict(data, d, w, sm, vm, g)
    assert p["actions"].shape == (4, 3, 2, 4)
    np.testing.assert_array_equal(
        p["actions"][:, 2], p["actions"][:, 0] + p["actions"][:, 1]
    )
    assert np.max(abs(p["actions"][:, 1])) > 1e-3
    for j in range(4):
        scores = [
            f.predict(dict(data, state=data["state"] + e * d[j]), d, w, sm, vm, g)[
                "scalar"
            ]
            for e in (-1e-5, 1e-5)
        ]
        np.testing.assert_allclose(
            (scores[1] - scores[0]) / 2e-5, p["actions"][:, 2, :, j], atol=1e-8
        )
    assert np.max(abs(p["bias_actions"])) < 1e-12
    assert np.max(abs(p["bias_scalar"])) > 1e-5
    with pytest.raises(ValueError, match="g-only"):
        f.predict(data, d, w, sm, vm, g | {"v": p["v"][0]})


def test_source_only_solve_and_unpenalized_constant():
    _, _, _, base, _, _ = setup()
    items = []
    for i in range(3):
        data, d, w, _, _, _ = setup(76 + i)
        v, dv = f.values(data["state"], d, w)
        items.append(
            {
                "index": i,
                "query": "neuroticism",
                "data": data,
                "d": d,
                "v0": v,
                "dv0": dv,
                "v": 1.1 * v + 0.4,
                "dv": 1.15 * dv,
            }
        )
    a, la = f.fit(items, [0, 1, 2], "neuroticism", base)
    b, lb = f.fit(
        items + [{"index": 7, "query": "neuroticism"}], [0, 1, 2], "neuroticism", base
    )
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])
    assert la == lb
    assert la["normal_error"] < 1e-10
    assert la["bias_column_norm"] > 0
    with pytest.raises(ValueError, match="Exact source"):
        f.fit(items[:-1], [0, 1, 2], "neuroticism", base)


def test_content_constant_leaves_key_channel_invariant():
    data, d, w, sm, vm, reader = setup()
    ell, dell = f.selector(data, d, w, sm, "response_coordinates")
    v, dv = f.apply(data, d, w, vm)
    a = f.readout(ell, dell, data["mask"], v[1], dv[1], reader)
    b = f.readout(ell, dell, data["mask"], v[1] + 0.25, dv[1], reader)
    np.testing.assert_allclose(a["actions"], b["actions"], atol=1e-12)
    assert np.max(abs(a["scalar"] - b["scalar"])) > 1e-3


def test_full_metric_inventory_and_components():
    from scripts.run_grain_source_value_channel import ALL_ARMS, summarize

    r = np.random.default_rng(761)
    groups, metrics = {}, []
    for cohort, n in (("source4", 4), ("new8", 8)):
        groups[cohort] = []
        for _ in range(n):
            a = r.normal(size=(2, len(ALL_ARMS), 3, 2, 4))
            a[:, :, 2] = a[:, :, 0] + a[:, :, 1]
            y = r.normal(size=(2, 2, 3, 2, 4))
            y[:, :, 2] = y[:, :, 0] + y[:, :, 1]
            groups[cohort].append(
                (a, r.normal(size=(2, len(ALL_ARMS), 2)), y, r.normal(size=(2, 2, 2)))
            )
        for q in ("neuroticism", "confirmation_seeking"):
            for v in f.VALUES:
                metrics.append(
                    {"cohort": cohort, "query": q, "value_arm": v, "value_mse": 1.0}
                )
    tables = summarize(groups, metrics)
    assert [len(t) for t in tables] == [8, 336, 1344, 168, 168, 84]
    assert set(tables[1].component) == {"key", "value", "total"}
    assert not any({"author_key", "text", "token"} & set(t.columns) for t in tables)
