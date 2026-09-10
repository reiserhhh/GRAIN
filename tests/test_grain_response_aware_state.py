#!/usr/bin/env python3
"""Frozen features, weighted normal equations and same-function response checks."""

import copy

import numpy as np
import pytest

from grain_research.vector_mechanisms.compiled_query_transfer import QUERIES
from grain_research.vector_mechanisms.response_aware_state import fit_response_maps
from grain_research.vector_mechanisms.source_state_transport import (
    apply_map,
    features,
    fit_maps,
)


def problem(mode):
    """Unequal token counts and causal source targets, independent of query labels."""
    rng = np.random.default_rng(53)
    items = []
    for i in range(4):
        for q in QUERIES:
            x = rng.normal(size=(8 + i, 7))
            d = rng.normal(size=(4, *x.shape))
            y = x + 0.25 * np.cumsum(x, axis=0) / np.arange(1, len(x) + 1)[:, None]
            dy = (
                d
                + 0.25 * np.cumsum(d, axis=1) / np.arange(1, len(x) + 1)[None, :, None]
            )
            items.append({"index": i, "query": q, "x": x, "y": y, "d": d, "dy": dy})
    model = fit_maps(items, [0, 1, 2], rank=3)[mode]
    return items, model


@pytest.mark.parametrize("mode", ["pointwise", "causal_context"])
def test_stacked_normal_equation_and_fixed_assets(mode):
    items, base = problem(mode)
    maps, train, stats = fit_response_maps(items, [0, 1, 2], base)
    for objective, model in maps.items():
        rows, targets = [], []
        for item in items[:6]:
            x, y, d, dy = (item[k] for k in ("x", "y", "d", "dy"))
            p, dp = features(x, d, base["mean"], base["basis"], mode)
            z, dz = (p - base["center"]) / base["scale"], dp / base["scale"]
            wx = np.sqrt(1 / (6 * len(x) * 7 * stats["rx"]))
            rows.append(wx * z)
            targets.append(wx * (y - x))
            if objective == "joint":
                wd = np.sqrt(1 / (24 * len(x) * 7 * stats["rd"]))
                for f in range(4):
                    rows.append(wd * dz[f])
                    targets.append(wd * (dy[f] - d[f]))
        z, y = np.concatenate(rows), np.concatenate(targets)
        penalty = np.full(z.shape[1], 0.01)
        penalty[-3] = 0
        expected = np.linalg.solve(z.T @ z + np.diag(penalty), z.T @ y)
        np.testing.assert_allclose(model["coef"], expected, atol=1e-12)
        for field in set(base) - {"coef"}:
            np.testing.assert_array_equal(model[field], base[field])
        assert train[objective]["normal_equation_max_error"] < 1e-12
        value = train[objective]["x"] + train[objective]["penalty"]
        if objective == "normalized_value":
            assert value <= train["joint"]["x"] + train["joint"]["penalty"] + 1e-12
    assert (
        sum(train["joint"][k] for k in ("x", "d", "penalty"))
        <= sum(train["normalized_value"][k] for k in ("x", "d", "penalty")) + 1e-12
    )


@pytest.mark.parametrize("mode", ["pointwise", "causal_context"])
def test_joint_is_same_function_and_basis_null_is_identity(mode):
    items, base = problem(mode)
    m = fit_response_maps(items, [0, 1, 2], base)[0]["joint"]
    x, d = items[-1]["x"], items[-1]["d"]
    _, dd, _, _ = apply_map(x, d, m)
    eps = 1e-5
    for f in range(4):
        plus = apply_map(x + eps * d[f], d, m)[0]
        minus = apply_map(x - eps * d[f], d, m)[0]
        np.testing.assert_allclose((plus - minus) / (2 * eps), dd[f], atol=1e-10)
    null = d - (d @ m["basis"]) @ m["basis"].T
    np.testing.assert_allclose(apply_map(x, null, m)[1], null, atol=1e-12)


def test_source_exclusion_and_bad_normalizers():
    items, base = problem("causal_context")
    models = fit_response_maps(items, [0, 1, 2], base)[0]
    changed = copy.deepcopy(items)
    for i in changed:
        if i["index"] == 3:
            i["x"] *= 100
            i["dy"] *= -100
    other = fit_response_maps(changed, [0, 1, 2], base)[0]
    for mode in models:
        for field in models[mode]:
            np.testing.assert_array_equal(models[mode][field], other[mode][field])
    for i in changed:
        i["dy"] = i["d"].copy()
    with pytest.raises(ValueError, match="normalizers"):
        fit_response_maps(changed, [0, 1, 2], base)
