#!/usr/bin/env python3
import numpy as np
import pytest
from grain_core.liquid_frozen_bank import mix
from grain_core.liquid_reference_error import (
    calibration_member,
    endpoint_weights,
    fit_chart,
    locate,
    matched_endpoints,
    population_io,
    resample_runs,
)


def test_reference_ties_invariance_and_basis():
    args = {"axes": ["a", "b"], "basis": "fixed"}
    ref = fit_chart([[0, 2], [0, 4], [1, 0], [1, 5]], population="example", **args)
    p, d = locate(ref, [[0, 2], [10, 9]], **args)
    np.testing.assert_allclose(p[0], [0.25, 0.375])
    np.testing.assert_array_equal(locate(ref, [[0, 2]], **args)[1], d[:1])
    with pytest.raises(ValueError):
        locate(ref, [[0, 2]], axes=["b", "a"], basis="fixed")
    with pytest.raises(ValueError):
        locate(ref, [[0, 2]], axes=["a", "b"], basis="unrelated")


def test_cell_transport_and_same_survival():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(6, 8, 4))
    r = rng.uniform(0.1, 0.9, size=(6, 8, 24))
    z, c = matched_endpoints(x, r)
    w = endpoint_weights(r)
    np.testing.assert_allclose(w.sum(1), 1)
    np.testing.assert_allclose(np.einsum("ne,ned->nd", w, x), z)
    for i in range(6):
        np.testing.assert_allclose(mix(x[i], r[i])[-1], z[i])
    impulse = np.zeros_like(x)
    impulse[:, 0] = 1
    aa, bb = matched_endpoints(impulse, r)
    np.testing.assert_allclose(aa, bb)
    assert np.max(abs(z - c)) > 0.01
    a, b = matched_endpoints(x, np.full((6, 8), 0.34))
    np.testing.assert_allclose(a, b)
    with pytest.raises(ValueError):
        matched_endpoints(x, r[:, :7])


def test_run_resampling_and_reference_split():
    blocks = [[0, 1], [2], [4, 5, 6]]
    for seed in range(5):
        selected = resample_runs(blocks, np.random.default_rng(seed))
        assert len(selected) == len(blocks) and all(r in blocks for r in selected)
    idx = list(map(str, range(40)))
    members = {i for i in idx if calibration_member(i, "population")}
    assert 0 < len(members) < len(idx)
    assert members == {i for i in reversed(idx) if calibration_member(i, "population")}


def test_external_units_must_be_independent():
    doc = {
        "basis": "frozen-A",
        "axes": ["a"],
        "population": "synthetic-example",
        "namespace": "example",
        "units": [{"id": str(j), "values": [j]} for j in range(4)],
    }
    chart = population_io("fit", doc)
    with pytest.raises(ValueError, match="calibration unit"):
        population_io("locate", doc, chart=chart)
    new = {**doc, "units": [{"id": "new", "values": [1.5]}]}
    assert population_io("locate", new, chart=chart)["units"][0]["percentiles_0_1"] == [
        0.5
    ]
    with pytest.raises(ValueError, match="namespace"):
        population_io("locate", {**new, "namespace": "other"}, chart=chart)
