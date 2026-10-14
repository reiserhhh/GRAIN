#!/usr/bin/env python3
"""Cycle sufficiency and attainable reading bounds on complete finite support."""

import subprocess
import sys

import numpy as np
import pytest

from grain_research.construction import math_policy_cycles as M


def test_arbitrary_tables_reconstruct_without_outcome_fitting():
    x = np.random.default_rng(1184).uniform(0, 9, (30, 12, 2, 2))
    c = M.coordinates(x)
    np.testing.assert_allclose(
        M.reconstruct(c["rows"], c["columns"], c["coordinate"]), x
    )
    residual = x - c["independent"]
    np.testing.assert_allclose(residual.sum(axis=-1), 0, atol=1e-14)
    np.testing.assert_allclose(residual.sum(axis=-2), 0, atol=1e-14)


def test_noninteger_reference_and_both_sharp_coordinate_endpoints():
    x = np.array([[1, 2], [2, 5]])
    c = M.coordinates(x)
    assert c["independent"][0, 0] == pytest.approx(0.9)
    for endpoint in ("lower_coordinate", "upper_coordinate"):
        witness = M.reconstruct(c["rows"], c["columns"], c[endpoint])
        assert witness.min() == pytest.approx(0)
        np.testing.assert_allclose(witness.sum(axis=-1), c["rows"])
        np.testing.assert_allclose(witness.sum(axis=-2), c["columns"])
        np.testing.assert_allclose(witness, np.rint(witness), atol=1e-14)


def test_interaction_is_observable_but_margins_cannot_identify_it():
    x = np.full((1, 2, 2), 4)
    k = M.CYCLE[None]
    neutral = M.response_account(x, k)
    positive = M.response_account(x + M.CYCLE[None], k)
    assert positive["observed"] == pytest.approx(0.25)
    assert positive["independent"] == neutral["independent"] == pytest.approx(0)
    assert positive["lower"] == pytest.approx(-1)
    assert positive["upper"] == pytest.approx(1)


def test_additive_kernel_has_no_cycle_effect_or_identification_width():
    x = np.array([[[1, 3], [2, 5]], [[0, 4], [0, 6]]])
    row = np.array([1.2, -3.7])[None, :, None]
    col = np.array([2.1, 6.5])[None, None, :]
    k = np.broadcast_to(row + col, x.shape)
    out = M.response_account(x, k)
    np.testing.assert_allclose(out["gamma"], 0, atol=1e-14)
    np.testing.assert_allclose(out["cycle"], 0, atol=1e-14)
    np.testing.assert_allclose(out["lower"], out["upper"])


def test_bounds_are_per_output_not_one_common_maximizing_policy():
    x = np.full((2, 2, 2), 3)
    k = np.broadcast_to(M.CYCLE[None, :, :, None], (2, 2, 2, 2)).copy()
    k[..., 1] *= -1
    out = M.response_account(x, k)
    np.testing.assert_allclose(out["upper"], [1, 1])
    assert np.all(out["upper_witness"][:, 0] == -out["upper_witness"][:, 1])
    c = M.coordinates(x)
    for q in range(2):
        witness = M.reconstruct(c["rows"], c["columns"], out["upper_witness"][:, q])
        actual = (witness * k[..., q]).sum() / x.sum()
        assert actual == pytest.approx(out["upper"][q])


def test_missing_groups_retained_without_fabricated_conditional_score():
    x = np.array([[[0, 0], [0, 0]], [[1, 2], [3, 4]]])
    c = M.coordinates(x)
    assert not c["observed"][0]
    assert (
        c["coordinate"][0] == c["lower_coordinate"][0] == c["upper_coordinate"][0] == 0
    )
    out = M.response_account(x, np.ones_like(x))
    assert not out["coverage"][0]
    assert out["observed"] == 1
    with pytest.raises(ValueError):
        M.response_account(np.zeros_like(x), np.ones_like(x))


@pytest.mark.parametrize(
    "bad", ["negative", "nan", "shape", "different_mass", "outside"]
)
def test_invalid_coordinates_or_margins_fail_closed(bad):
    x = np.full((1, 2, 2), 2.0)
    if bad in ("negative", "nan", "shape"):
        if bad == "negative":
            x[0, 0, 0] = -1
        elif bad == "nan":
            x[0, 0, 0] = np.nan
        else:
            x = x[:, 0]
        with pytest.raises(ValueError):
            M.coordinates(x)
    else:
        c = M.coordinates(x)
        if bad == "different_mass":
            c["columns"][0, 0] += 1
        else:
            c["coordinate"] += 3
        with pytest.raises(ValueError):
            M.reconstruct(c["rows"], c["columns"], c["coordinate"])


def test_random_output_bounds_contain_every_feasible_common_policy():
    rng = np.random.default_rng(1185)
    x, k = rng.integers(1, 20, (12, 2, 2)), rng.normal(size=(12, 2, 2, 6, 4))
    c, out = M.coordinates(x), M.response_account(x, k)
    for _ in range(15):
        value = c["lower_coordinate"] + rng.uniform(size=12) * (
            c["upper_coordinate"] - c["lower_coordinate"]
        )
        witness = M.reconstruct(c["rows"], c["columns"], value)
        actual = (witness[..., None, None] * k).sum(axis=(0, 1, 2)) / x.sum()
        assert np.all(actual >= out["lower"] - 1e-12)
        assert np.all(actual <= out["upper"] + 1e-12)


def test_numeric_core_does_not_load_neural_runtime():
    command = [
        sys.executable,
        "-c",
        (
            "import sys;from grain_research.construction import math_policy_cycles;"
            "assert not {'torch','transformers'}.intersection(sys.modules)"
        ),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
