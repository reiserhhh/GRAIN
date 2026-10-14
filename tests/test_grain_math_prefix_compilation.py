#!/usr/bin/env python3
"""Synthetic checks for source-only prefix compilation and complete action contrasts."""

import numpy as np
import pytest

from grain_research.construction import math_prefix_compilation as M
from grain_research.construction.math_span_binding import selected_masks
from scripts.run_grain_math_prefix_compilation import prepare


def test_prefix_features_never_access_query_endpoint():
    class NoQuery(dict):
        def __getitem__(self, key):
            assert not key.endswith("_Q")
            return super().__getitem__(key)

    arrays = NoQuery(
        {f"L16_{name}": np.array([[i, i + 1, 99]]) for i, name in enumerate(M.SPANS)}
    )
    spans = {name: {"positions": [i + 1]} for i, name in enumerate(M.SPANS)}
    values, order = M.prefix_coordinates(arrays, spans, np.eye(3)[:, :2], 10)
    np.testing.assert_equal(values.reshape(6, 2), np.arange(6)[:, None] + [0, 1])
    assert values.shape == (12,) and order == 0
    spans["T_property"]["positions"] = [0]
    assert M.prefix_coordinates(arrays, spans, np.eye(3)[:, :2], 10)[1] == 1
    spans["A_property"]["positions"] = [10]
    with pytest.raises(ValueError, match="precede"):
        M.prefix_coordinates(arrays, spans, np.eye(3)[:, :2], 10)


def test_missing_or_nonorthogonal_prefix_fail_closed():
    with pytest.raises(ValueError, match="orthonormal"):
        M.prefix_coordinates({}, {}, np.ones((3, 2)), 10)
    with pytest.raises(ValueError, match="six complete"):
        M.prefix_coordinates({}, {}, np.eye(3)[:, :2], 10)


def test_four_fixed_views_and_complete_degree_two():
    x = np.arange(36, dtype=float).reshape(3, 12)
    order = np.array([0, 1, 0])
    assert [M.feature_map(x, order, arm)[0].shape[1] for arm in M.ARMS] == [
        0,
        4,
        12,
        20,
    ]
    phi, names = M.feature_map(x, order, "property_order_quadratic")
    for i in range(5, len(names)):
        left, right = names[i].split("*")
        np.testing.assert_equal(
            phi[:, i], phi[:, names.index(left)] * phi[:, names.index(right)]
        )
    assert not any("owner" in name or "family" in name for name in names)
    with pytest.raises(ValueError, match="binary"):
        M.feature_map(x, [0, 2, 1], "property_linear")


def test_source_reference_stays_fixed_and_raw_formula_replays():
    x = np.arange(20, dtype=float).reshape(10, 2)
    y = np.column_stack((x[:, 0] * 0.2, x[:, 1] ** 2))
    fn = M.compile_source(x[:6], y[:6])
    before = M.apply_function(x, fn)
    raw = x @ np.asarray(fn["raw_coefficient"]) + fn["raw_intercept"]
    np.testing.assert_allclose(before, raw)
    np.testing.assert_equal(fn["mean"], x[:6].mean(axis=0))
    altered = x.copy()
    altered[6:] = 1e8
    np.testing.assert_equal(M.apply_function(altered, fn)[:6], before[:6])
    assert fn["source_rows"] == 6


def test_constant_column_not_invented_when_application_changes():
    x = np.column_stack((np.arange(10), np.ones(10)))
    fn = M.compile_source(x, np.arange(10)[:, None])
    assert fn["active"] == [True, False]
    changed = x.copy()
    changed[:, 1] = 500
    np.testing.assert_equal(M.apply_function(x, fn), M.apply_function(changed, fn))
    with pytest.raises(ValueError, match="complete"):
        M.apply_function(x[:, :1], fn)


def test_constant_arm_can_run_with_zero_feature_columns():
    fn = M.compile_source(np.empty((6, 0)), np.arange(6)[:, None])
    np.testing.assert_equal(
        M.apply_function(np.empty((4, 0)), fn), np.full((4, 1), 2.5)
    )
    assert M.compare_values(np.arange(4), [2.5] * 4)["r"] is None


def test_all_owner_polarity_addresses_sign_and_missing_corner():
    meta = [{"family": 0, "role": "source", "mask": m} for m in selected_masks()]
    y = np.array([[-2 * ((r["mask"] >> 0) & 1) * ((r["mask"] >> 2) & 1)] for r in meta])
    rows = M.owner_contrasts(meta, y)
    assert len(rows) == 4
    assert [r["values"][0] for r in rows if r["axis"] == "A"] == [2, 2]
    with pytest.raises(ValueError, match="corners"):
        M.owner_contrasts(meta[:-1], y[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        M.owner_contrasts(meta + [meta[0]], np.vstack((y, y[0])))


def test_packet_never_overwrites_prior_artifacts(tmp_path):
    with pytest.raises(FileExistsError, match="preserve"):
        prepare(tmp_path)
