#!/usr/bin/env python3
"""Synthetic data checks for the expanded, phase-aware factor program."""

import numpy as np
import pandas as pd
import pytest

from grain_research.vector_mechanisms.factor_catalog import (
    Scale,
    association,
    audit_scored_meps,
    bh_adjust,
    catalog,
    recompute,
    scales,
)
from scripts.analyze_grain_all_factor_inventory import linked_matrix


def test_catalog_exact_core_and_scale_coverage():
    c = catalog()
    assert len(c["core"]) == 9
    assert {x["positive"] for x in c["core"] if x["kind"] == "mbti_axis"} == {
        "E",
        "N",
        "F",
        "J",
    }
    assert len(scales()) == len({s.name for s in scales()}) == 22
    assert {s.phase for s in scales()} == {"pre", "post"}
    assert not any(s.reliability for s in scales() if "disclosability" in s.name)


def test_selected_reverse_positions_and_missingness():
    spec = next(s for s in scales() if s.name == "post_session_depth")
    frame = pd.DataFrame([[1, 2, 3, 4, 5], [1, np.nan, 3, 4, 5]], columns=spec.items)
    score, items = recompute(frame, spec)
    np.testing.assert_allclose(items.iloc[0], [7, 2, 5, 4, 3])
    assert score.iloc[0] == 4.2
    assert np.isnan(score.iloc[1])


def test_zero_to_ten_not_reverse_one_to_seven():
    score, _ = recompute(
        pd.DataFrame({"a": [0, 10]}), Scale("pre_a", "q", ("a",), 0, 10)
    )
    assert score.tolist() == [0, 10]
    with pytest.raises(ValueError, match="range"):
        recompute(pd.DataFrame({"a": [11]}), Scale("pre_a", "q", ("a",), 0, 10))


def test_auth_gate_precedes_any_file_read(tmp_path):
    with pytest.raises(PermissionError):
        audit_scored_meps(tmp_path)


def test_association_constants_and_fisher_bounds():
    assert association([1] * 10, np.arange(10))["status"] == "constant_or_small"
    x, y = np.arange(10), np.array([0, 1, 3, 2, 4, 7, 6, 5, 9, 8])
    result = association(x, y)
    assert -1 <= result["r_lower"] < result["pearson_r"] < result["r_upper"] <= 1
    assert association(x, [0] * 5 + [1] * 5, binary=True)["auc"] == 1
    with pytest.raises(ValueError):
        association(x, y, binary=True)


def test_bh_keeps_order_and_missing():
    np.testing.assert_allclose(
        bh_adjust([0.04, np.nan, 0.001, 0.02]),
        [0.04, np.nan, 0.003, 0.03],
        equal_nan=True,
    )


def test_cross_query_join_is_by_author_not_row():
    rows = []
    for u in range(6):
        rows += [
            {"author_id": str(u), "trait": "a", "prediction": u, "y_true": u},
            {"author_id": str(u), "trait": "b", "prediction": -u, "y_true": -u},
        ]
    frame = pd.DataFrame(rows).sample(frac=1, random_state=3)
    result, _ = linked_matrix(frame, "big5", "native", "raw")
    off = next(x for x in result if x["query"] == "a" and x["criterion"] == "b")
    assert off["n"] == 6 and off["pearson_r"] == pytest.approx(-1)
    with pytest.raises(ValueError, match="Duplicate"):
        linked_matrix(pd.concat([frame, frame.iloc[:1]]), "big5", "native", "raw")
