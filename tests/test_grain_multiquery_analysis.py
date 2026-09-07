#!/usr/bin/env python3
"""Tests for response geometry and removal of edit-cell main effects."""

import numpy as np
import pandas as pd

from scripts.analyze_grain_multiquery_sensitivity import (
    balanced_projection,
    corr,
    paired_shared,
)
from scripts.run_grain_multiquery_sensitivity import QUERIES


def test_constant_is_not_a_perfect_correlation():
    assert corr([1, 1, 1], [1, 1, 1]) is None
    assert np.isclose(corr([1, 2, 3], [-1, -2, -3]), -1)


def test_shared_edit_main_effect_is_not_author_shared_information():
    rows = []
    for author in range(4):
        for view in range(2):
            for edit, effect in (("e1", 1.0), ("e2", -1.0)):
                for q in QUERIES:
                    rows.append(
                        {
                            "author_id": str(author),
                            "view": view,
                            "edit": edit,
                            "query": q,
                            "natural": effect,
                            "patch": effect,
                            "linear": effect,
                        }
                    )
    result = paired_shared(pd.DataFrame(rows))
    assert np.allclose(result.loc[result["mode"] == "raw", "r"], 1)
    assert result.loc[result["mode"] == "edit_view_centered", "r"].isna().all()


def test_balanced_oblique_projection_is_idempotent_and_preserves_function():
    d = np.array([[100, 1], [100, -1], [-100, 1], [-100, -1]], float)
    g = np.tile([0.0, 1.0], (4, 1))
    v, w = balanced_projection(d, g, "balanced", 1)
    assert np.allclose(w.T @ v, np.eye(1))
    p = v @ w.T
    assert np.allclose(p @ p, p)
    assert np.allclose(np.sum((d @ w) * (g @ v), axis=1), np.sum(d * g, axis=1))
