#!/usr/bin/env python3
"""Finite readout, grouping and aggregation tests without private data."""

import numpy as np
import pandas as pd
import pytest

from grain_research.vector_mechanisms.scalar_component import (
    SmallReadout,
    construct_oof,
)


def test_weights_make_repeated_views_equal_to_author_loss():
    x = np.arange(20, dtype=float)[:, None]
    y = np.sin(x[:, 0])
    a = SmallReadout("cubic").fit(x, y)
    b = SmallReadout("cubic").fit(
        np.repeat(x, 3, 0), np.repeat(y, 3), np.full(60, 1 / 3)
    )
    np.testing.assert_allclose(a.predict(x), b.predict(x), atol=1e-13)


def test_geometry_and_prediction_do_not_refit_on_held_values():
    x = np.arange(20, dtype=float)[:, None]
    m = SmallReadout("affine").fit(x, x[:, 0])
    state = m.state()
    m.predict([[10000]])
    assert m.state() == state
    assert m.extrapolated([[10000], [10]]).tolist() == [True, False]


def test_scalar_cubic_can_recover_an_orthogonal_scalar_function():
    x = np.linspace(-1, 1, 41)[:, None]
    y = x[:, 0] ** 2
    a = SmallReadout("affine").fit(x, y)
    b = SmallReadout("cubic").fit(x, y)
    assert np.mean((b.predict(x) - y) ** 2) < 0.01 * np.mean((a.predict(x) - y) ** 2)


def test_component_compression_can_hide_a_deterministic_target():
    rng = np.random.default_rng(31)
    x = rng.normal(size=(300, 3))
    y = x[:, 0] - x[:, 1]
    a = SmallReadout("cubic").fit(x.sum(1)[:, None], y)
    b = SmallReadout("components").fit(x, y)
    assert np.mean((b.predict(x) - y) ** 2) < 0.01 * np.mean(
        (a.predict(x.sum(1)[:, None]) - y) ** 2
    )


def test_cubic_jensen_account_and_affine_commutation():
    rng = np.random.default_rng(4)
    grouped = rng.normal(size=(20, 3, 1))
    x = grouped.reshape(-1, 1)
    y = x[:, 0] ** 2 + 0.4 * x[:, 0] ** 3
    cubic = SmallReadout("cubic").fit(x, y)
    parts = cubic.jensen_parts(grouped)
    assert parts["closure_error"].max() < 1e-13
    assert np.mean(np.abs(parts["direct"])) > 0.1
    affine = SmallReadout("affine").fit(x, y)
    assert abs(affine.jensen_parts(grouped)["direct"]).max() < 1e-13


def panel():
    """Balanced synthetic author observations with explicitly provided folds."""
    rng = np.random.default_rng(18)
    rows = []
    for author in range(20):
        for view in range(3):
            x = rng.normal(size=3)
            rows.append(
                {
                    "author_id": str(author),
                    "query": "toy",
                    "view": view,
                    "fold": author % 5,
                    "mean": x[0],
                    "configuration": x[1],
                    "final": x[2],
                    "score": x.sum(),
                    "teacher": x[0] - 0.3 * x[1],
                }
            )
    return pd.DataFrame(rows)


def test_oof_coverage_and_no_held_target_in_its_own_model():
    data = panel()
    predictions, models, _ = construct_oof(data)
    assert len(predictions) == 20 * 3 * 7
    assert len(models) == 5 * 6
    data.loc[data.fold == 0, "teacher"] += 100
    _, changed, _ = construct_oof(data)
    assert [m for m in models if m["fold"] == 0] == [
        m for m in changed if m["fold"] == 0
    ]


def test_reject_criterion_column_and_cross_fold_author():
    data = panel()
    with pytest.raises(ValueError):
        construct_oof(data.assign(y_true=0))
    data.loc[0, "fold"] = 1
    with pytest.raises(ValueError):
        construct_oof(data)


def test_reject_invalid_weights_dimensions_or_incomplete_views():
    x = np.arange(4, dtype=float)[:, None]
    with pytest.raises(ValueError):
        SmallReadout("components").fit(x, x[:, 0])
    with pytest.raises(ValueError):
        SmallReadout("affine").fit(x, x[:, 0], [-1, 1, 1, 1])
    with pytest.raises(ValueError):
        construct_oof(panel().iloc[1:])
