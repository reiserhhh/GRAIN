#!/usr/bin/env python3
"""Small synthetic tests for author-safe semantic-affinity links."""

import joblib
import numpy as np
import pandas as pd
import pytest

from grain_research.semantic_link import (
    SemanticLink,
    affinity_targets,
    author_mean,
    author_weights,
    opportunity_numeric,
    paired_author_mse,
    validate_addresses,
    validate_reference_exposure,
    weighted_metrics,
)


def test_equal_author_weight_and_mean():
    ids = np.array([1, 1, 1, 4])
    w = author_weights(ids)
    assert w[ids == 1].sum() == pytest.approx(w[ids == 4].sum())
    ids, m = author_mean(np.arange(4)[:, None], ids)
    np.testing.assert_array_equal(ids, [1, 4])
    np.testing.assert_allclose(m[:, 0], [1, 3])


def test_repeated_budget_arms_are_not_duplicate_authors():
    f = pd.DataFrame(
        {
            "author_id": ["a", "a", "b", "b"],
            "old_reference_member": [False] * 4,
            "old_e1_member": [False] * 4,
        }
    )
    validate_reference_exposure(f, ["a", "b"])
    with pytest.raises(ValueError, match="Missing"):
        validate_reference_exposure(f, ["a", "b", "c"])
    f.loc[1, "old_e1_member"] = True
    with pytest.raises(ValueError, match="overlap"):
        validate_reference_exposure(f, ["a", "b"])


def test_affinity_key_direction_not_probability():
    q = np.array([[1.0, 0], [-1.0, 0]])
    items = [{"facet": "A1", "key": 1}, {"facet": "A1", "key": -1}]
    y, names = affinity_targets(q, q, items)
    np.testing.assert_allclose(y, [[2, 2], [-2, -2]])
    assert names == ["mean:A1", "max:A1"]
    with pytest.raises(ValueError, match="both keyed"):
        affinity_targets(q, q[:1], items[:1])
    with pytest.raises(ValueError, match="Zero vector"):
        affinity_targets(np.zeros((1, 2)), q, items)


def test_mean_and_max_are_distinct_functions():
    q = np.array([[1.0, 0], [0.0, 1], [-1.0, 0]])
    items = [{"facet": "A1", "key": k} for k in (1, 1, -1)]
    target, _ = affinity_targets(np.array([[1.0, 0]]), q, items)
    np.testing.assert_allclose(target, [[1.5, 2.0]])


def material():
    text = "I like reading."
    return {
        "bundles": [
            {
                "author_id": "x",
                "source_group": "big5_only",
                "views": [[{"body": text, "comment_id": "c"}]],
            }
        ],
        "sentences": [
            {
                "author_index": 0,
                "comment_index": 0,
                "comment_id": "c",
                "start": 0,
                "end": len(text),
                "text": text,
            }
        ],
    }


def test_addresses_and_opportunity():
    data = material()
    comments = validate_addresses(data)
    z = opportunity_numeric(data["sentences"], comments)
    assert z.shape == (1, 8) and np.isfinite(z).all()
    data["sentences"][0]["start"] = 1
    with pytest.raises(ValueError, match="differs"):
        validate_addresses(data)


def test_duplicate_or_unauthorized_material():
    data = material()
    data["sentences"] *= 2
    with pytest.raises(ValueError, match="Duplicate"):
        validate_addresses(data)
    data = material()
    data["bundles"][0]["source_group"] = "bridge"
    with pytest.raises(ValueError, match="Unauthorized"):
        validate_addresses(data)


def test_quadratic_recovers_interaction_and_reload(tmp_path):
    rng = np.random.default_rng(5)
    g = rng.normal(size=(200, 2))
    y = (g[:, 0] * g[:, 1])[:, None]
    z = rng.normal(size=(200, 1))
    sub = np.repeat("known", 200)
    authors = np.repeat(np.arange(100), 2)
    train = authors < 70
    linear = SemanticLink("joint_linear", 0.001).fit(
        g[train], z[train], sub[train], authors[train], y[train]
    )
    quad = SemanticLink("joint_quadratic", 0.001).fit(
        g[train], z[train], sub[train], authors[train], y[train]
    )
    qp = quad.predict(g[~train], z[~train], sub[~train])
    lp = linear.predict(g[~train], z[~train], sub[~train])
    assert np.mean((qp - y[~train]) ** 2) < 1e-6
    assert np.mean((lp - y[~train]) ** 2) > 0.1
    joblib.dump(quad, tmp_path / "reader.joblib")
    np.testing.assert_array_equal(
        qp,
        joblib.load(tmp_path / "reader.joblib").predict(
            g[~train], z[~train], sub[~train]
        ),
    )


def test_held_categories_and_scaler_not_fitted():
    g = np.arange(60).reshape(30, 2).astype(float)
    z = np.arange(30)[:, None]
    authors = np.arange(30)
    sub = np.array(["source"] * 20 + ["held_secret"] * 10)
    reader = SemanticLink("joint_linear", 1).fit(
        g[:20], z[:20], sub[:20], authors[:20], z[:20]
    )
    assert reader.categories_ == ["source"]
    before = reader.g_scaler_.mean_.copy()
    reader.predict(g[20:] + 10000, z[20:], sub[20:])
    np.testing.assert_array_equal(before, reader.g_scaler_.mean_)
    assert np.atleast_2d(reader.head_.coef_).shape[1] == 4


def test_author_paired_improvement_and_constant_prediction():
    y = np.arange(20.0)[:, None]
    authors = np.repeat(np.arange(10), 2)
    result = paired_author_mse(y, y, y + 1, authors, np.ones(1), draws=20)
    assert (
        result["normalized_mse_improvement"]
        == result["ci_low"]
        == result["ci_high"]
        == 1
    )
    r, _, _ = weighted_metrics(y, np.ones_like(y), authors, np.ones(1))
    assert np.isnan(r).all()


def test_unknown_mode_fails():
    g = np.ones((3, 2))
    with pytest.raises(ValueError, match="Unknown"):
        SemanticLink("typo", 1).fit(g, g, ["x"] * 3, [1, 2, 3], g)
