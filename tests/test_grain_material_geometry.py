#!/usr/bin/env python3
"""Exact geometry witnesses, independent of private personality labels."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.material_geometry import (
    cross_covariance,
    local_geometry,
    product_components,
)


def test_product_and_rotation_keep_marginal_spreads():
    """Shifting pairing changes covariance, not the two observed margins."""
    g, p = np.arange(20).reshape(5, 4) ** 2, np.array([1, 2, 8, 5, 0])
    r = local_geometry(g, p)
    np.testing.assert_allclose(r["S"], r["A"] * r["rho"])
    np.testing.assert_allclose(r["shifted_S"], r["A"] * r["shifted_rho"])
    rotated = local_geometry(g, np.roll(p, 1))
    np.testing.assert_allclose(r["A"], rotated["A"])
    assert not np.allclose(r["S"], r["shifted_S"])


def test_spread_scaling_does_not_change_alignment():
    """Positive coordinate scaling multiplies covariance but preserves rho."""
    g, p = np.array([[1, 3], [2, 1], [7, 9]], float), np.array([4, 0, 2])
    x, y = local_geometry(g, p), local_geometry(3 * g + 7, 2 * p - 5)
    np.testing.assert_allclose(y["S"], 6 * x["S"])
    np.testing.assert_allclose(y["rho"], x["rho"])


def test_zero_spread_marked_not_inferred_independence():
    """An undefined observed correlation has an explicit accounting flag."""
    r = local_geometry(np.ones((4, 2)), [1, 2, 3, 4])
    assert r["degenerate"].all()
    np.testing.assert_array_equal(r["rho"], [0, 0])
    assert r["identity_error"] == 0


def test_all_wrong_cyclic_shifts_are_not_an_independent_mean_null():
    """Their mean is exactly minus the true covariance divided by n-1."""
    g, p = np.array([[1, 2], [2, 1], [9, 4], [0, 8]], float), np.array([5, 1, 3, 0])
    correct = cross_covariance(g, p[:, None])
    wrong = np.mean(
        [cross_covariance(g, np.roll(p, k)[:, None]) for k in range(1, 4)], axis=0
    )
    np.testing.assert_allclose(wrong, -correct / 3, atol=1e-12)


def test_reference_components_close_after_multiview_aggregation():
    """Average of products preserves the within-author interaction term."""
    a = np.array([[1, 2, 5], [4, 6, 9]], float)
    r = np.array([[0.9, -0.8, 0.3], [-0.1, 0.3, -0.4]])
    terms = product_components(a, r, a.mean(), r.mean())
    np.testing.assert_allclose(sum(terms.values()).mean(1), (a * r).mean(1))
    assert not np.allclose((a * r).mean(1), a.mean(1) * r.mean(1))


def test_signed_criterion_contributions_sum_to_total_correlation():
    """Signed covariance contributions are not independent variance shares."""
    a = np.array([1, 4, 3, 8], float)
    r = np.array([-0.4, 0.6, 0.3, -0.2])
    y = np.array([1, 0, 4, 2], float)
    s = a * r
    terms = product_components(a, r, a.mean(), r.mean())
    denom = np.std(s) * np.std(y)
    account = sum(
        np.mean((x - x.mean()) * (y - y.mean())) / denom for x in terms.values()
    )
    assert abs(account - np.corrcoef(s, y)[0, 1]) < 1e-12


def test_cross_operator_spectrum_is_orthogonal_coordinate_invariant():
    """Joint structure can persist while displayed coordinate links rotate."""
    rng = np.random.default_rng(13)
    x, y = rng.normal(size=(20, 4)), rng.normal(size=(20, 3))
    a, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    b, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    c, rotated = cross_covariance(x, y), cross_covariance(x @ a, y @ b)
    np.testing.assert_allclose(rotated, a.T @ c @ b, atol=1e-12)
    np.testing.assert_allclose(np.linalg.svd(c)[1], np.linalg.svd(rotated)[1])


def test_bad_alignment_rejected():
    """No silent mismatching or nonfinite inputs."""
    with pytest.raises(ValueError):
        local_geometry(np.zeros((3, 2)), [1, 2])
    with pytest.raises(ValueError):
        cross_covariance(np.zeros((3, 2)), np.full((3, 1), np.nan))


def test_source_index_does_not_confuse_held_edits_with_original_views():
    """A mixed source/edited cache keeps its original physical feature rows."""
    import pandas as pd

    from scripts.run_grain_material_geometry import source_feature_index

    table = pd.DataFrame(
        {
            "author_id": ["held", "source", "held"],
            "view": [0, 0, 0],
            "group": ["held", "big5_only", "held"],
            "edit": ["e1", "source", "e2"],
        }
    )
    result = source_feature_index(table, [("source", 0)])
    assert result.loc[("source", 0), "feature_index"] == 1
    assert len(result) == 1


def test_source_index_rejects_silent_cohort_drop():
    """Missing required source views cannot be repaired by silently filtering."""
    import pandas as pd

    from scripts.run_grain_material_geometry import source_feature_index

    table = pd.DataFrame(
        {
            "author_id": ["source"],
            "view": [0],
            "group": ["big5_only"],
            "edit": ["source"],
        }
    )
    with pytest.raises(ValueError):
        source_feature_index(table, [("source", 0), ("missing", 0)])
