#!/usr/bin/env python3
"""Controls for author comparisons under exact, incomplete observation laws."""

import numpy as np
import pytest

from grain_research.construction import math_score_fiber as M


def toy():
    w, _ = M.character_basis()
    return w[:, 1:7], np.full(8, 1 / 8), np.sqrt(8) * w[:, 7]


def test_rank_seven_summary_leaves_third_order_fiber():
    scores, p, third = toy()
    c = M.row_chart(M.observation_design(scores, "scores_only"))
    assert c["rank"] == 7
    lo, hi, _ = M.fiber_endpoints(p, c)
    np.testing.assert_allclose(c["design"] @ lo, c["design"] @ hi, atol=1e-12)
    interval = M.target_interval(third, (lo, hi))
    assert interval[0] == pytest.approx(-1)
    assert interval[1] == pytest.approx(1)
    assert (
        np.max(
            abs(
                c["null"] @ c["null"].T
                - np.outer(M.cofactor_null(c["design"]), M.cofactor_null(c["design"]))
            )
        )
        < 1e-12
    )


def test_joint_readouts_complete_without_changing_six_mean_scores():
    scores, p, third = toy()
    pp, pm = p * (1 + 0.5 * third), p * (1 - 0.5 * third)
    np.testing.assert_allclose(pp @ scores, pm @ scores, atol=1e-12)
    for scheme, expected in (
        ("scores_only", 7),
        ("owner_count", 7),
        ("owner_response", 7),
        ("score_squares", 7),
        ("score_cross_products", 8),
        ("score_all_products", 8),
    ):
        chart = M.row_chart(M.observation_design(scores, scheme))
        assert chart["rank"] == expected
        if expected == 8:
            lower, upper, _ = M.target_interval(third, M.fiber_endpoints(pp, chart))
            assert lower == pytest.approx(0.5)
            assert upper == pytest.approx(0.5)
    np.testing.assert_allclose(
        np.outer(pp @ scores, pp @ scores),
        np.outer(pm @ scores, pm @ scores),
        atol=1e-12,
    )


def test_sharp_interval_matches_lp_and_valid_duals():
    scores, p, third = toy()
    chart = M.row_chart(M.observation_design(scores, "scores_only"))
    result = M.lp_interval(third, p, chart)
    assert result["lower"]["value"] == pytest.approx(-1)
    assert result["upper"]["value"] == pytest.approx(1)
    assert max(result[s]["error"] for s in result) < 1e-12


def test_author_ranking_can_reverse_inside_same_score_fiber():
    scores, p, third = toy()
    chart = M.row_chart(M.observation_design(scores, "scores_only"))
    interval = M.target_interval(third, M.fiber_endpoints(p, chart))
    lower, upper = M.comparison_interval(interval, interval)
    assert lower == pytest.approx(-2)
    assert upper == pytest.approx(2)


def test_boundary_distribution_has_narrower_fiber():
    scores, _, third = toy()
    p = np.eye(8)[0]
    chart = M.row_chart(M.observation_design(scores, "scores_only"))
    lower, upper, _ = M.target_interval(third, M.fiber_endpoints(p, chart))
    assert upper - lower == pytest.approx(0, abs=1e-12)


@pytest.mark.parametrize("p", [[-1] + [2 / 7] * 7, [0.2] * 8, [np.nan] * 8, [0.5, 0.5]])
def test_bad_probability_is_not_repaired(p):
    with pytest.raises(ValueError):
        M.probability(p, 8)


def test_multidimensional_fiber_rejected_not_silently_reduced():
    with pytest.raises(ValueError, match="zero/one"):
        M.fiber_endpoints(np.full(8, 1 / 8), M.row_chart(np.ones((1, 8))))


def test_original_unit_constraints_under_rescaling():
    scores, p, third = toy()
    a = M.observation_design(scores, "scores_only")
    a[1:] *= np.logspace(-3, 3, 6)[:, None]
    chart = M.row_chart(a)
    assert chart["equilibration_error"] < 1e-12
    assert M.lp_interval(third, p, chart)["upper"]["value"] == pytest.approx(1)
