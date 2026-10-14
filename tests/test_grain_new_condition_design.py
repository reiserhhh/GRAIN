"""Synthetic second-moment, hidden-support and leakage-interface checks."""

import inspect

import numpy as np
import pytest

from grain_research.new_condition_design import (
    basis,
    choose,
    fit_expanded,
    fit_source,
    gains,
    hidden_basis,
    moments,
    replicate_noise,
    report_noise,
    score_weights,
    true_risk,
)
from scripts.run_grain_new_condition_design import ROOT, read, simulate


def model():
    """A fixed, non-fitted source model for exact formula tests."""
    return dict(
        kind="quadratic",
        mean=np.array([0.1, 0.2, 0]),
        covariance=np.diag([0.04, 0.09, 0.01]),
        stationary=0.03,
        rate=0.5,
        noise_levels=np.array([-1, 0, 1]),
        noise=np.array([0.06, 0.06, 0.06]),
        log_noise_coef=np.array([np.log(0.06), 0, 0]),
    )


def test_hidden_source_nullspace():
    assert np.array_equal(hidden_basis([-1, 0, 1]), np.zeros(3))
    np.testing.assert_allclose(hidden_basis([-0.5, 0.5]), [-1, 1])
    assert np.linalg.matrix_rank(basis([-1, 0, 1], "hidden")) == 3
    assert np.linalg.matrix_rank(basis([-1, -0.5, 0, 0.5, 1], "hidden")) == 4


@pytest.mark.parametrize("bad", [2, np.nan, np.inf])
def test_condition_support(bad):
    with pytest.raises(ValueError):
        basis([bad])


def test_cell_not_universal_interpolator():
    with pytest.raises(ValueError):
        basis([0.25], "cells")


def test_duplicate_gives_zero_gain():
    m = moments(model(), [0.5, 0.5, -0.5], [19, 19, 0], [1, 1, 2], "current_response")
    g = gains(m, [0])
    assert abs(g[1]) < 1e-12
    f = score_weights(m, [0])
    duplicate = score_weights(m, [0, 1])
    assert abs(f[2] - duplicate[2]) < 1e-12
    mn = moments(
        model(),
        [0.5, 0.5, -0.5],
        [19, 19, 0],
        [1, 1, 2],
        "current_response",
        naive_duplicates=True,
    )
    assert gains(mn, [0])[1] > 0


def test_conflicting_alias_rejected():
    with pytest.raises(ValueError):
        moments(model(), [0.5, -0.5], [19, 19], [1, 1], "current_response")


@pytest.mark.parametrize(
    "target", ["new_condition_profile", "new_condition_contrast", "current_response"]
)
def test_schur_direct_variance_and_exact_risk(target):
    mm = model()
    mm["mean"] = np.zeros(3)
    m = moments(mm, [-1, -0.5, 0, 0.5, 1], range(5), range(5), target)
    selected = [0, 3]
    before = score_weights(m, selected)
    gs = gains(m, selected)
    for j in [1, 2, 4]:
        w, off, var = score_weights(m, selected + [j])
        assert abs(before[2] - var - gs[j]) < 1e-12
        assert abs(true_risk(m, selected + [j], w, off) - var) < 1e-12


def test_design_no_outcome_interface_and_same_event_budget():
    assert "evidence" not in inspect.signature(choose).parameters
    z = [-1, 0, 1, 0.5, 0.5]
    t = [0, 1, 2, 19, 19]
    ids = [0, 1, 2, 3, 3]
    m = moments(model(), z, t, ids, "current_response")
    for policy in ["source", "random", "coverage"]:
        selected, _ = choose(m, z, t, ids, policy, 4, 17)
        assert len(set(np.array(ids)[selected])) == 4
    with pytest.raises(ValueError):
        choose(m, z, t, ids, "source", 5, 17)


def test_population_bias_part_of_risk():
    m = moments(model(), [-1, 0, 1], [0, 1, 2], [0, 1, 2], "new_condition_profile")
    assert np.isclose(true_risk(m, [], [], m["target_mean"] + 2), m["variance"] + 4)


def test_single_report_variance_not_replicate_mean():
    np.testing.assert_allclose(report_noise(model(), [-0.5, 0.5]), 0.06)
    rng = np.random.default_rng(82)
    y = rng.normal(size=(1000, 6, 2))
    _, v = replicate_noise(y, np.array([-1, 0, 1] * 2))
    np.testing.assert_allclose(v, 1, atol=0.1)


def test_identical_source_different_application_hidden_world():
    cfg = read(ROOT / "configs/grain_new_condition_design.json")
    scores = np.tile(np.array([[-1] * 4, [1] * 4]), (5, 1, 1))
    a, ta = simulate(cfg, scores, 191, "smooth", "source")
    b, tb = simulate(cfg, scores, 191, "hidden_mode", "source")
    np.testing.assert_array_equal(a["observations"], b["observations"])
    np.testing.assert_array_equal(a["bank_indices"], b["bank_indices"])
    c, tc = simulate(cfg, scores, 191, "smooth", "application")
    d, td = simulate(cfg, scores, 191, "hidden_mode", "application")
    assert not np.array_equal(c["observations"], d["observations"])
    np.testing.assert_array_equal(tc["profile"][:, :3], td["profile"][:, :3])
    np.testing.assert_array_equal(tc["state"], td["state"])


def test_source_fit_and_expanded_finite_smoke():
    cfg = read(ROOT / "configs/grain_new_condition_design.json")
    cfg["source_authors"] = 32
    cfg["expanded_source_authors"] = 24
    scores = np.tile(np.array([[-1] * 4, [1] * 4]), (5, 1, 1))
    a, _ = simulate(cfg, scores, 191, "smooth", "source")
    fit = fit_source(a["observations"], a["condition"], a["time"], maxiter=10)
    assert np.isfinite(fit["covariance"]).all()
    assert np.all(np.linalg.eigvalsh(fit["covariance"]) > 0)
    b, _ = simulate(cfg, scores, 191, "smooth", "expanded")
    ex = fit_expanded(b["observations"], b["condition"], b["time"], fit)
    assert ex["covariance"].shape == (5, 5)
    assert np.linalg.eigvalsh(ex["covariance"]).min() > 0
