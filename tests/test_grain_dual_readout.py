"""Exact algebra and interface tests for GRAIN-C1-02 dual readouts."""

import inspect

import numpy as np
import pytest

from grain_research.construction import dual_readout as DR
from grain_research.construction import hidden_evidence_map as H


def _pair(cos=0.6, dim=10, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim)
    v /= np.linalg.norm(v)
    q = rng.normal(size=dim)
    q -= v * (v @ q)
    q /= np.linalg.norm(q)
    s = cos * v + np.sqrt(1 - cos**2) * q
    return 1.5 * v, s


def test_exact_two_direction_independent():
    v, s = _pair()
    c = 0.25 * np.outer(v, v) + 1.0 * np.outer(s, s)
    rr = DR.rank_rule(c)
    assert rr["k"] == 2
    w = DR.lcmv_single(rr["U"], rr["L"], v)
    np.testing.assert_allclose(w, DR.dual_basis(v, s), atol=1e-10)
    assert abs(w @ v - 1) < 1e-12 and abs(w @ s) < 1e-10


def test_correlated_boundary_algebra():
    v, s = _pair()
    sb, sr, rho = 0.5, 1.0, 0.6
    g = np.column_stack([v, s])
    c = g @ np.array([[sb**2, rho * sb * sr], [rho * sb * sr, sr**2]]) @ g.T
    rr = DR.rank_rule(c)
    w = DR.lcmv_single(rr["U"], rr["L"], v)
    assert abs(w @ v - 1) < 1e-12
    assert abs(w @ s - (-rho * sb / sr)) < 1e-10


def test_two_constraint_lcmv_exact():
    rng = np.random.default_rng(1)
    v, s = _pair(seed=2)
    extra = rng.normal(size=10)
    c = 0.25 * np.outer(v, v) + np.outer(s, s) + 0.3 * np.outer(extra, extra)
    rr = DR.rank_rule(c)
    w = DR.lcmv_two(rr["U"], rr["L"], v, s)
    assert abs(w @ v - 1) < 1e-10 and abs(w @ s) < 1e-10


def test_rank_rule_handles_non_psd_and_zero():
    rr = DR.rank_rule(-np.eye(4) * 0.1)
    assert rr["k"] == 0 and rr["tau"] == pytest.approx(0.1)


def test_augmented_generator_matches_and_isolated():
    view, aug, truth = DR.build_world_augmented(4242, "W2_conditional")
    np.testing.assert_array_equal(aug.pilot_z, truth.pilot_latent["z"])
    for f in H.ESTIMATOR_FIELDS:
        a, b = getattr(view, f), getattr(aug, f)
        assert (a is b) or (a is None and b is None) or np.array_equal(a, b) or a == b
    assert set(DR.AUGMENTED_FIELDS) - set(H.ESTIMATOR_FIELDS) == {"pilot_z"}
    assert "truth" not in inspect.signature(DR.fit_dual).parameters


def test_dual_outputs_ignore_truth_and_pilot_dual_reads_pilot_z():
    view, aug, truth = DR.build_world_augmented(4242, "W2_conditional")
    w1 = DR.fit_dual("lcmv_dual", view).direction.copy()
    w2 = DR.fit_dual("pilot_dual", aug).direction.copy()
    for key in ("a", "b", "f", "s"):
        truth.app_latent[key][:] = 0
        truth.source_latent[key][:] = 0
    truth.pilot_latent["z"][:] = 0
    truth.maps["u_s"][:] = 0
    np.testing.assert_array_equal(DR.fit_dual("lcmv_dual", view).direction, w1)
    np.testing.assert_array_equal(DR.fit_dual("pilot_dual", aug).direction, w2)
    perm = DR.AugmentedEstimatorView(**{f: getattr(aug, f) for f in H.ESTIMATOR_FIELDS},
                                     pilot_z=np.random.default_rng(3).permutation(aug.pilot_z))
    assert not np.allclose(DR.fit_dual("pilot_dual", perm).direction, w2)
    with pytest.raises(TypeError):
        DR.fit_dual("pilot_dual", view)


def test_missing_anchor_refuses():
    view, aug, _ = DR.build_world_augmented(4242, "W3_missing_anchor")
    assert DR.fit_dual("lcmv_dual", view).status == "MISSING_ANCHOR"
    assert DR.fit_dual("pilot_dual", aug).status == "MISSING_ANCHOR"
