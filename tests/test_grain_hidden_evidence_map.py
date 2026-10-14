"""Interface and unit tests for GRAIN-C1-01 (hidden evidence map)."""

import inspect

import numpy as np
import pytest

from grain_research.construction import hidden_evidence_map as H
from grain_research.gptpro_foundation_v3.f04_recovery import (
    fit_source_moments,
    score_response,
)


@pytest.fixture(scope="module")
def w2():
    return H.build_world(4242, "W2_conditional")


def test_view_has_no_truth_fields():
    names = H.ESTIMATOR_FIELDS
    for bad in ("maps", "theta", "fingerprint", "style", "expression"):
        assert not any(bad in n for n in names)
    assert set(names) == {
        "source_x", "source_z", "source_author", "app_x", "app_z", "app_author",
        "pilot_x", "pilot_writer", "pilot_label", "pilot_label_kind", "lexicon_index", "lexicon_sign"}


def test_estimators_take_no_truth_argument():
    for fn in (H.fit_arm, H.fit_moments, H.blup, H.score_targets,
               H.direction_author_stability, H.direction_response_stability):
        params = inspect.signature(fn).parameters
        assert "truth" not in params


def test_estimator_outputs_invariant_to_truth_mutation(w2):
    view, truth = w2
    before = {a: H.fit_arm(a, view).direction.copy() for a in ("pilot_ridge", "stability_response", "keyed")}
    digest = H.view_digest(view)
    for key in ("a", "b", "f", "s"):
        truth.app_latent[key][:] = np.random.default_rng(1).normal(size=truth.app_latent[key].shape)
    truth.maps["u_e"][:] = 0.0
    after = {a: H.fit_arm(a, view).direction for a in before}
    assert H.view_digest(view) == digest
    for a, value in before.items():
        np.testing.assert_array_equal(value, after[a])


def test_vectorized_kernel_matches_f04():
    rng = np.random.default_rng(3)
    z = np.tile(np.array(H.CONDITIONS), 8)
    n = 40
    a, b = rng.normal(size=n), rng.normal(.5, .5, n)
    y = a[:, None] + b[:, None] * z + rng.normal(0, .5, (n, len(z)))
    rows = [{"author_id": f"u{u}", "role": "source", "z": float(z[t]), "y": float(y[u, t]),
             "event_id": f"u{u}/{t}", "t": t} for u in range(n) for t in range(len(z))]
    ref = fit_source_moments(rows)
    mine = H.fit_moments(y.reshape(-1), np.tile(z, n), np.repeat(np.arange(n), len(z)))
    np.testing.assert_allclose(mine["mu"], ref["mu"], atol=1e-10)
    np.testing.assert_allclose(mine["Sigma"], ref["Sigma"], atol=1e-10)
    assert abs(mine["sigma2"] - ref["sigma2"]) < 1e-12
    fit = {"mode": "linear", "mu": ref["mu"], "Sigma": ref["Sigma"], "sigma2": ref["sigma2"]}
    beta, post, _ = H.blup(mine, H.design(z), y[:3])
    for u in range(3):
        prefix = [r for r in rows if r["author_id"] == f"u{u}"]
        out = score_response(prefix, fit)
        np.testing.assert_allclose(beta[u], out["coefficients"], atol=1e-10)
        np.testing.assert_allclose(post, out["covariance"], atol=1e-10)


def test_classical_calibration_unit_gain():
    rng = np.random.default_rng(5)
    lab = rng.normal(size=200)
    proj = 0.3 + 2.5 * lab + rng.normal(0, .01, 200)
    alpha, gamma, t = H.calibrate(proj, lab, with_t=True)
    assert abs(gamma - 2.5) < 0.01 and abs(alpha - .3) < .01 and t > 100
    r = H.Readout("x", np.array([1.0]), alpha, gamma)
    np.testing.assert_allclose(r.phi(proj[:, None]), lab, atol=.02)


def test_missing_anchor_refuses_supervised_arms():
    view, _ = H.build_world(4242, "W3_missing_anchor")
    assert view.pilot_label is None
    for arm in ("pilot_ridge", "pilot_pcr"):
        assert H.fit_arm(arm, view).status == "MISSING_ANCHOR"
    r = H.fit_arm("stability_author", view)
    assert r.status == "UNIT_AND_ORIENTATION_UNIDENTIFIED"


def test_contrast_cancels_additive_author_constant(w2):
    view, _ = w2
    r = H.fit_arm("pilot_ridge", view)
    fit = H.fit_moments(r.phi(view.source_x), view.source_z, view.source_author)
    base = H.score_targets(r, fit, view.app_x, view.app_z, view.app_author)
    shift = np.repeat(np.random.default_rng(2).normal(size=H.N_APP), H.EVENTS)[:, None] * r.direction[None, :]
    shift = shift / (r.direction @ r.direction) * r.unit
    moved = H.score_targets(r, fit, view.app_x + shift, view.app_z, view.app_author)
    np.testing.assert_allclose(moved["delta"]["ols"], base["delta"]["ols"], atol=1e-9)
    assert np.std(moved["level"]["ols"] - base["level"]["ols"]) > .5
