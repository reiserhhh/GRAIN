"""Focused tests for GRAIN-C3-01 declared-target construction."""

import numpy as np
import pytest

from grain_research.construction import declared_target as DT
from grain_research.gptpro_foundation_v3.f04_recovery import score_response


def _world(n=120, k=3, seed=0):
    rng = np.random.default_rng(seed)
    ys, cs = [], []
    for _ in range(n):
        c = np.where(rng.random(25) < 0.4, 1.0, -1.0)
        b0, b1 = rng.normal(0, 1, k), rng.normal(0, 0.5, k)
        ys.append(b0 + np.outer(c, b1) + rng.normal(0, 1, (25, k)))
        cs.append(c)
    return ys, cs


def test_blup_matches_f04_covariance_form():
    ys, cs = _world()
    fits, _ = DT.fit_moments(ys, cs)
    fit = {"mode": "linear", **{k: fits[0][k] for k in ("mu", "Sigma", "sigma2")}}
    pre = [{"z": float(cs[0][t]), "y": float(ys[0][t, 0]), "event_id": f"e{t}"} for t in range(25)]
    ref = score_response(pre, fit, conditions=(-1.0, 1.0))["coefficients"]
    mine, _ = DT.blup(fits[0], ys[0][:, 0], cs[0])
    np.testing.assert_allclose(mine, ref, atol=1e-10)


def test_moments_recover_declared_covariance():
    ys, cs = _world(n=2000, k=1, seed=1)
    fits, info = DT.fit_moments(ys, cs)
    s = np.asarray(fits[0]["Sigma"])
    assert abs(s[0, 0] - 1.0) < 0.12 and abs(s[1, 1] - 0.25) < 0.06 and info["authors_used"] > 1500


def _bundle():
    ys, cs = _world()
    fits, info = DT.fit_moments(ys, cs)
    rule = DT.context_rule(["venueA", "venueB"])
    return DT.build_bundle({"facets": 3}, rule, fits, info, {}, {})


def test_bundle_tamper_and_hash(tmp_path):
    b = _bundle()
    h = DT.save_bundle(b, tmp_path / "b.json")
    assert DT.load_bundle(tmp_path / "b.json", h)["context_rule"]["top"] == ["venueA", "venueB"]
    with pytest.raises(ValueError):
        DT.load_bundle(tmp_path / "b.json", "0" * 64)
    b["context_rule"]["top"].append("venueC")
    with pytest.raises(ValueError):
        DT.verify_bundle(b)


def test_cold_inputs_reject_labels_text_and_duplicates():
    b = _bundle()
    ok = {"event_id": "e1", "venue": "venueA", "evidence": [0.1, 0.2, 0.3]}
    with pytest.raises(ValueError):
        DT.cold_score(b, [{"author_key": "a", "events": [{**ok, "text": "x"}]}])
    with pytest.raises(ValueError):
        DT.cold_score(b, [{"author_key": "a", "label": 1, "events": [ok]}])
    with pytest.raises(ValueError):
        DT.cold_score(b, [{"author_key": "a", "events": [ok, ok]}])


def test_null_and_one_class_support_flags():
    b = _bundle()
    none = DT.cold_score(b, [{"author_key": "a", "events": []}])[0]
    assert none["support"] == "no_events_prior_only"
    np.testing.assert_allclose(none["level"]["mean"], [f["mu"][0] for f in b["fits"]])
    one = DT.cold_score(b, [{"author_key": "a", "events": [{"event_id": "e", "venue": "venueA", "evidence": [1, 1, 1]}]}])[0]
    assert one["support"] == "model_borrowed_contrast"


def test_contrast_weights_follow_context_sign():
    b = _bundle()
    fit = b["fits"][0]
    c = np.array([1.0, 1.0, -1.0, -1.0, -1.0])
    x = np.column_stack([np.ones(5), c])
    cov = np.linalg.inv(np.linalg.inv(np.asarray(fit["Sigma"])) + x.T @ x / fit["sigma2"])
    w = DT.TARGETS["contrast"] @ cov @ x.T / fit["sigma2"]
    assert (w[c > 0] > 0).all() and (w[c < 0] < 0).all()


def test_within_venue_coordinate_uses_source_venue_means_only():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "anon", Path(__file__).resolve().parents[1] / "scripts/run_grain_anonymous_evidence.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    venue = np.array([["a"] * 18 + ["b"] * 18, ["a"] * 36, ["c"] * 36])
    ev = np.zeros((3, 36, 1))
    ev[0, :18, 0], ev[0, 18:, 0], ev[1, :, 0], ev[2, :, 0] = 1.0, 3.0, 1.0, 5.0
    out = mod.within_venue(ev, venue, np.array([0, 1]))
    np.testing.assert_allclose(out[0, :18, 0], 0.0)        # venue a mean (source) = 1
    np.testing.assert_allclose(out[1, :, 0], 0.0)
    assert abs(out[2, 0, 0] - (5.0 - ev[[0, 1]][:, :25].mean())) < 1e-12   # unseen venue -> pooled source mean
