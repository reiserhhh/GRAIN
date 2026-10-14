#!/usr/bin/env python3
"""Minimal construction tests, not evidence of empirical psychometric validity."""

import numpy as np
import pytest

from grain_research.construction import math_target_observation_design as M


def toy():
    """Eight identities: condition0 reveals nuisance, condition1 reveals target."""
    p = np.zeros((8, 2, 8))
    p[:, 0, 0] = 1
    y = np.repeat((np.arange(8) // 4)[:, None], 5, axis=1)
    for u in range(8):
        p[u, 1, u % 4] = 0.5
        p[u, 1, 4 + y[u, 0]] = 0.5
    return p, y


def test_identity_information_is_not_target_information():
    p, y = toy()
    d = M.compile_design(p, y)
    assert d["identity_mi"][3, 0, 0] == pytest.approx(np.log(4))
    assert d["identity_mi"][3, 0, 1] == pytest.approx(np.log(2))
    assert np.all(d["target_mi"][3, 0, 0] == 0)
    np.testing.assert_allclose(d["value"][3, 0, 1], 0.25)
    assert np.all(d["policies"][0, 3, 0, :, 1] == 1)
    assert np.all(d["policies"][1, 3, 0, :, 1] == 1)
    assert np.all(d["policies"][2, 3, 0, :, 0] == 1)


def test_source_law_risk_reduction_matches_forecast():
    p, y = toy()
    d = M.compile_design(p, y)
    e = M.evaluate(d, p, y)
    np.testing.assert_allclose(e["gain"], e["forecast"], atol=1e-14)
    assert e["support"].min() == 1
    assert d["tower_error"] < 1e-14
    assert d["variance_error"] < 1e-14


def test_held_label_change_can_make_expected_positive_gain_negative():
    p, y = toy()
    e = M.evaluate(M.compile_design(p, y), p, 1 - y)
    np.testing.assert_allclose(e["gain"][:, 0], -0.75)
    np.testing.assert_allclose(e["forecast"][:, 0], 0.25)


def test_constant_target_has_zero_gain():
    p, y = toy()
    y[:, 2] = 1
    d = M.compile_design(p, y)
    assert np.nanmax(np.abs(d["value"][..., 2])) == 0
    assert np.nanmax(np.abs(d["target_mi"][..., 2])) == 0


def test_menu_information_stays_in_before_score():
    p, y = toy()
    p[:4, 1, 4:] = 0
    p[:4, 1, :4] *= 2
    p[4:, 1, :4] = 0
    p[4:, 1, 4:] *= 2
    d = M.compile_design(p, y)
    np.testing.assert_equal(d["before"][1, 0], 0)
    np.testing.assert_equal(d["before"][2, 0], 1)
    assert np.nanmax(d["value"]) == 0


def test_unsupported_prefix_has_no_zero_score_or_fallback():
    p, y = toy()
    d = M.compile_design(p, y)
    out = M.recommend(
        M.bundle(d),
        {"event_id": "new", "current_cell": 1, "availability": [0, 1], "target": 0},
    )
    assert out["supported"] is False
    assert "source_before" not in out
    p[:, 0] = 0
    p[:, 0, 1] = 1
    e = M.evaluate(d, p, y)
    assert np.all(e["support"] == 0)
    assert np.all(e["missing_pre"] == 1)


def test_outcome_support_missing_applies_to_all_policies():
    p, y = toy()
    d = M.compile_design(p, y)
    p[:, 1, 4:] = 0
    p[:, 1, 6] = 0.5
    e = M.evaluate(d, p, y)
    assert np.all(e["support"] == 0)
    assert np.all(e["missing_outcome"] == 1)


def test_cold_selector_rejects_label_and_duplicate_menu_inputs():
    p, y = toy()
    b = M.bundle(M.compile_design(p, y))
    record = {"event_id": "e", "current_cell": 0, "availability": [0, 1], "target": 0}
    assert M.recommend(b, record)["condition_probabilities"]["target_risk"] == [
        0.0,
        1.0,
    ]
    with pytest.raises(ValueError):
        M.recommend(b, record | {"label": 1})
    with pytest.raises(ValueError):
        M.recommend(b, record | {"availability": [0, 0]})
    with pytest.raises(ValueError):
        M.recommend(b, record | {"target": True})


def test_event_windows_reject_copied_ids():
    cells = np.array([[[0, 1], [4, 5]]])
    units = np.arange(4).reshape(1, 2, 2)
    p = M.event_probabilities(cells, units)
    assert p.shape == (1, 2, 8)
    units[0, 1, 0] = units[0, 0, 0]
    with pytest.raises(ValueError):
        M.event_probabilities(cells, units)


def test_all_profiles_and_probability_tower_preserved():
    p, y = toy()
    d = M.compile_design(p, y)
    bits = (np.arange(32)[:, None] >> np.arange(5)) & 1
    np.testing.assert_allclose(d["after_profile"][3, 0, 0] @ bits, d["after"][3, 0, 0])
    np.testing.assert_allclose(
        d["outcome"][3, 0, 0] @ d["after_profile"][3, 0, 0], d["before_profile"][3, 0]
    )


def test_aggregate_reader_is_author_order_invariant():
    p, y = toy()
    a = M.compile_design(p, y)
    b = M.compile_design(p[::-1], y[::-1])
    for key in ("before", "after", "value", "target_mi", "identity_mi", "policies"):
        np.testing.assert_allclose(a[key], b[key], equal_nan=True)


def test_invalid_probability_law_rejected():
    p, y = toy()
    p[0, 1, 0] += 1
    with pytest.raises(ValueError):
        M.compile_design(p, y)


def test_runner_frozen_build_evaluation_and_literal_audit(tmp_path):
    """Exercise the full stage plumbing on synthetic addressed archive records."""
    from scripts import run_grain_observed_family_hm170 as R
    from scripts import run_grain_target_observation_design_hm206 as S

    cells = np.zeros((16, 2, 2), int)
    cells[:, 1, 0] = (np.arange(16) // 2) % 4
    cells[:, 1, 1] = 4 + np.arange(16) // 8
    codes = np.where(np.arange(16) < 8, 0, 31)
    folds = np.arange(16) % 2
    R.write(
        tmp_path / "manifest.json",
        {"code": {}, "inputs": {}, "config": {"numeric_tolerance": 2e-10}},
    )
    R.archive(
        tmp_path / "observations.npz",
        source=np.arange(16),
        cells=cells,
        units=np.arange(64).reshape(16, 2, 2),
        codes=codes,
        folds=folds,
        probabilities=M.event_probabilities(cells, np.arange(64).reshape(16, 2, 2)),
        shuffled_codes=codes[::-1],
    )
    R.seal(tmp_path, "prepared_seal.json", ("observations.npz",))
    S.build(tmp_path)
    S.evaluate(tmp_path)
    S.verify(tmp_path)
    R.check(tmp_path, ("built_seal.json", "evaluated_seal.json", "verified_seal.json"))
    assert R.read(tmp_path / "verification.json")["PASS"]
    assert len(R.read(tmp_path / "summary.json")["pooled"]) == 60
    assert all(
        row["shared_support_mean"] == 1
        for row in R.read(tmp_path / "summary.json")["coverage"]
    )
