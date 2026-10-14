#!/usr/bin/env python3
"""Tests of observation laws, context-sensitive updates and native nulls."""

import copy

import numpy as np
import pytest

from grain_research.construction import math_joint_evidence as J


def bundle():
    """Same one-event target law, different within-target author covariance."""
    p = np.tile([0.3, 0.3, *([0.4 / 6] * 6)], (4, 1))
    p[0, :2], p[1, :2] = [0.55, 0.05], [0.05, 0.55]
    return J.build_bundle(p, np.tile([[0], [0], [1], [1]], (1, 5)))


@pytest.mark.parametrize("cell", range(8))
def test_all_three_single_event_readers_agree(cell):
    b = bundle()
    a = J.evaluate(b, [cell])
    np.testing.assert_allclose(
        a["probabilities"],
        np.tile(np.asarray(b["event_probability"])[cell], (3, 1)),
        atol=2e-15,
    )


def test_equal_single_event_fields_do_not_determine_joint_target_meaning():
    b = bundle()
    np.testing.assert_allclose(J.evaluate(b, [0])["probabilities"], 0.5)
    joint = J.evaluate(b, [0, 1])["probabilities"]
    np.testing.assert_allclose(joint[:2], 0.5)
    np.testing.assert_allclose(joint[2], 0.09 / (0.09 + 0.0275))


def test_source_target_iid_log_odds_sum_and_mean_probability_identity():
    b = bundle()
    groups = J.evidence_groups(np.arange(12) % 8, np.arange(4), np.arange(4, 8))
    out = [J.evaluate(b, a) for a in groups]
    probabilities = np.array([r["probabilities"] for r in out])
    odds = np.array([r["binary_log_odds"] for r in out])
    np.testing.assert_allclose(J.composition(probabilities, 0.5)[0], 0, atol=1e-15)
    np.testing.assert_allclose(J.composition(odds)[0], 0, atol=1e-14)


def test_same_event_increment_depends_on_previous_context():
    b = bundle()
    base = J.evaluate(b, [0])["binary_log_odds"]
    repeat = J.evaluate(b, [0, 0])["binary_log_odds"] - base
    different = J.evaluate(b, [0, 1])["binary_log_odds"] - base
    np.testing.assert_allclose(repeat[0], different[0], atol=1e-15)
    assert np.max(np.abs(repeat[1] - different[1])) > 0.1


def test_true_unsupported_joint_is_missing_not_smoothed_to_half():
    p = np.tile(np.r_[np.full(4, 0.25), np.zeros(4)], (4, 1))
    p[[1, 3]] = np.r_[np.zeros(4), np.full(4, 0.25)]
    b = J.build_bundle(p, np.tile([[0], [0], [1], [1]], (1, 5)))
    a = J.evaluate(b, [0, 4])
    assert np.isfinite(a["probabilities"][:2]).all()
    assert np.isnan(a["probabilities"][2]).all()
    assert not a["posterior_supported"][1].any()


def test_empty_observation_returns_prior_but_is_not_measurement():
    b = bundle()
    r = J.cold_record(b, {"author_key": "none", "events": []})
    assert r["status"] == "NO_OBSERVATIONS"
    np.testing.assert_allclose(r["probabilities"], 0.5)


def test_repeated_cells_are_distinct_evidence_but_copied_ids_rejected():
    b = bundle()
    record = {
        "author_key": "x",
        "events": [{"event_id": "a", "cell": 0}, {"event_id": "b", "cell": 0}],
    }
    r = J.cold_record(b, record)
    assert r["probabilities"][2][0] != pytest.approx(0.5)
    record["events"][1]["event_id"] = "a"
    with pytest.raises(ValueError, match="Distinct"):
        J.cold_record(b, record)


def test_event_order_does_not_change_this_working_exchangeable_law():
    b = bundle()
    a, c = J.evaluate(b, [0, 1, 2]), J.evaluate(b, [2, 0, 1])
    np.testing.assert_array_equal(a["probabilities"], c["probabilities"])


def test_label_inversion_reverses_log_odds_not_evidence_likelihood():
    b = bundle()
    inverted = J.build_bundle(
        b["source_mass"], 1 - np.asarray(b["source_binary_targets"])
    )
    a, c = J.evaluate(b, [0, 1]), J.evaluate(inverted, [0, 1])
    np.testing.assert_allclose(c["probabilities"], 1 - a["probabilities"])
    np.testing.assert_allclose(c["binary_log_odds"], -a["binary_log_odds"], atol=2e-15)
    np.testing.assert_allclose(c["log_evidence"], a["log_evidence"])


def test_native_pre_softmax_null_survives_arbitrary_condition_gauge():
    rng = np.random.default_rng(174)
    r, s, c = rng.normal(size=(3, 10))
    z = np.array([r + s + c, r + c, r + s, s, c])
    z += np.array([4, 9, 9, -3, -3])[:, None] * np.arange(10)
    np.testing.assert_allclose(J.composition(J.categorical_contrasts(z)), 0, atol=1e-14)
    gauged = z + np.array([17, -9, 8, -7, 19])[:, None]
    np.testing.assert_allclose(
        J.composition(J.categorical_contrasts(gauged)), 0, atol=1e-14
    )
    witness = np.zeros((5, 10))
    witness[:, 1] = [2, 0, 3, 2, -1]
    np.testing.assert_array_equal(
        J.composition(J.categorical_contrasts(witness)), np.zeros(9)
    )
    softmax = np.exp(witness - witness.max(1, keepdims=True))
    softmax /= softmax.sum(1, keepdims=True)
    assert np.max(np.abs(J.composition(softmax))) > 0.01


def test_all_nine_native_axes_expose_context_interaction():
    z = np.zeros((5, 10))
    z[2, 1:] = np.arange(1, 10) / 10
    np.testing.assert_allclose(
        J.composition(J.categorical_contrasts(z)), np.arange(1, 10) / 10
    )


@pytest.mark.parametrize("cells", [[-1], [8], [0.5], [[1]]])
def test_invalid_cells_rejected(cells):
    with pytest.raises(ValueError):
        J.counts(cells)


def test_cold_outcomes_and_bundle_fitting_fields_rejected():
    b = bundle()
    J.validate_bundle(b)
    with pytest.raises(ValueError, match="excludes"):
        J.cold_record(b, {"author_key": "x", "events": [], "native": 0.7})
    changed = copy.deepcopy(b)
    changed["temperature"] = 0.5
    with pytest.raises(ValueError, match="extra"):
        J.validate_bundle(changed)


def test_group_indices_preserve_order_and_reject_overlap():
    groups = J.group_indices([8, 0, 6, 2], [7, 1, 9, 3])
    np.testing.assert_array_equal(groups[3], [0, 2, 6, 8])
    with pytest.raises(ValueError, match="disjoint"):
        J.group_indices(np.arange(4), np.arange(2, 6))
