#!/usr/bin/env python3
"""Tests of target-profile mass, singleton identity and dependence separation."""

import copy

import numpy as np
import pytest

from grain_research.construction import math_joint_evidence as J
from grain_research.construction import math_target_profile as P


def bundle():
    """Four measured profiles with distinct within-profile empirical authors."""
    y = np.repeat(
        [[0, 0, 0, 0, 0], [0, 1, 1, 1, 1], [1, 0, 0, 0, 0], [1, 1, 1, 1, 1]], 2, axis=0
    )
    p = np.array(
        [[10 + i, 19 - i, 8 + (i // 2), 13 - (i // 2), 4, 6, 7, 9] for i in range(8)],
        float,
    )
    p /= p.sum(1, keepdims=True)
    return P.build_bundle(J.build_bundle(p, y))


@pytest.mark.parametrize("cell", range(8))
def test_all_singleton_target_means_and_priors_are_preserved(cell):
    b = bundle()
    np.testing.assert_allclose(
        P.evaluate(b, [cell])["probabilities"],
        np.tile(b["parent_bundle"]["event_probability"][cell], (3, 1)),
        atol=2e-15,
    )
    assert max(P.invariants(b).values()) < 2e-15


def test_all_joint_profiles_retained_without_class_selection():
    b = bundle()
    assert b["profile_counts"] == [2, 2, 2, 2]
    np.testing.assert_allclose(b["profile_weights"], 0.25)
    assert P.validate_bundle(b) is b


def test_signed_total_covariance_identity_with_negative_offdiagonals():
    a = P.two_event_accounts(bundle())
    np.testing.assert_allclose(
        a["within_profile"] + a["between_profiles"], a["total_dependence"], atol=2e-15
    )
    assert np.min(a["within_profile"]) < 0
    assert np.linalg.norm(a["within_profile"]) > 0
    assert np.linalg.norm(a["between_profiles"]) > 0
    np.testing.assert_allclose(
        a["signed_within"] + a["signed_between"], a["signed_total"], atol=2e-15
    )


def test_no_within_profile_variation_makes_author_and_profile_agree():
    b = bundle()
    parent = b["parent_bundle"]
    y = np.asarray(parent["source_binary_targets"])
    p = np.repeat(np.asarray(b["profile_mass"]), 2, axis=0)
    b = P.build_bundle(J.build_bundle(p, y))
    out = P.evaluate(b, [0, 1, 1, 3])
    np.testing.assert_allclose(
        out["probabilities"][1], out["probabilities"][2], atol=2e-15
    )
    np.testing.assert_allclose(P.two_event_accounts(b)["within_profile"], 0, atol=2e-15)


def test_within_profile_variation_can_change_joint_target_meaning():
    b = bundle()
    out = P.evaluate(b, [0, 1])
    assert np.max(np.abs(out["probabilities"][1] - out["probabilities"][2])) > 1e-8


def test_no_joint_support_remains_missing_not_zero():
    b = P.build_bundle(
        J.build_bundle(np.eye(8), np.repeat([[0] * 5, [1] * 5], 4, axis=0))
    )
    out = P.evaluate(b, [0, 1])
    np.testing.assert_allclose(out["probabilities"][1], 0)
    assert np.isnan(out["probabilities"][2]).all()
    assert not out["posterior_supported"][2].any()
    assert not out["finite_log_odds"][1].any()


def test_author_permutation_is_not_an_extra_measurement():
    b = bundle()
    old = b["parent_bundle"]
    idx = np.arange(7, -1, -1)
    other = P.build_bundle(
        J.build_bundle(
            np.asarray(old["source_mass"])[idx],
            np.asarray(old["source_binary_targets"])[idx],
        )
    )
    np.testing.assert_allclose(
        P.evaluate(b, [0, 1, 2])["probabilities"],
        P.evaluate(other, [0, 1, 2])["probabilities"],
        atol=2e-15,
    )


def test_empty_observation_is_prior_not_positive_measurement():
    b = bundle()
    out = P.cold_record(b, {"author_key": "none", "events": []})
    assert out["status"] == "NO_OBSERVATIONS"
    np.testing.assert_allclose(
        out["probabilities"], np.tile(b["parent_bundle"]["prior"], (3, 1))
    )


@pytest.mark.parametrize("field", ("profile_weights", "profile_mass", "profile_labels"))
def test_changed_source_parameters_rejected(field):
    b = copy.deepcopy(bundle())
    if field == "profile_weights":
        b[field][0] += 0.01
    elif field == "profile_mass":
        b[field][0][0] += 0.01
    else:
        b[field][0][0] = 1
    with pytest.raises(ValueError, match="Changed"):
        P.validate_bundle(b)


def test_duplicate_ids_and_application_outcomes_rejected():
    record = {
        "author_key": "a",
        "events": [{"event_id": "one", "cell": 0}, {"event_id": "one", "cell": 1}],
    }
    with pytest.raises(ValueError, match="Distinct"):
        P.cold_record(bundle(), record)
    record["events"][1]["event_id"] = "two"
    assert P.cold_record(bundle(), record)["events"] == 2
    record["target"] = 1
    with pytest.raises(ValueError, match="outcomes"):
        P.cold_record(bundle(), record)
