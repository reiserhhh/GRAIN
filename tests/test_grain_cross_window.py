#!/usr/bin/env python3
"""Tests of genuine disjoint-window laws, nulls and outcome-free cold scoring."""

import copy

import numpy as np
import pytest

from grain_research.construction import math_cross_window as W


def source(reverse=False):
    """Two profiles, three different authors/profile, normalized two-window laws."""
    y = np.repeat([[0] * 5, [1] * 5], 3, axis=0)
    p = np.array(
        [[10 + i, 18 - i, 7 + i % 3, 9 - i % 3, 4, 6, 8, 10] for i in range(6)], float
    )
    p /= p.sum(1, keepdims=True)
    late = p[::-1] if reverse else p.copy()
    return np.stack([p, late], axis=1), y


def test_all_laws_preserve_single_window_target_marginals():
    p, y = source()
    law = W.source_laws(p, y)
    for q in range(5):
        for g in (0, 1):
            np.testing.assert_allclose(
                law[:, q, g].sum(-1),
                np.tile(p[y[:, q] == g, 0].sum(0) / len(p), (3, 1)),
                atol=2e-15,
            )
            np.testing.assert_allclose(
                law[:, q, g].sum(-2),
                np.tile(p[y[:, q] == g, 1].sum(0) / len(p), (3, 1)),
                atol=2e-15,
            )
    assert (law >= 0).all()


def test_distinct_author_reference_excludes_self_exactly():
    p, y = source()
    law = W.source_laws(p, y)
    c = W.centered_accounts(p, y)["cross"]
    np.testing.assert_allclose(law[2] - law[0], c, atol=2e-15)
    np.testing.assert_allclose(law[2] - law[1], 1.5 * c, atol=2e-15)


@pytest.mark.parametrize("reverse", (False, True))
def test_common_change_identity_preserves_signed_cross_covariance(reverse):
    p, y = source(reverse)
    a = W.centered_accounts(p, y)
    np.testing.assert_allclose(
        (a["v0"] + a["v1"]) / 2, a["common"] + a["change"], atol=2e-15
    )
    np.testing.assert_allclose(
        a["cross_symmetric"], a["common"] - a["change"], atol=2e-15
    )
    if reverse:
        assert np.linalg.eigvalsh(a["cross_symmetric"][0].sum(0)).min() < -1e-6
    else:
        np.testing.assert_allclose(a["change"], 0, atol=2e-15)


def test_no_within_profile_heterogeneity_gives_no_person_link():
    p, y = source()
    for g in (0, 1):
        p[y[:, 0] == g] = p[y[:, 0] == g].mean(0)
    law = W.source_laws(p, y)
    np.testing.assert_allclose(law[0], law[1], atol=2e-15)
    np.testing.assert_allclose(law[1], law[2], atol=2e-15)
    np.testing.assert_allclose(W.centered_accounts(p, y)["cross"], 0, atol=2e-15)


def test_condition_choice_response_identity_keeps_negative_cross_terms():
    p, y = source()
    law = W.source_laws(p, y)
    a = W.condition_accounts(law[2], law[1])
    np.testing.assert_allclose(
        a["choice"] + a["response"] + a["unresolved"], law[2] - law[1], atol=2e-15
    )
    assert a["conditional_supported"].all()
    assert np.any(a["response"] < 0)


def test_missing_condition_is_unresolved_not_filled_response():
    a, b = np.zeros((8, 8)), np.zeros((8, 8))
    a[0, 0], b[4, 4] = 1, 1
    v = W.condition_accounts(a, b)
    assert not v["conditional_supported"].any()
    np.testing.assert_array_equal(v["choice"], 0)
    np.testing.assert_array_equal(v["response"], 0)
    np.testing.assert_array_equal(v["unresolved"], a - b)


def test_null_permutations_are_fixed_and_do_not_select_traits():
    p, y = source()
    observed, null, tail = W.matched_reference(p, y, 19, 176)
    assert observed.shape == (6,) and null.shape == (19, 6) and tail.shape == (6,)
    other = W.matched_reference(p, y, 19, 176)
    for a, b in zip((observed, null, tail), other, strict=True):
        np.testing.assert_array_equal(a, b)
    assert ((tail > 0) & (tail <= 1)).all()


def test_pair_scorer_is_a_functional_not_whole_history_posterior():
    p, y = source()
    bundle = W.build_bundle(p, y)
    events = [
        {"event_id": "a", "window": 0, "cell": 0},
        {"event_id": "b", "window": 1, "cell": 1},
    ]
    a = W.cold_record(bundle, {"author_key": "one", "events": events})
    events += [
        {"event_id": "c", "window": 0, "cell": 0},
        {"event_id": "d", "window": 1, "cell": 1},
    ]
    b = W.cold_record(bundle, {"author_key": "one", "events": events})
    np.testing.assert_allclose(
        a["pair_target_readout"], b["pair_target_readout"], atol=2e-15
    )
    assert a["cross_pairs"] == 1 and b["cross_pairs"] == 4
    assert "not_whole_history_posterior" in a["interpretation"]


def test_sparse_pair_support_returns_missing_not_renormalized_score():
    p = np.zeros((4, 2, 8))
    p[:2, :, 0], p[2:, :, 1] = 1, 1
    y = np.repeat([[0] * 5, [1] * 5], 2, axis=0)
    b = W.build_bundle(p, y)
    r = {
        "author_key": "none",
        "events": [
            {"event_id": "a", "window": 0, "cell": 0},
            {"event_id": "b", "window": 1, "cell": 1},
        ],
    }
    a = W.cold_record(b, r)
    assert not np.asarray(a["supported"]).any()
    assert a["pair_target_readout"] == [[None] * 5] * 3


def test_empty_window_is_not_a_positive_measurement():
    p, y = source()
    r = W.cold_record(W.build_bundle(p, y), {"author_key": "none", "events": []})
    assert r["status"] == "MISSING_WINDOW"
    assert r["cross_pairs"] == 0


@pytest.mark.parametrize(
    "bad", ("outcome", "duplicate", "float_window", "invalid_cell")
)
def test_outcomes_identity_and_cell_errors_rejected(bad):
    p, y = source()
    r = {
        "author_key": "one",
        "events": [
            {"event_id": "a", "window": 0, "cell": 0},
            {"event_id": "b", "window": 1, "cell": 1},
        ],
    }
    if bad == "outcome":
        r["target"] = 1
    elif bad == "duplicate":
        r["events"][1]["event_id"] = "a"
    elif bad == "float_window":
        r["events"][1]["window"] = 1.0
    else:
        r["events"][1]["cell"] = 8
    with pytest.raises(ValueError):
        W.cold_record(W.build_bundle(p, y), r)


def test_nonbinary_source_labels_and_singleton_profiles_rejected():
    p, y = source()
    y = y.astype(float)
    y[0, 0] = 0.3
    with pytest.raises(ValueError):
        W.build_bundle(p, y)
    p, y = source()
    y[0, 0] = 1
    with pytest.raises(ValueError, match="two authors"):
        W.build_bundle(p, y)


@pytest.mark.parametrize("field", ("pair_readout", "smoothing"))
def test_altered_bundle_or_extra_fitting_parameters_rejected(field):
    p, y = source()
    b = W.build_bundle(p, y)
    assert W.validate_bundle(b) is b
    b = copy.deepcopy(b)
    if field == "pair_readout":
        b[field][0][0][0][0] += 0.01
    else:
        b[field] = 0.1
    with pytest.raises(ValueError, match="Changed"):
        W.validate_bundle(b)
