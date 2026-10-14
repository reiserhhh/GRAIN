#!/usr/bin/env python3
"""Distinguish conditional evidence,routing and cancellation without tuning."""

import numpy as np
import pytest
from scipy.special import logsumexp

from grain_research.construction import math_relational_diagnostic_update as P
from grain_research.construction import math_target_evidence_routing as M


def law(correlated=False, factor="other"):
    """Explicit32-profile experiment,not discovered personality classes."""
    b = P.BITS
    base = (
        np.where(b[:, 0] == b[:, 1], 0.4, 0.1) / 8 if correlated else np.ones(32) / 32
    )
    if factor == "other":
        d = (2 * b[:, 1] - 1) * 0.9
    elif factor == "own":
        d = (2 * b[:, 0] - 1) * 0.9
    else:
        d = (2 * b[:, 0] - 1) * (2 * b[:, 1] - 1) * 0.9
    r = P.update(np.log(base)[None], d[None])
    return r, M.analyze(r["log_profile"], r["log_factor"])


def test_other_target_evidence_routes_without_conditional_change():
    _, a = law(True)
    assert a["corners"][0, 0, 3] - a["corners"][0, 0, 0] > 0.1
    np.testing.assert_allclose(a["conditional_log_odds_update"][0, 0], 0)
    np.testing.assert_allclose(a["probability_accounts"][0, 0, [0, 2]], 0, atol=1e-15)
    assert a["probability_accounts"][0, 0, 1] > 0.1


def test_other_target_evidence_cannot_route_into_independent_target():
    _, a = law(False)
    np.testing.assert_allclose(a["probability_accounts"][0, 0], 0, atol=1e-15)


def test_local_target_evidence_has_conditional_effect():
    _, a = law(False, "own")
    np.testing.assert_allclose(a["conditional_log_odds_update"][0, 0], 1.8)
    assert a["probability_accounts"][0, 0, 0] > 0.3
    np.testing.assert_allclose(a["probability_accounts"][0, 0, 1:], 0, atol=1e-15)


def test_hidden_conditional_effect_can_cancel_in_marginal_reading():
    _, a = law(False, "joint")
    assert np.max(np.abs(a["conditional_log_odds_update"][0, 0])) > 1
    assert np.min(a["conditional_log_odds_update"][0, 0]) < -1
    np.testing.assert_allclose(a["probability_accounts"][0, 0].sum(), 0, atol=1e-15)


def test_all_targets_corners_and_log_scores_reconstruct():
    rng = np.random.default_rng(210)
    r = P.update(rng.normal(size=(32, 32)), rng.normal(size=(32, 32)))
    a = M.analyze(r["log_profile"], r["log_factor"])
    np.testing.assert_allclose(a["corners"][..., 0], r["targets"][:, 1])
    np.testing.assert_allclose(a["corners"][..., 3], r["targets"][:, 0])
    np.testing.assert_allclose(
        a["probability_accounts"].sum(-1),
        r["targets"][:, 0] - r["targets"][:, 1],
        atol=1e-15,
    )
    g = np.arange(32)
    scores = M.scored_accounts(a["corners"], g)
    bit = P.BITS[g]
    expected = np.where(
        bit == 1,
        np.log(r["targets"][:, 0] / r["targets"][:, 1]),
        np.log((1 - r["targets"][:, 0]) / (1 - r["targets"][:, 1])),
    )
    np.testing.assert_allclose(scores.sum(-1), expected, atol=1e-14)


def test_conditional_odds_identity_and_literal_all_bit_addresses():
    r, a = law(True, "joint")
    for q in range(5):
        low, high = M.edges(q)
        odds = r["log_profile"][0, :, high] - r["log_profile"][0, :, low]
        # Scalar pair iteration avoids advanced-index axis ambiguity.
        expected = [
            r["log_profile"][0, 0, hi]
            - r["log_profile"][0, 0, lo]
            - r["log_profile"][0, 1, hi]
            + r["log_profile"][0, 1, lo]
            for lo, hi in zip(low, high, strict=True)
        ]
        assert np.isfinite(odds).all()
        np.testing.assert_allclose(
            a["conditional_log_odds_update"][0, q], expected, atol=1e-15
        )
        np.testing.assert_allclose(
            a["corners"][0, q], M.literal(r["log_profile"][0], q), atol=1e-15
        )


def test_common_factor_changes_no_account():
    r, a = law(True)
    b = M.analyze(r["log_profile"], r["log_factor"] + 4.5)
    for key in a:
        np.testing.assert_allclose(a[key], b[key], atol=1e-15)


def test_rejects_unnormalized_missing_or_inconsistent_law():
    r, _ = law()
    for lp, d in (
        (r["log_profile"] + 1, r["log_factor"]),
        (r["log_profile"][..., :-1], r["log_factor"]),
        (r["log_profile"], r["log_factor"] + np.arange(32)),
        (r["log_profile"] * np.nan, r["log_factor"]),
    ):
        with pytest.raises(ValueError):
            M.analyze(lp, d)
    with pytest.raises(ValueError):
        M.edges(True)


def test_cold_atlas_does_not_guess_other_traits():
    g = np.repeat(np.arange(32), 2)
    z = np.arange(128).reshape(64, 2, 1) / 100
    bank = P.compile_bank(z, g, 0.7)
    record = {
        "frame_key": "cold",
        "windows": [
            [{"event_id": "a", "vector": [0.3], "condition": 0}],
            [{"event_id": "b", "vector": [0.5], "condition": 1}],
        ],
    }
    r = M.cold(bank, record)
    assert len(r["conditional_profiles"]) == 80
    assert all(
        row["hypothetical_not_observed_author_labels"]
        for row in r["conditional_profiles"]
    )
    np.testing.assert_allclose(
        np.sum(r["probability_accounts"], -1), r["original_update"], atol=1e-15
    )
    record["label"] = 1
    with pytest.raises(ValueError):
        M.cold(bank, record)


def test_symmetric_nonzero_interaction_is_retained():
    rng = np.random.default_rng(211)
    r = P.update(np.zeros((1, 32)), rng.normal(size=(1, 32)))
    a = M.analyze(r["log_profile"], r["log_factor"])
    assert np.max(np.abs(a["probability_accounts"][..., 2])) > 1e-4
    assert abs(logsumexp(r["log_profile"][0, 0])) < 1e-15
