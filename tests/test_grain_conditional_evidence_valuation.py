#!/usr/bin/env python3
"""Mechanism controls for conditional mixtures, not precision/accuracy searches."""

import numpy as np
import pytest

from grain_research.construction import math_conditional_evidence_valuation as M
from grain_research.construction import math_relational_diagnostic_update as R


def corners(view=0):
    """Reuse one focal contrast under two fixed companion observations."""
    x = np.array(
        [[[[0.4], [1.0]], [[-0.4], [1.0]]], [[[0.4], [-1.0]], [[-0.4], [-1.0]]]]
    )
    return x if view == 0 else x[:, :, ::-1].copy()


def composition_bank():
    """A singleton/profile mixture with target-by-background association."""
    g = np.arange(32)
    a = (2 * (g % 2) - 1) * (2 * ((g >> 1) % 2) - 1)
    b = 2 * ((g >> 1) % 2) - 1
    return R.compile_bank(np.stack((a, b), 1)[..., None], g, 0.7)


def coupling_bank():
    """Identical profile marginals, different paired-author relation."""
    g = np.repeat(np.arange(32), 2)
    a = np.tile([-1.0, 1.0], 32)
    b = a * (2 * (g % 2) - 1)
    return R.compile_bank(np.stack((a, b), 1)[..., None], g, 0.7)


def test_product_law_can_reverse_without_within_profile_dependence():
    r = M.read(composition_bank(), corners(), 0)
    assert r["actions"][1, 0, 0] > 0.1
    assert r["actions"][1, 1, 0] < -0.1
    np.testing.assert_allclose(r["actions"][0], r["actions"][1], atol=1e-14)
    np.testing.assert_allclose(r["within_profile"], 0, atol=1e-14)
    np.testing.assert_allclose(r["composition"], r["mixed"], atol=1e-14)


def test_author_coupling_changes_evidence_with_same_profile_marginals():
    r = M.read(coupling_bank(), corners(), 0)
    np.testing.assert_allclose(r["actions"][1], 0, atol=1e-14)
    assert abs(r["mixed"][0, 0]) > 0.1
    assert abs(r["within_profile"][0, 0]) > 0.1
    np.testing.assert_allclose(r["closure"], 0, atol=1e-14)


def test_direct_old_law_and_literal_distances_close_on_all_targets():
    bank, x = coupling_bank(), corners()
    r = M.read(bank, x, 0)
    odds = []
    for c, a in np.ndindex(2, 2):
        p = R.literal(bank, x[c, a])["profile"]
        odds.append(np.log((p @ R.BITS) / (p @ (1 - R.BITS))))
    odds = np.array(odds).reshape(2, 2, 2, 5)
    actions = (odds[:, 0] - odds[:, 1]).transpose(1, 0, 2)
    np.testing.assert_allclose(r["actions"], actions, atol=1e-14)
    np.testing.assert_allclose(r["direct_action_error"], 0, atol=1e-14)


def test_target_valuation_rejects_common_gauge_not_common_evidence():
    w = np.arange(32) / 9
    r = np.sin(np.arange(32))
    np.testing.assert_allclose(
        M.valuation(w + 7, r - 13), M.valuation(w, r), atol=1e-14
    )


def test_symmetric_accounts_swap_sign_and_keep_both_orders():
    p = np.arange(20).reshape(2, 2, 5) ** 2 / 100
    r, flipped = M.split_accounts(p), M.split_accounts(p[::-1, ::-1])
    for k in ("mixed", "composition", "within_profile"):
        np.testing.assert_allclose(flipped[k], -r[k], atol=1e-14)
    np.testing.assert_allclose(r["closure"], 0, atol=1e-14)


def test_swapping_both_source_views_preserves_oriented_effect():
    b = coupling_bank()
    other = R.compile_bank(b["prototypes"][:, ::-1], b["codes"], b["variance"])
    r, s = M.read(b, corners(), 0), M.read(other, corners(1), 1)
    for k in ("actions", "mixed", "composition", "within_profile"):
        np.testing.assert_allclose(r[k], s[k], atol=1e-14)


@pytest.mark.parametrize("index", [(1, 0, 0, 0), (0, 1, 1, 0)])
def test_rejects_changed_focal_or_companion_support(index):
    x = corners()
    x[index] += 0.01
    with pytest.raises(ValueError):
        M.read(coupling_bank(), x, 0)


def cold_record():
    """Keep true identity separate from a coincident vector value."""
    x = corners()
    return {
        "frame_key": "control",
        "focal_view": 0,
        "corners": [
            [
                [
                    {"event_ids": [f"focal-{a}"], "mean": x[c, a, 0].tolist()},
                    {"event_ids": [f"companion-{c}"], "mean": x[c, a, 1].tolist()},
                ]
                for a in (0, 1)
            ]
            for c in (0, 1)
        ],
    }


def test_cold_accepts_only_addressed_evidence_not_outcomes():
    record = cold_record()
    r = M.cold(coupling_bank(), record)
    assert len(r["mixed"]) == 2 and len(r["mixed"][0]) == 5
    record["label"] = 1
    with pytest.raises(ValueError):
        M.cold(coupling_bank(), record)


def test_cold_rejects_event_alias_and_missing_or_duplicate_support():
    record = cold_record()
    record["corners"][1][0][0]["event_ids"] = ["different"]
    with pytest.raises(ValueError):
        M.cold(coupling_bank(), record)
    for ids in ([], ["a", "a"], [""]):
        record = cold_record()
        record["corners"][0][0][0]["event_ids"] = ids
        with pytest.raises(ValueError):
            M.cold(coupling_bank(), record)


def test_nonfinite_or_missing_profile_is_not_neutral():
    b = composition_bank()
    for x in (corners() * np.nan, corners()[..., :0]):
        with pytest.raises(ValueError):
            M.read(b, x, 0)
    b["prototypes"] = b["prototypes"][1:]
    b["codes"] = b["codes"][1:]
    with pytest.raises(ValueError):
        M.read(b, corners(), 0)
