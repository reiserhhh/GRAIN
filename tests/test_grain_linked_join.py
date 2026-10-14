#!/usr/bin/env python3
"""Source linkage is different from normalized evidence combination."""

import numpy as np
import pytest
from scipy.special import expit

from grain_research.construction import math_event_anchor_field as A
from grain_research.construction import math_linked_join as M


def bank(repeats=1):
    """Provide a complete finite profile instrument, not a human latent model."""
    codes = np.repeat(np.arange(32), repeats)
    bits = (codes[:, None] >> np.arange(5)) & 1
    z = np.repeat(bits[:, 0, None, None], 4, axis=1).astype(float)
    return {
        "source_comments": z,
        "codes": codes,
        "anchors": bits.astype(float),
        "variance": np.array(1.0),
        "directions": np.ones((1, 1)),
    }


def endpoints():
    """Use four distinct addressed observations with repeated coordinate values."""
    return np.array([[[[2.0], [0.0]], [[2.0], [0.0]]]])


def test_product_normalization_interacts_without_source_linkage():
    b, x = bank(), M.corners(endpoints()).reshape(8, 2, 1)
    r = M.pair_readings(b, x, "E5", "pooled_comments")
    actions = M.mixed_actions(r["targets"].reshape(1, 2, 4, 2, 5))
    expected = expit(3) - 2 * expit(1) + expit(-1)
    assert actions["profile_product"][0, 0, 0] == pytest.approx(expected)
    assert abs(expected) > 0.2
    np.testing.assert_allclose(actions["linkage_correction"], 0, atol=1e-15)


def test_direct_author_and_profile_product_sums():
    b = bank(2)
    b["source_comments"][1::2, :2] += 0.4
    b["source_comments"][1::2, 2:] -= 0.3
    x = np.array([[[0.2], [0.7]], [[0.8], [0.1]]])
    r = M.pair_readings(b, x, "E5", "registered_windows")
    z, g = b["source_comments"], b["codes"]
    for p in range(len(x)):
        ka = np.array([np.exp(-((x[p, 0] - a[:2]) ** 2) / 2).mean() for a in z])
        kb = np.array([np.exp(-((x[p, 1] - a[2:]) ** 2) / 2).mean() for a in z])
        j = np.array([(ka[g == c] * kb[g == c]).mean() for c in range(32)])
        i = np.array([ka[g == c].mean() * kb[g == c].mean() for c in range(32)])
        direct = np.stack((j / j.sum(), i / i.sum()))
        np.testing.assert_allclose(r["profile"][p], direct, atol=1e-15)


def test_empirical_prior_and_identical_singletons():
    b = bank(2)
    b["source_comments"][1::2, :2] += 0.5
    ell = np.stack(
        [A.author_log_law(np.array([[0.3]]), b["source_comments"][:, :2], 1)] * 2, 1
    )
    r = M.profile_laws(ell, b["codes"])
    np.testing.assert_allclose(r["profile_prior"], 1 / 32)
    one = A.readings(ell[:, 0], b["codes"], b["anchors"])
    # A missing partner has integrated likelihood one in BOTH declared laws.
    integrated = M.profile_laws(
        np.stack((ell[:, 0], np.zeros_like(ell[:, 0])), 1), b["codes"]
    )
    np.testing.assert_allclose(
        integrated["targets"][:, 0], one["probability"], atol=1e-15
    )
    np.testing.assert_allclose(
        integrated["targets"][:, 1], one["probability"], atol=1e-15
    )


def test_pooled_symmetry_directed_slot_sensitivity():
    b = bank(2)
    b["source_comments"][:, 2:] = 0.4 * b["source_comments"][:, 2:] + 0.8
    p = M.corners(endpoints()).reshape(8, 2, 1)
    pooled = M.pair_readings(b, p, "E5", "pooled_comments")["targets"].reshape(
        2, 4, 2, 5
    )
    ordered = M.pair_readings(b, p, "E5", "registered_windows")["targets"].reshape(
        2, 4, 2, 5
    )
    np.testing.assert_allclose(pooled[0], pooled[1], atol=1e-15)
    assert np.max(abs(ordered[0] - ordered[1])) > 0.01


def test_shared_author_equivalence_for_current_cold_law():
    b, p = bank(2), np.array([[[0.2], [0.8]]])
    r = M.pair_readings(b, p, "E5", "pooled_comments")
    direct = A.readings(
        A.author_log_law(p[0], b["source_comments"], 1).sum(0)[None],
        b["codes"],
        b["anchors"],
    )
    np.testing.assert_allclose(r["targets"][:, 0], direct["probability"], atol=1e-15)


def test_all_signed_cross_contacts_and_exact_reconstruction():
    rng = np.random.default_rng(270)
    t = rng.random((3, 2, 4, 2, 5))
    a = M.mixed_actions(t)
    np.testing.assert_allclose(
        a["linked"], a["profile_product"] + a["linkage_correction"], atol=1e-15
    )
    y = rng.normal(size=(3, 5))
    s = a["linkage_correction"][:, 0]
    np.testing.assert_allclose(
        M.contact(s, y), np.mean(s[:, :, None] * y[:, None, :], 0)
    )


def record():
    """A cold numeric edit with no labels or outcomes."""
    return {
        "frame_key": "x",
        "endpoints": [
            [
                {"event_id": f"{w}-{s}", "vector": endpoints()[0, w, s].tolist()}
                for s in range(2)
            ]
            for w in range(2)
        ],
    }


def test_cold_fields_and_duplicate_guard():
    r = record()
    result = M.cold(bank(), r)
    assert len(result["readings"]) == 4
    r["endpoints"][1][0]["event_id"] = "0-0"
    with pytest.raises(ValueError, match="distinct"):
        M.cold(bank(), r)


@pytest.mark.parametrize("extra", ["answer", "label", "text", "trait"])
def test_cold_rejects_answers(extra):
    r = record()
    r["endpoints"][0][0][extra] = 1
    with pytest.raises(ValueError, match="answers"):
        M.cold(bank(), r)


def test_missing_profiles_and_invalid_points_rejected():
    with pytest.raises(ValueError, match="support"):
        M.profile_laws(np.zeros((1, 2, 2)), np.array([0, 31]))
    with pytest.raises(ValueError, match="complete|Complete"):
        M.pair_readings(bank(), np.zeros((1, 1, 1)), "E5", "pooled_comments")
