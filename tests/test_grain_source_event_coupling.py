#!/usr/bin/env python3
"""Paired-source observation laws, not a personality-accuracy benchmark."""

import numpy as np
import pytest

from grain_research.construction import math_source_event_coupling as M


def toy():
    """Supply every profile and source pairs with target-dependent dependence only."""
    codes = np.arange(32)
    prototypes = np.empty((32, 2, 2, 1))
    for i in codes:
        prototypes[i, :, 0, 0] = (-1, 1)
        prototypes[i, :, 1, 0] = (-1, 1) if i & 1 else (1, -1)
    return {
        "prototypes": prototypes,
        "codes": codes,
        "relink": np.tile([1, 0], (32, 1)),
        "variance": np.array(1.0),
    }


def test_pair_information_without_author_or_marginal_information():
    b = toy()
    k = M.event_logs(np.array([[[1.0], [1.0]]]), b["prototypes"], 1)
    result = M.profile_readings(k, b["codes"], b["relink"])
    v = result["targets"]
    assert v[0, 0, 0] > 0.75
    np.testing.assert_allclose(v[0, 1:3], 0.5, atol=1e-15)
    assert v[0, 3, 0] < 0.25


def test_author_information_without_within_author_pair_information():
    codes = np.repeat(np.arange(32), 2)
    prototypes = np.empty((64, 2, 2, 1))
    for i, code in enumerate(codes):
        left = 1 if i % 2 else -1
        prototypes[i, :, 0, 0] = left
        prototypes[i, :, 1, 0] = left if code & 1 else -left
    logs = M.event_logs(np.array([[[1.0], [1.0]]]), prototypes, 1)
    result = M.profile_readings(logs, codes, np.tile([1, 0], (64, 1)))
    v = result["targets"][0]
    np.testing.assert_allclose(v[0], v[1], atol=1e-15)
    assert v[0, 0] > 0.75
    np.testing.assert_allclose(v[2], 0.5, atol=1e-15)


def test_integrated_partner_agrees_for_all_four_laws():
    b = toy()
    k = M.event_logs(np.array([[[0.4], [0.8]]]), b["prototypes"], 1)
    k[:, :, 1] = 0
    r = M.profile_readings(k, b["codes"], b["relink"])
    for law in range(1, 4):
        np.testing.assert_allclose(r["profile"][:, 0], r["profile"][:, law], atol=1e-15)


def test_condition_matched_relink_preserves_every_role_marginal():
    t = np.arange(48).reshape(2, 24)
    c = np.zeros((2, 24), int)
    c[1, 6:10] = 1
    l, r, perm = M.pair_indices(t, c)
    for u in range(2):
        np.testing.assert_array_equal(np.sort(r[perm[u]]), r)
        np.testing.assert_array_equal(c[u, r[perm[u]]], c[u, r])
        np.testing.assert_array_equal(l[perm[u]] // 12, l // 12)
        np.testing.assert_array_equal(c[u, l[perm[u]]], c[u, l])


def test_literal_positive_laws():
    b = toy()
    k = M.event_logs(np.array([[[0.2], [0.7]]]), b["prototypes"], 1)
    r = M.profile_readings(k, b["codes"], b["relink"])
    a, z = np.exp(k[0, :, 0]), np.exp(k[0, :, 1])
    f = (a * z).mean(1)
    j = a.mean(1) * z.mean(1)
    relink = (a * z[:, ::-1]).mean(1)
    expected = np.array([f, j, j, relink])
    expected /= expected.sum(-1, keepdims=True)
    np.testing.assert_allclose(r["profile"][0], expected, atol=1e-15)


def test_components_telescope_without_deleting_cross_terms():
    v = np.random.default_rng(273).random((7, 4, 5))
    c = M.components(v)
    np.testing.assert_allclose(
        c["total_component"], c["pair_component"] + c["author_component"], atol=1e-15
    )


def record():
    """Two declared consecutive records, not evidence of all intervening events."""
    return {
        "frame_key": "a",
        "sampling_regime": "next_cached_record_within_registered_window",
        "events": [
            {
                "event_id": str(i),
                "vector": [1.0],
                "time": i,
                "window": 0,
                "condition": 0,
            }
            for i in range(2)
        ],
    }


def test_cold_repeats_and_scope_guards():
    r = record()
    assert len(M.cold(toy(), r)["laws"]) == 4
    r["events"][1]["event_id"] = "0"
    with pytest.raises(ValueError, match="distinct"):
        M.cold(toy(), r)
    r = record()
    r["events"][1]["window"] = 1
    with pytest.raises(ValueError, match="same-window"):
        M.cold(toy(), r)


@pytest.mark.parametrize("extra", ["label", "answer", "text", "personality"])
def test_cold_rejects_private_or_target_fields(extra):
    r = record()
    r["events"][0][extra] = 1
    with pytest.raises(ValueError, match="labels/answers"):
        M.cold(toy(), r)


def test_missing_profiles_invalid_permutations_and_ties_fail():
    b = toy()
    k = M.event_logs(np.array([[[0.2], [0.7]]]), b["prototypes"], 1)
    with pytest.raises(ValueError, match="support"):
        M.profile_readings(k[:, :-1], b["codes"][:-1], b["relink"][:-1])
    with pytest.raises(ValueError, match="permutation"):
        M.profile_readings(k, b["codes"], np.zeros((32, 2), int))
    t = np.arange(24, dtype=float)[None]
    t[0, 1] = 0
    with pytest.raises(ValueError, match="strict"):
        M.pair_indices(t, np.zeros((1, 24), int))
