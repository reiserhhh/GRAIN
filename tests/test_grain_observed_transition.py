#!/usr/bin/env python3
"""Synthetic tests for matching and identified transition readings."""

import copy

import numpy as np
import pytest

from grain_research.construction import math_observed_transition as M


def example_law():
    """Use a full-support lawful target/current/continuation association."""
    j = np.ones((32, 8, 8))
    sign = 2 * M.BITS[:, 0] - 1
    j[:, :, :4] *= (1 + 0.4 * sign)[:, None, None]
    j[:, :, 4:] *= (1 - 0.4 * sign)[:, None, None]
    return j / j.sum()


def example_record():
    """Supply only registered numeric observations with distinct identities."""
    event = {
        "author_key": "u",
        "event_id": "a",
        "timestamp": 1,
        "venue_code": 0,
        "condition": 0,
        "coordinates": [0.0] * 15,
    }
    later = event | {"event_id": "b", "timestamp": 2}
    return {
        "regime": M.REGIME,
        "representation_id": "fixed",
        "current": event,
        "next": later,
    }


def test_matching_and_screened_update():
    """The comparator changes association,not either observable or singleton law."""
    j = example_law()
    i = M.screened_law(j)
    np.testing.assert_allclose(i.sum(-1), j.sum(-1), atol=1e-14)
    np.testing.assert_allclose(i.sum(-3), j.sum(-3), atol=1e-14)
    jf, sf = M.target_fields(j), M.target_fields(i)
    np.testing.assert_allclose(jf["before"], sf["before"], atol=1e-14)
    np.testing.assert_allclose(sf["increment"], 0, atol=1e-14)
    assert jf["increment"][0, 0, 0] > 0
    assert jf["increment"][0, 0, 4] < 0


def test_literal_query_with_same_current_opposite_observed_continuation():
    """An observed continuation can change direction while the current state is fixed."""
    bundle = M.compile_bundle(example_law(), [0.0, 0.0], "fixed")
    r = example_record()
    same = M.query(bundle, r)
    other = copy.deepcopy(r)
    other["next"]["venue_code"] = 1
    changed = M.query(bundle, other)
    assert same["before"] == changed["before"]
    assert same["increment"][0] < 0 < changed["increment"][0]
    j = example_law()
    np.testing.assert_allclose(same["profile_after"], j[:, 0, 4] / j[:, 0, 4].sum())


def test_crossed_agreement_retains_negative_action():
    """An opposite author bank is negative,not rotated or clipped into success."""
    f = M.target_fields(np.broadcast_to(example_law(), (2, 2, 32, 8, 8)))
    rho = np.ones((8, 8)) / 64
    positive = M.agreement(f["increment"], f["supported"], rho)
    h = f["increment"].copy()
    h[1] *= -1
    negative = M.agreement(h, f["supported"], rho)
    assert positive["by_target"][0] > 0 > negative["by_target"][0]
    np.testing.assert_allclose(positive["by_target"], -negative["by_target"])


def test_missing_support_is_not_renormalized_or_neutral():
    """Unknown transition support invalidates full primary;it is not a zero reading."""
    j = example_law()
    j[:, 0, 0] = 0
    j /= j.sum()
    f = M.target_fields(np.broadcast_to(j, (2, 2, 32, 8, 8)))
    stat = M.agreement(f["increment"], f["supported"], np.ones((8, 8)) / 64)
    assert not stat["full_primary_available"]
    assert stat["reference_mass"] == 63 / 64
    bundle = M.compile_bundle(j, [0.0, 0.0], "fixed")
    r = example_record()
    r["next"]["venue_code"] = 1
    result = M.query(bundle, r)
    assert result["status"] == "UNOBSERVED_TRANSITION"
    assert result["increment"] is None


@pytest.mark.parametrize(
    "kind", ["duplicate", "author", "time", "label", "representation"]
)
def test_wrong_provenance_or_hidden_target_rejected(kind):
    """Input identity checks cannot be bypassed by silently adding target information."""
    b = M.compile_bundle(example_law(), [0.0, 0.0], "fixed")
    r = example_record()
    if kind == "duplicate":
        r["next"]["event_id"] = "a"
    elif kind == "author":
        r["next"]["author_key"] = "v"
    elif kind == "time":
        r["next"]["timestamp"] = 0
    elif kind == "label":
        r["current"]["target_score"] = 1
    else:
        r["representation_id"] = "changed"
    with pytest.raises(ValueError):
        M.query(b, r)


def test_author_equal_compilation_preserves_full_profile_prior():
    """A long comment bag cannot change a profile's registered source weight."""
    g, folds = np.tile(np.arange(32), 2), np.repeat([0, 1], 32)
    a = np.zeros((64, 2, 17), int)
    o = np.broadcast_to((g % 2)[:, None, None], a.shape)
    p = M.frequencies(a, o)
    j, _ = M.compile_laws(p, g, folds, np.ones(32) / 32)
    np.testing.assert_allclose(j.sum((-2, -1)), np.ones((2, 2, 32)) / 32)
    with pytest.raises(ValueError):
        M.compile_laws(p[:-1], g[:-1], folds[:-1], np.ones(32) / 32)


def test_full_profile_relation_can_exist_with_zero_all_five_target_updates():
    """Five unchanged target means do not imply absence of joint-profile information."""
    j = np.ones((32, 8, 8))
    parity = np.prod(2 * M.BITS - 1, axis=1)
    j[:, :, :4] *= (1 + 0.4 * parity)[:, None, None]
    j[:, :, 4:] *= (1 - 0.4 * parity)[:, None, None]
    j /= j.sum()
    np.testing.assert_allclose(M.target_fields(j)["increment"], 0, atol=1e-14)
    assert np.max(np.abs(j - M.screened_law(j))) > 0
