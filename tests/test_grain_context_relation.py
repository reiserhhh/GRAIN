#!/usr/bin/env python3
"""Test exact evidence contacts,opportunities and strict cold identities."""

import copy

import numpy as np
import pytest

from grain_research.construction import math_context_relation as C
from grain_research.construction import math_joint_evidence as J
from grain_research.construction import math_joint_meaning as M


def source():
    """Small designed SOURCE law;not simulated psychological validity."""
    rng = np.random.default_rng(178)
    p = rng.uniform(0.1, 2, (8, 2, 2, 8))
    p /= p.sum(-1, keepdims=True)
    return p, np.repeat([[0] * 5, [1] * 5], 4, axis=0)


def test_source_view_laws_keep_profile_and_pooled_margins():
    p, y = source()
    b = C.validate_bundle(C.build_bundle(p, y))
    t = np.asarray(b["profile_tables"])
    for v, z0, z1 in np.ndindex(2, 2, 2):
        slices = (slice(4 * z0, 4 * z0 + 4), slice(4 * z1, 4 * z1 + 4))
        a, f = t[v, 1, :, slices[0], slices[1]], t[v, 2, :, slices[0], slices[1]]
        for x, z in zip(M.margins(a), M.margins(f), strict=True):
            np.testing.assert_allclose(x, z, atol=1e-10, rtol=0)


def test_direct_context_contacts_equal_subset_contrast():
    f = np.random.default_rng(8).normal(size=(3, 5, 8, 8))
    cells = np.arange(12) % 8
    s, c = np.array([0, 1, 6, 7]), np.array([2, 3, 4, 8])
    v = np.array([C.pair_sum(f, cells[ix], ix)[0] for ix in J.group_indices(s, c)])
    np.testing.assert_allclose(
        C.context_action(v), C.contact_action(f, cells, s, c), atol=1e-16
    )


def test_constant_field_can_have_opportunity_action():
    s, c = np.array([0, 1, 2, 3]), np.array([4, 5, 6, 7])
    cells = np.zeros(12, int)
    f = np.ones((8, 8))
    values = np.array([C.pair_sum(f, cells[ix], ix)[0] for ix in J.group_indices(s, c)])
    assert C.context_action(values) != 0
    assert C.context_action(values) == pytest.approx(C.contact_action(f, cells, s, c))


def test_singleton_sums_cancel_but_contextual_pair_values_do_not():
    s, c = np.array([0, 1, 6, 7]), np.array([2, 3, 4, 8])
    cells = np.arange(12) % 8
    vals = np.arange(12, dtype=float) ** 2
    singleton = np.array([vals[ix].sum() / 12 for ix in J.group_indices(s, c)])
    assert C.context_action(singleton) == pytest.approx(0, abs=1e-14)
    field = np.eye(8)
    pair = np.array(
        [C.pair_sum(field, cells[ix], ix)[0] for ix in J.group_indices(s, c)]
    )
    assert abs(C.context_action(pair)) > 0


def test_no_pair_is_zero_mass_but_missing_conditional_average():
    mass, avg, known, n = C.pair_sum(
        np.ones((2, 8, 8)), np.array([0, 1]), np.array([0, 1])
    )
    assert n == 0 and known.all() and not mass.any() and np.isnan(avg).all()


def test_unobserved_active_pair_remains_missing():
    field = np.ones((8, 8))
    field[1, 2] = np.nan
    mass, avg, known, n = C.pair_sum(field, np.array([1, 2]), np.array([0, 6]))
    assert n == 1 and not known and np.isnan(mass) and np.isnan(avg)


@pytest.mark.parametrize("slots", ([0, 0], [0, 12], [-1, 6]))
def test_invalid_slot_identity_rejected(slots):
    with pytest.raises(ValueError):
        C.pair_sum(np.ones((8, 8)), np.array([0, 1]), np.array(slots))


def test_event_order_does_not_change_addressed_pair_mass():
    field = np.arange(64.0).reshape(8, 8)
    a = C.pair_sum(field, np.array([1, 2, 3]), np.array([0, 6, 7]))[0]
    b = C.pair_sum(field, np.array([3, 1, 2]), np.array([7, 0, 6]))[0]
    assert a == b


def test_cold_rejects_outcome_duplicate_and_false_integer():
    p, y = source()
    b = C.build_bundle(p, y)
    r = {
        "author_key": "app/0",
        "view": 0,
        "events": [
            {"event_id": "a", "slot": 0, "cell": 0},
            {"event_id": "b", "slot": 6, "cell": 1},
        ],
    }
    assert C.cold_record(b, r)["pairs"] == 1
    for name in ("outcome", "duplicate", "boolean"):
        bad = copy.deepcopy(r)
        if name == "outcome":
            bad["target"] = 1
        elif name == "duplicate":
            bad["events"][1]["event_id"] = "a"
        else:
            bad["events"][1]["slot"] = True
        with pytest.raises(ValueError):
            C.cold_record(b, bad)


def test_bundle_extra_fitting_and_field_tamper_rejected():
    p, y = source()
    b = C.build_bundle(p, y)
    bad = copy.deepcopy(b)
    bad["native_fit"] = False
    with pytest.raises(ValueError):
        C.validate_bundle(bad)
    bad = copy.deepcopy(b)
    bad["pair_fields"][0][0][0][0][0] += 0.1
    with pytest.raises(ValueError):
        C.validate_bundle(bad)


def test_signed_agreement_preserves_zero_and_opposite_values():
    r = C.signed_agreement([1, -1, 0, np.nan], [1, 1, 2, 3])
    assert r["observations"] == 3 and r["missing"] == 1
    assert r["same_sign"] == r["opposite_sign"] == 1
    assert C.signed_agreement([0, 0], [1, 2])["cosine"] is None
