#!/usr/bin/env python3
"""Test lower-observation invariance and meaningful cold evidence composition."""

import copy
import itertools

import numpy as np
import pytest

from grain_research.construction import math_joint_evidence as J
from grain_research.construction import math_ternary_meaning as T


def source():
    """Two full profiles, four normalized authors per profile, two fixed views."""
    rng = np.random.default_rng(180)
    p = rng.uniform(0.2, 2, (8, 2, 3, 8))
    p /= p.sum(-1, keepdims=True)
    return p, np.repeat([[0] * 5, [1] * 5], 4, axis=0)


def test_all_full_profile_pairs_and_pooled_content_are_identical():
    f = np.random.default_rng(18).uniform(0.2, 2, (3, 3, 3, 3))
    f /= f.sum()
    h, receipt = T.homogeneous(f)
    for a, b in zip(T.margins(f), T.margins(h), strict=True):
        np.testing.assert_allclose(a, b, atol=1e-10, rtol=0)
    assert receipt["sweeps"] < 10000


def test_parity_target_is_invisible_in_every_pair_but_visible_in_triple():
    f = np.zeros((2, 2, 2, 2))
    for g, a, b, c in itertools.product(range(2), repeat=4):
        f[g, a, b, c] = 0.1 if g == (a ^ b ^ c) else 0.025
    h, _ = T.homogeneous(f)
    np.testing.assert_allclose(h, 0.0625, atol=1e-12)
    c = np.array([[0] * 5, [1] * 5])
    psi = T.fields(c, np.stack((h, f)))
    np.testing.assert_allclose(np.abs(psi[1] - psi[0]), 0.3, atol=1e-12)
    for axis in (1, 2, 3):
        np.testing.assert_allclose(f.sum(axis), h.sum(axis), atol=1e-12)


def test_reference_is_identity_for_a_declared_lower_order_loglinear_law():
    rng = np.random.default_rng(181)
    f = np.exp(
        rng.normal(size=(2, 2, 2, 1))
        + rng.normal(size=(2, 2, 1, 2))
        + rng.normal(size=(2, 1, 2, 2))
        + rng.normal(size=(1, 2, 2, 2))
    )
    f /= f.sum()
    h, _ = T.homogeneous(f)
    np.testing.assert_allclose(h, f, atol=3e-9, rtol=0)


def test_label_independent_content_does_not_create_target_meaning():
    abc = np.random.default_rng(19).uniform(0.1, 2, (2, 2, 2))
    abc /= abc.sum()
    f = np.array([0.4 * abc, 0.6 * abc])
    h, _ = T.homogeneous(f)
    np.testing.assert_allclose(
        T.fields([[0] * 5, [1] * 5], np.stack((h, f))), 0.6, atol=1e-12
    )


def test_empty_and_structural_zero_support_is_not_filled():
    h, r = T.homogeneous(np.zeros((2, 4, 4, 4)))
    assert not h.any() and r["sweeps"] == 0
    f = np.zeros_like(h)
    f[:, 0, 0, 0] = [0.2, 0.8]
    h, _ = T.homogeneous(f)
    np.testing.assert_array_equal(f, h)


def test_fixed_solver_failure_is_not_rescued():
    f = np.random.default_rng(8).uniform(0.1, 2, (2, 3, 3, 3))
    with pytest.raises(ValueError, match="budget exhausted"):
        T.homogeneous(f, 1e-15, 1)


def test_bundle_is_coherent_and_target_kl_is_not_joint_information_fraction():
    p, y = source()
    b = T.build_bundle(p, y)
    T.validate_bundle(b)
    for t in np.asarray(b["profile_tables"]):
        phi = T.profile_fields(t)
        np.testing.assert_allclose(phi.sum(1), 1, atol=1e-12)
        stat = T.statistics(b["profiles"], t)
        assert max(stat["target_kl_nats"]) <= stat["joint_kl_nats"] + 1e-8


def test_cold_profile_marginals_equal_all_five_targets():
    p, y = source()
    b = T.build_bundle(p, y)
    r = {
        "author_key": "x",
        "view": 0,
        "events": [{"event_id": str(s), "slot": s, "cell": s % 8} for s in (0, 4, 8)],
    }
    v = T.cold_record(b, r)
    np.testing.assert_allclose(
        np.array(v["conditional_profile_average"]) @ b["profiles"],
        v["conditional_target_average"],
    )
    assert v["triples"] == 1


@pytest.mark.parametrize(
    "kind", ("duplicate_id", "duplicate_slot", "outcome", "float_cell", "bad_view")
)
def test_cold_rejects_unqualified_events_and_outcomes(kind):
    p, y = source()
    b = T.build_bundle(p, y)
    r = {
        "author_key": "x",
        "view": 0,
        "events": [{"event_id": str(s), "slot": s, "cell": s % 8} for s in (0, 4, 8)],
    }
    if kind == "duplicate_id":
        r["events"][1]["event_id"] = "0"
    elif kind == "duplicate_slot":
        r["events"][1]["slot"] = 0
    elif kind == "outcome":
        r["target"] = 1
    elif kind == "float_cell":
        r["events"][0]["cell"] = 0.0
    else:
        r["view"] = True
    with pytest.raises(ValueError):
        T.cold_record(b, r)
    bad = copy.deepcopy(b)
    bad["outcome"] = 1
    with pytest.raises(ValueError):
        T.validate_bundle(bad)


def test_no_opportunity_is_not_neutral_personality():
    p, y = source()
    v = T.cold_record(
        T.build_bundle(p, y), {"author_key": "x", "view": 0, "events": []}
    )
    assert v["status"] == "NO_TRIPLE_OPPORTUNITY"
    assert np.asarray(v["target_mass_readout"]).sum() == 0
    assert v["conditional_target_average"] == [[None] * 5, [None] * 5]


def test_exact_context_contact_rule_counts_opportunity_not_independent_trials():
    field = np.random.default_rng(9).uniform(size=(2, 5, 8, 8, 8))
    cells = np.arange(12) % 8
    s, c = [0, 4, 5, 9], [1, 6, 8, 10]
    values = np.array(
        [T.triple_sum(field, cells[ids], ids)[0] for ids in J.group_indices(s, c)]
    )
    action = values[2] - values[1] - values[3] + values[4]
    np.testing.assert_allclose(action, T.contact_action(field, cells, s, c), atol=1e-15)


def test_crossfit_folds_are_author_disjoint_and_preserve_profiles():
    _, y = source()
    ids = np.array([9, 2, 5, 3, 10, 14, 12, 15])
    folds = T.source_folds(ids, y)
    assert set(ids[folds == 0]).isdisjoint(ids[folds == 1])
    assert (
        len(np.unique(y[folds == 0], axis=0))
        == len(np.unique(y[folds == 1], axis=0))
        == 2
    )
