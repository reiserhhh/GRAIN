#!/usr/bin/env python3
"""Action-transport tests, including a positive and a falsifiable toy world."""

import copy

import numpy as np
import pytest

from grain_research.construction import math_transfer_update as M


def world(reverse=False):
    """Same singleton laws, with a target-specific within-person pair relation."""
    codes = np.repeat(np.arange(32), 8)
    folds = np.tile(np.repeat([0, 1], 4), 32)
    p = np.zeros((len(codes), 2, 2, 8))
    for u, (c, f) in enumerate(zip(codes, folds, strict=True)):
        early = u % 2
        late = early if c & 1 else 1 - early
        if reverse and f == 1:
            late = 1 - late
        p[u, :, 0, early] = 1
        p[u, :, 1, late] = 1
    return p, codes, folds, np.full(32, 1 / 32)


def test_law_control_fixes_all_profile_singletons_not_only_label_means():
    p, c, f, pi = world()
    laws, _, _ = M.compile_fields(p, c, f, pi)
    np.testing.assert_allclose(laws[:, :, 0].sum(-1), laws[:, :, 1].sum(-1))
    np.testing.assert_allclose(laws[:, :, 0].sum(-2), laws[:, :, 1].sum(-2))
    np.testing.assert_allclose(laws.sum((-1, -2)), np.broadcast_to(pi, (2, 2, 2, 32)))


def test_shared_relation_transports_without_individual_histogram_catalogue():
    p, c, f, pi = world()
    _, h, support = M.compile_fields(p, c, f, pi)
    rho = np.zeros((8, 8))
    rho[:2, :2] = 0.25
    a = M.transport(h, support, rho)
    assert a["primary"] > 0.04
    np.testing.assert_allclose(a["by_target"][0], 0.25)
    np.testing.assert_allclose(a["coherence"][0], 1)
    np.testing.assert_allclose(a["by_condition"].sum(0), a["by_target"])
    np.testing.assert_allclose(np.diag(a["signed_cross_gram"]), a["by_target"])


def test_recurrent_author_relation_can_fail_between_author_banks():
    p, c, f, pi = world(reverse=True)
    _, h, support = M.compile_fields(p, c, f, pi)
    rho = np.zeros((8, 8))
    rho[:2, :2] = 0.25
    a = M.transport(h, support, rho)
    assert a["within_bank_disjoint_text"][0] > 0.24
    assert a["by_target"][0] < -0.24
    assert np.linalg.eigvalsh(a["signed_cross_gram"]).min() < 0


def test_update_and_covariance_identity_do_not_need_product_independence():
    p, c, _, pi = world()
    v = M.update_actions(M.source_laws(p[:, 0], c, pi))
    direct = np.divide(
        v["gamma_numerator"],
        v["pair_mass"][:, None],
        out=np.full_like(v["gamma_numerator"], np.nan),
        where=v["pair_mass"][:, None] > 0,
    )
    np.testing.assert_allclose(direct, v["increment"], atol=2e-15)
    np.testing.assert_allclose(v["before"][0], v["before"][1])


def test_no_author_variation_inside_profile_gives_no_relation_increment():
    p, c, _, pi = world()
    for g in range(32):
        p[c == g] = p[c == g].mean(0)
    v = M.update_actions(M.source_laws(p[:, 0], c, pi))
    np.testing.assert_allclose(v["relation"][:, :2, :2], 0)


def test_matched_null_preserves_singletons_and_late_quarter_assignment():
    p, c, f, pi = world()
    x, permutation = M.permute_late(p, c, f, np.random.default_rng(192))
    np.testing.assert_array_equal(x[:, :, 1], p[permutation, :, 1])
    np.testing.assert_array_equal(c[permutation], c)
    np.testing.assert_array_equal(f[permutation], f)
    before, _, _ = M.compile_fields(p, c, f, pi)
    after, _, _ = M.compile_fields(x, c, f, pi)
    np.testing.assert_allclose(before[:, :, 0], after[:, :, 0])
    np.testing.assert_allclose(before[:, :, 1].sum(-1), after[:, :, 1].sum(-1))
    np.testing.assert_allclose(before[:, :, 1].sum(-2), after[:, :, 1].sum(-2))


def test_compiled_pair_query_requires_distinct_events_and_no_author_outcomes():
    p, c, _, pi = world()
    bundle = M.make_bundle(M.source_laws(p[:, 0], c, pi))
    events = [
        {"event_id": "first", "window": 0, "cell": 0},
        {"event_id": "second", "window": 1, "cell": 0},
    ]
    a = M.query_pair(bundle, events)
    assert a["supported"]
    np.testing.assert_allclose(a["relation_increment"][0], 0.5)
    events[1]["event_id"] = "first"
    with pytest.raises(ValueError, match="same event"):
        M.query_pair(bundle, events)
    other = copy.deepcopy(bundle)
    other["source_author_ids"] = [0]
    with pytest.raises(ValueError, match="catalogue"):
        M.query_pair(other, events)


def test_missing_pairs_are_missing_and_reference_mass_not_renormalized():
    p, c, f, pi = world()
    laws, h, support = M.compile_fields(p, c, f, pi)
    rho = np.full((8, 8), 1 / 64)
    a = M.transport(h, support, rho)
    assert a["supported_pair_count"] == 4
    assert a["reference_mass"] == 4 / 64
    v = M.update_actions(laws[0, 0])
    assert np.isnan(v["relation"][:, 7, 7]).all()


def test_cell_distributions_are_normalized_and_empty_inputs_rejected():
    p = M.distributions(np.array([[[0, 1, 0], [4, 5, 4]]]))
    np.testing.assert_allclose(p.sum(2), 1)
    with pytest.raises(ValueError, match="Empty"):
        M.distributions(np.zeros((1, 2, 0), int))
