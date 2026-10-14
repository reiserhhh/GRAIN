#!/usr/bin/env python3
"""Finite observation-unit and author-integration tests, no data dependency."""

import numpy as np
import pytest

from grain_research.construction import math_author_observation_channel as M
from grain_research.construction import math_joint_evidence as J


def test_round_trip_is_not_tower_identity():
    """Nonnested conditional expectations attenuate even a binary target."""
    p = np.array([[0.8, 0.2], [0.2, 0.8]])
    y = np.array([[0.0], [1.0]])
    c, a = M.channel(p), M.target_accounts(p, y)
    np.testing.assert_allclose(c["round_trip"], [[0.68, 0.32], [0.32, 0.68]])
    np.testing.assert_allclose(a["return_scores"][:, 0], [-0.18, 0.18])
    np.testing.assert_allclose(c["centered_spectrum"], [0.36, 0], atol=1e-15)
    assert a["literal_return_error"] < 1e-15


def test_indistinguishable_authors_and_perfect_observations():
    """Channels range from complete author alias to perfect finite identity."""
    p = np.tile([0.4, 0.6], (5, 1))
    np.testing.assert_allclose(M.channel(p)["round_trip"], np.ones((5, 5)) / 5)
    np.testing.assert_allclose(M.channel(np.eye(5))["round_trip"], np.eye(5))


def test_contraction_and_source_energy_order():
    """All source cross terms remain while PSD inequalities hold."""
    rng = np.random.default_rng(234)
    p = rng.dirichlet(np.ones(8), 20)
    y = rng.normal(size=(20, 5))
    c, a = M.channel(p), M.target_accounts(p, y)
    np.testing.assert_allclose(c["round_trip"].sum(1), 1, atol=2e-15)
    assert c["centered_spectrum"].max() < 1 + 1e-12
    for x, z in (("label_gram", "field_gram"), ("field_gram", "return_gram")):
        assert np.linalg.eigvalsh(a[x] - a[z]).min() >= -1e-12


def test_pair_selector_garbling_and_blackwell_order():
    """Finite selectors refine the channel without a human-iid assumption."""
    rng = np.random.default_rng(17)
    p0, p1 = rng.dirichlet(np.ones(3), (2, 10))
    q, g = M.window_pair(p0, p1)
    np.testing.assert_allclose(q @ g, (p0 + p1) / 2, atol=1e-15)
    gain = M.channel(q)["round_trip"] - M.channel((p0 + p1) / 2)["round_trip"]
    assert np.linalg.eigvalsh(gain).min() >= -1e-12


def test_missing_cells_preserved_without_smoothing():
    """An unobserved category stays explicit, not a fabricated observation."""
    c = M.channel([[0.3, 0.7, 0], [0.6, 0.4, 0]])
    assert c["support"].tolist() == [True, True, False]
    assert c["reference"][-1] == 0
    for bad in ([[0.1, 0.1]], [[1.1, -0.1], [0.4, 0.6]]):
        with pytest.raises(ValueError):
            M.channel(bad)


def bundle():
    """Return a small complete catalogue with five linked binary labels."""
    p = np.full((4, 8), 0.01)
    p[0, 0], p[1, 1], p[2, 0], p[3, 1] = 0.71, 0.71, 0.36, 0.36
    p[2, 1], p[3, 0] = 0.36, 0.36
    p /= p.sum(1, keepdims=True)
    y = np.array([[0] * 5, [1] * 5, [0] * 5, [1] * 5])
    return J.build_bundle(p, y)


def test_one_event_equivalence_and_distinct_event_repetition():
    """All rules agree on one event; equal cells do not imply duplicate IDs."""
    b = bundle()
    for cell in range(8):
        v = M.cold_history(b, [cell], ["a"])["probabilities"]
        np.testing.assert_allclose(v, np.tile(v[0], (3, 1)), atol=1e-15)
    v = M.cold_history(b, [0, 0], ["a", "b"])
    assert v["events"] == 2
    assert np.max(abs(v["probabilities"][0] - v["probabilities"][2])) > 0.01
    with pytest.raises(ValueError):
        M.cold_history(b, [0, 0], ["a", "a"])


def test_history_masks_keep_choice_cells():
    """All5 groups preserve original cells, including the condition bit."""
    masks = np.ones((3, 12), bool)
    masks[1, :4], masks[2, 4:8] = False, False
    cells = np.arange(12) % 8
    groups = M.history_groups(cells, masks, [str(i) for i in range(12)])
    assert [len(a) for a, _ in groups] == [12, 8, 8, 4, 4]
    assert groups[3][0].tolist() == [0, 1, 2, 3]
    assert groups[4][0].tolist() == [4, 5, 6, 7]
    masks[2, :4] = False
    with pytest.raises(ValueError):
        M.history_groups(cells, masks, [str(i) for i in range(12)])


def test_unsupported_shared_history_is_not_zero_filled():
    """Joint support can fail despite each cell having one-event support."""
    p = np.zeros((2, 8))
    p[0, :4], p[1, 4:] = 0.25, 0.25
    b = J.build_bundle(p, np.array([[0] * 5, [1] * 5]))
    r = M.cold_history(b, [0, 4], ["a", "b"])
    assert not r["posterior_supported"][1].any()
    assert np.isnan(r["probabilities"][2]).all()
    assert r["effective_source_authors"] is None


def test_target_inversion_changes_meaning_not_author_weights():
    """Observation likelihood remains distinct from the attached target meaning."""
    b = bundle()
    inverse = J.build_bundle(
        np.asarray(b["source_mass"]), 1 - np.asarray(b["source_binary_targets"])
    )
    a, z = (M.cold_history(k, [0, 0, 1], ["a", "b", "c"]) for k in (b, inverse))
    np.testing.assert_allclose(a["author_weights"], z["author_weights"])
    np.testing.assert_allclose(a["probabilities"], 1 - z["probabilities"])
