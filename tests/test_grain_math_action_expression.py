#!/usr/bin/env python3
"""Tests that can fail without a model or any psychological ground truth."""

from collections import Counter

import numpy as np
import pytest

from grain_research.construction import math_action_expression as M


def test_actor_even_and_odd_reconstruct_without_erasing_foreign():
    rng = np.random.default_rng(115)
    x = rng.normal(size=(3, 2, 6, 2, 2, 2, 4))
    f = M.actor_fields(x)
    np.testing.assert_allclose(f["own"], f["even"] + f["odd"])
    np.testing.assert_allclose(f["other"], f["even"] - f["odd"])
    direct = (x[..., 1, 1, :] - x[..., 0, 1, :]) / 2
    np.testing.assert_allclose(f["own"], direct)


def test_assignment_odd_vanishes_for_role_blind_content():
    x = np.zeros((2, 2, 2, 3))
    x[:, 1] = [1, 2, 3]
    f = M.actor_fields(x)
    np.testing.assert_allclose(f["odd"], 0)
    np.testing.assert_allclose(f["even"], [np.array([1, 2, 3]) / 2] * 2)


def test_assignment_swap_reverses_odd_not_even():
    x = np.random.default_rng(15).normal(size=(6, 2, 2, 2, 7))
    a, b = M.actor_fields(x), M.actor_fields(x[..., ::-1, :])
    np.testing.assert_allclose(a["even"], b["even"])
    np.testing.assert_allclose(a["odd"], -b["odd"])


def test_common_offsets_do_not_create_direction_transport():
    x = np.random.default_rng(3).normal(size=(6, 2, 2, 2, 4))
    offset = np.random.default_rng(4).normal(size=(6, 2, 1, 1, 4))
    a, b = M.actor_fields(x), M.actor_fields(x + offset)
    for key in a:
        np.testing.assert_allclose(a[key], b[key], atol=1e-14)


def test_identity_and_wrong_label_control():
    x = np.repeat(np.eye(6)[:, None], 2, axis=1)
    same = M.direction_match(x, x)
    assert same["matched_mean_cosine"] == 1
    assert same["matching_rank_with_ties"] == 1
    wrong = M.direction_match(x, x[::-1])
    assert wrong["matched_mean_cosine"] == 0
    assert wrong["matched_minus_permutation_mean"] < 0


def test_zero_direction_is_not_a_zero_cosine():
    x = np.repeat(np.eye(6)[:, None], 2, axis=1)
    x[2, 1] = 0
    result = M.direction_match(x, x)
    assert result["status"] == "UNAVAILABLE_DIRECTION"
    assert result["matched_mean_cosine"] is None


def test_negative_direction_remains_negative():
    x = np.repeat(np.eye(6)[:, None], 2, axis=1)
    assert M.direction_match(x, -x)["matched_mean_cosine"] == -1


def test_lexical_control_has_no_owner_channel():
    pairs = [[[[f], [f, f + 10]] for f in range(6)] for _ in range(3)]
    gains, vocabulary = M.lexical_fields(pairs)
    assert len(vocabulary) == 12
    assert gains.shape == (3, 6, 2, 12)
    np.testing.assert_allclose(gains[:, :, 0], gains[:, :, 1])


def test_exact_author_contrast_matches_positive_event_probabilities():
    x = np.random.default_rng(7).normal(size=(6, 2, 2, 2, 5))
    field = M.actor_fields(x)["own"]
    direct = []
    for family in range(6):
        value = np.zeros(5)
        for c in range(2):
            low, high = x[family, c, :, 1]
            plus = 0.3 * low + 0.7 * high
            minus = 0.7 * low + 0.3 * high
            value += (plus - minus) / 12
        direct.append(value)
    np.testing.assert_allclose(M.author_contrasts(field), direct, atol=1e-14)


def test_material_matches_actor_conditions_and_owner_token_bags():
    spec = {
        "conditions": [["was asked", "was not asked"]],
        "behaviours": [[["stood aside", "joined in"]]],
    }
    a = M.material(spec, 0, 0, 0, 1, 1, 1)
    b = M.material(spec, 0, 0, 0, 1, 1, -1)
    assert a["condition_text"] == b["condition_text"]
    assert a["condition_text"].count("was not asked") == 2
    assert Counter(a["action_text"].split()) == Counter(b["action_text"].split())


@pytest.mark.parametrize("shape", [(2, 2, 3), (2, 3, 2, 4)])
def test_incomplete_cube_fails(shape):
    with pytest.raises(ValueError):
        M.actor_fields(np.ones(shape))
