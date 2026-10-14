#!/usr/bin/env python3
"""Finite source isolation, matched controls and actor/author sanity checks."""

import numpy as np
import pytest

from grain_research.construction import math_actor_action as M


def test_source_never_uses_application_values():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(3, 2, 6, 2, 2, 2, 3, 2, 4))
    first = M.source_actions(x)
    x[1:] = 1e9
    x[0, 1] = -1e9
    second = M.source_actions(x)
    for k in first:
        np.testing.assert_array_equal(first[k], second[k])


def test_control_head_norms_and_seed_are_fixed():
    x = np.random.default_rng(4).normal(size=(3, 2, 6, 2, 2, 2, 3, 2, 4))
    a = M.source_actions(x)
    for k, v in a.items():
        np.testing.assert_allclose(
            np.linalg.norm(v, axis=(2, 4)), np.linalg.norm(a["source_odd"], axis=(2, 4))
        )
        np.testing.assert_array_equal(v, M.source_actions(x)[k])


def test_actor_split_keeps_generic_vs_selective_separate():
    x = np.zeros((5, 2, 6, 2, 2, 2, 2, 6, 4))
    for b in range(2):
        for o in range(2):
            sign = (2 * b - 1) * (2 * o - 1)
            x[0, :, :, :, b, o, 0] = 3 * sign
            x[0, :, :, :, b, o, 1] = -sign
            x[2, :, :, :, b, o] = 2 * sign
    a = M.effect_accounts(x)
    np.testing.assert_allclose(a["actor_difference"][1, :, :, 6], 2)
    np.testing.assert_allclose(a["actor_common"][1, :, :, 6], 1)
    np.testing.assert_allclose(a["actor_common"][2, :, :, 6], -1)
    np.testing.assert_allclose(a["actor_difference"][0], 0)


def test_direct_author_integral_equals_character():
    x = np.random.default_rng(3).normal(size=(2, 2, 2, 2, 6, 4))
    chars = M.decompose(x.reshape(8, -1)).reshape(8, 2, 6, 4)
    np.testing.assert_allclose(M.author_difference(x), (0.8 / 6) * chars[6], atol=1e-15)


def test_missing_shape_and_zero_direction_fail_closed():
    with pytest.raises(ValueError):
        M.head_match(np.zeros((3, 2, 4)), np.ones((3, 2, 4)))
    with pytest.raises(ValueError):
        M.effect_accounts(np.zeros((4, 2, 6, 2, 2, 2, 2, 6, 4)))


def test_complete_tables_preserve_every_family_actor_and_unit():
    from scripts.run_grain_actor_action_hm116 import result_tables

    x = np.random.default_rng(8).normal(size=(5, 2, 6, 2, 2, 2, 2, 6, 4))
    spec = {"targets": [f"f{i}" for i in range(6)], "author_amplitude": 0.4}
    target, author, summary = result_tables(x, M.effect_accounts(x), spec)
    assert (len(target), len(author), len(summary)) == (2880, 2880, 40)
    assert {r["person"] for r in author} == {"A", "B"}
    assert {r["family"] for r in author} == set(spec["targets"])
    assert {r["unit"] for r in target} == {"score", "log_odds", "mass", "entropy"}
    assert all(
        r["actor_differential_bo"] == 0 for r in target if r["arm"] == "baseline"
    )


def test_actor_integral_reports_person_difference_without_erasing_common():
    x = np.zeros((2, 2, 2, 2, 6, 4))
    for b in range(2):
        for o in range(2):
            chi = (2 * b - 1) * (2 * o - 1)
            x[:, b, o, 0] = 3 * chi
            x[:, b, o, 1] = chi
    got = M.author_difference(x, 0.4)
    np.testing.assert_allclose(got[0], 0.4)
    np.testing.assert_allclose(got[1], 0.4 / 3)
