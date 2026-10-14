#!/usr/bin/env python3
"""Shared marginal effects and individual relationships must not be conflated."""

import itertools

import numpy as np
import pytest

from grain_research.construction import math_author_coupling as C
from grain_research.construction import math_relation_origin as M


def select(stats, indexes):
    return {name: value[indexes] for name, value in stats.items()}


def world_profiles(mode, seed=20261009):
    _, stats = M.synthetic_world(mode, seed)
    ref = M.fit_reference(select(stats, slice(0, 60)))
    held = select(stats, slice(60, None))
    return stats, held, ref, M.profiles(held, ref)


def average_wins(profiles, arm):
    n = len(profiles[arm]) // 3
    arrays = {name: value.reshape(n, 3, -1) for name, value in profiles.items()}
    means = []
    for left, right in itertools.permutations(range(3), 2):
        distance = C.squared_distances(arrays[arm][:, left], arrays[arm][:, right])
        location = C.squared_distances(
            arrays["joint_location"][:, left], arrays["joint_location"][:, right]
        )
        shape = C.squared_distances(
            arrays["joint_shape"][:, left], arrays["joint_shape"][:, right]
        )
        means.append(M.candidate_wins(distance, location, shape)[0].diagonal().mean())
    return np.mean(means)


def test_shared_linear_shape_recurrence_is_not_personal_relation():
    _, held, ref, p = world_profiles("shared_linear")
    np.testing.assert_allclose(M.components(held, ref)["residual_link"], 0, atol=1e-13)
    assert average_wins(p, "coupling") == 1
    assert average_wins(p, "shared_link") == 1
    assert average_wins(p, "residual_link") == 0.5


def test_personal_relation_repeats_with_identical_marginals():
    _, _, _, p = world_profiles("personal_rotation")
    assert average_wins(p, "joint_location") == 0.5
    assert average_wins(p, "joint_shape") == 0.5
    assert average_wins(p, "residual_link") == 1


@pytest.mark.parametrize("seed", [20261009, 20261010, 20261011])
def test_independent_relations_fail_recurrence(seed):
    _, _, _, p = world_profiles("independent_rotation", seed)
    assert 0.3 < average_wins(p, "residual_link") < 0.7


def test_shared_nonlinear_rule_also_has_repeatable_linear_residual():
    _, held, ref, p = world_profiles("shared_nonlinear")
    assert np.linalg.norm(M.components(held, ref)["residual_link"]) > 0.1
    assert average_wins(p, "residual_link") == 1
    # Every person still uses exactly y = g**2, so no personal rule is needed.


def test_raw_residual_covariance_identity():
    material, stats = M.synthetic_world("personal_rotation", 8)
    ref = M.fit_reference(stats)
    c = M.components(stats, ref)
    g, y = material["a000_p0__g"], material["a000_p0__y"]
    residual = y - g @ ref["link"]
    expected = (g - g.mean(0)).T @ (residual - residual.mean(0)) / len(g)
    np.testing.assert_allclose(c["residual_link"][0], expected, atol=1e-13)


def test_source_within_link_not_total_between_author_regression():
    _, stats = M.synthetic_world("shared_linear", 11)
    moved = {name: values.copy() for name, values in stats.items()}
    offsets = np.arange(len(stats["mu_g"]))[:, None] * np.ones((1, 4))
    for key, mean in (("g", "mu_g"), ("y", "mu_y")):
        shift = offsets if key == "g" else -3 * offsets
        old = moved[mean].copy()
        moved["second_" + key] += (
            np.einsum("ni,nj->nij", old, shift)
            + np.einsum("ni,nj->nij", shift, old)
            + np.einsum("ni,nj->nij", shift, shift)
        )
        moved[mean] += shift
    np.testing.assert_allclose(M.fit_reference(moved)["link"], np.eye(4), atol=1e-10)


def test_shift_null_has_own_source_link_and_normal_equations():
    _, stats = M.synthetic_world("shared_linear", 15)
    ref = M.fit_reference(stats)
    vg, _ = M.within_covariances(stats)
    np.testing.assert_allclose(
        vg.mean(0) @ ref["link"], stats["cross"].mean(0), atol=1e-13
    )
    np.testing.assert_allclose(
        vg.mean(0) @ ref["shift_link"], stats["shifted"].mean(0), atol=1e-13
    )
    assert np.linalg.norm(ref["link"] - ref["shift_link"]) > 0.1
    np.testing.assert_allclose(
        M.components(stats, ref)["shifted_residual_link"].mean(0), 0, atol=1e-13
    )


def test_signed_energy_identity_retains_cross_term():
    _, held, ref, _ = world_profiles("shared_nonlinear")
    energy = M.centered_energy(held, ref)
    np.testing.assert_allclose(energy[:, 0], energy[:, 1:].sum(1), atol=1e-12)
    assert np.max(abs(energy[:, 3])) > 0.1


def test_general_invertible_transport_preserves_link_and_distances():
    material, stats = M.synthetic_world("personal_rotation", 24)
    ref = M.fit_reference(stats)
    rng = np.random.default_rng(16)
    a, b = (
        rng.normal(size=(4, 4)) + 4 * np.eye(4),
        rng.normal(size=(4, 4)) + 4 * np.eye(4),
    )
    moved = []
    for author, panel in itertools.product(range(40), range(3)):
        key = f"a{author:03d}_p{panel}"
        moved.append(
            C.panel_statistics(material[key + "__g"] @ a, material[key + "__y"] @ b)
        )
    transported = C.stack_statistics(moved)
    new_ref = M.fit_reference(transported)
    np.testing.assert_allclose(
        new_ref["link"], np.linalg.solve(a, ref["link"]) @ b, atol=1e-12
    )
    p, q = M.profiles(stats, ref), M.profiles(transported, new_ref)
    for arm in M.ARMS:
        np.testing.assert_allclose(
            C.squared_distances(p[arm][:6], p[arm][6:12]),
            C.squared_distances(q[arm][:6], q[arm][6:12]),
            atol=1e-10,
        )


def test_held_changes_do_not_affect_source_link():
    stats, held, ref, _ = world_profiles("personal_rotation")
    before = {name: values.copy() for name, values in ref.items()}
    held["cross"] *= 100
    M.profiles(held, ref)
    for name, values in before.items():
        np.testing.assert_array_equal(values, ref[name])
    np.testing.assert_array_equal(
        M.fit_reference(select(stats, slice(0, 60)))["link"], ref["link"]
    )


def test_singular_within_shape_fails_even_if_total_covariance_full_rank():
    material = [
        C.panel_statistics(
            np.tile([i, i**2, i**3, i**4], (8, 1)),
            np.tile([i, i**2, i**3, i**4], (8, 1)),
        )
        for i in range(8)
    ]
    with pytest.raises(ValueError, match="Degenerate"):
        M.fit_reference(C.stack_statistics(material))
