#!/usr/bin/env python3
"""Report-edit consistency tests independent of neural or private inputs."""

import numpy as np
import pytest

from grain_research.construction import math_evidence_update_integrability as M


def fixture():
    """Supply disjoint clauses and a mixed component, not human trait truth."""
    row = {
        "history": list(range(6)),
        "scopes": {"extraversion": [1, 2], "neuroticism": [3, 4]},
    }
    coefficients = np.arange(4 * 6, dtype=float).reshape(4, 1, 1, 6, 1) / 10
    return row, coefficients


@pytest.mark.parametrize("policy", M.POLICIES)
def test_involution_does_not_establish_commutation(policy):
    row, c = fixture()
    for word in ("", "ee", "nn", "eenn"):
        result, state = M.walk(c, row, (1, -1), word, policy)
        assert state == (1, -1)
        np.testing.assert_allclose(result, 0, atol=1e-14)


@pytest.mark.parametrize("policy", M.POLICIES)
def test_signed_loop_and_two_routes(policy):
    row, c = fixture()
    loop, end = M.walk(c, row, (-1, 1), "enen", policy)
    reverse, _ = M.walk(c, row, (-1, 1), "nene", policy)
    en, s1 = M.walk(c, row, (-1, 1), "en", policy)
    ne, s2 = M.walk(c, row, (-1, 1), "ne", policy)
    assert end == (-1, 1) and s1 == s2 == (1, -1)
    np.testing.assert_allclose(
        loop, M.predicted_loop(c, row, (-1, 1), policy), atol=1e-14
    )
    np.testing.assert_allclose(loop, en - ne, atol=1e-14)
    np.testing.assert_allclose(loop, -reverse, atol=1e-14)


def test_adding_marginals_does_not_repair_loop():
    row, c = fixture()
    a, _ = M.walk(c, row, (1, 1), "enen", "relation_other")
    b, _ = M.walk(c, row, (1, 1), "enen", "full_other")
    assert np.linalg.norm(a) > 0
    np.testing.assert_allclose(a, b, atol=1e-14)


@pytest.mark.parametrize("policy", ("shared_other", "shared_closure"))
def test_shared_potential_preserves_all_sectors_and_paths(policy):
    row, c = fixture()
    for state in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        for word in ("e", "n", "en", "ne", "enen", "eenn"):
            result, end = M.walk(c, row, state, word, policy)
            expected = M.potential(c, row, end, policy) - M.potential(
                c, row, state, policy
            )
            np.testing.assert_allclose(result, expected, atol=1e-14)
    assert not np.array_equal(
        M.potential(c, row, (1, 1), policy), M.potential(c, row, (-1, 1), policy)
    )


def test_causal_support_compatibility_not_equal_masks():
    row, c = fixture()
    c[3, :, :, :3] = 0
    closure, _ = M.walk(c, row, (1, 1), "enen", "full_closure")
    assert M.supports(row, "full_closure")[0] != M.supports(row, "full_closure")[1]
    np.testing.assert_allclose(closure, 0, atol=1e-14)
    local, _ = M.walk(c, row, (1, 1), "enen", "full_other")
    assert np.linalg.norm(local) > 0


def test_coherence_not_faithful_reconstruction():
    row, c = fixture()
    cube = np.stack(
        [c[0] + e * c[1] + n * c[2] + e * n * c[3] for e in (-1, 1) for n in (-1, 1)]
    ).reshape(2, 2, 1, 1, 6, 1)
    np.testing.assert_allclose(M.compile_cube(cube), c, atol=1e-14)
    edited = M.potential(c, row, (1, 1), "shared_other")
    assert not np.allclose(edited, cube[1, 1])


def test_unknown_axis_and_policy_rejected():
    row, c = fixture()
    with pytest.raises(ValueError):
        M.walk(c, row, (1, 1), "x", "full_other")
    with pytest.raises(ValueError):
        M.increment(c, row, (1, 1), "e", "fitted")
