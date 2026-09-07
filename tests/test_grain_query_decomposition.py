#!/usr/bin/env python3
"""Statistical identities and author-out construction for shared query experiments."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.query_decomposition import (
    centered_values,
    directional_cumulants,
    fold_query_references,
    mix_raw_query,
    pairing_query_expansion,
)
from grain_research.vector_mechanisms.query_tilt import (
    pairing_component,
    query_jacobian,
    tilted_readout,
)


def test_pairing_centered_expectation_and_full_mass_derivative():
    rng = np.random.default_rng(734)
    k, v, q = rng.normal(size=(14, 4)), rng.normal(size=(14, 3)), rng.normal(size=4)
    groups = [[1, 2, 3], [6, 7, 8, 9, 10]]
    centered = centered_values(v, groups)
    output, p = tilted_readout(k, centered, q)
    np.testing.assert_allclose(output, pairing_component(p, v, groups), atol=1e-14)
    delta, eps = rng.normal(size=4), 1e-6
    derivative = (
        tilted_readout(k, centered, q + eps * delta)[0]
        - tilted_readout(k, centered, q - eps * delta)[0]
    ) / (2 * eps)
    np.testing.assert_allclose(
        derivative, query_jacobian(k, centered, q) @ delta, rtol=1e-7, atol=1e-9
    )
    np.testing.assert_allclose(centered[[0, 4, 5, 11, 12, 13]], 0)
    with pytest.raises(ValueError):
        centered_values(v, [[1], [1]])


def test_reference_queries_exclude_entire_author_fold():
    q = np.arange(6 * 3 * 2 * 4, dtype=float).reshape(6, 3, 2, 4)
    folds = np.array([0, 1, 2, 0, 1, 2])
    before = fold_query_references(q, folds)
    q[folds == 1] += 10000
    after = fold_query_references(q, folds)
    np.testing.assert_array_equal(before[1], after[1])
    assert not np.allclose(before[0], after[0])
    np.testing.assert_array_equal(mix_raw_query(before[0], after[0], 1), after[0])
    with pytest.raises(ValueError):
        mix_raw_query(before[0], after[0], 2)


def test_query_cumulants_match_finite_differences_and_tilt():
    rng = np.random.default_rng(835)
    k, v = rng.normal(size=(20, 4)), rng.normal(size=(20, 3))
    q, delta, beta = rng.normal(size=4), rng.normal(size=4), rng.normal(size=3)
    v = centered_values(v, [[1, 2, 3, 4], [7, 8, 9, 10, 11]])
    terms = directional_cumulants(k, v, q, q + delta, beta)

    def score(alpha):
        return tilted_readout(k, v, q + alpha * delta)[0] @ beta

    eps = 1e-4
    np.testing.assert_allclose(
        terms["first"], (score(eps) - score(-eps)) / (2 * eps), rtol=1e-6
    )
    np.testing.assert_allclose(
        terms["second"],
        (score(eps) - 2 * score(0) + score(-eps)) / eps**2,
        rtol=1e-5,
        atol=1e-7,
    )
    np.testing.assert_allclose(terms["exact_increment"], score(1) - score(0))
    _, p = tilted_readout(k, v, q)
    u = k @ delta
    weights = p * np.exp(u - u.max())
    np.testing.assert_allclose(weights @ (v @ beta) / weights.sum(), score(1))
    assert terms["directional_fisher"] >= 0


def test_multihead_expansion_matches_pairing_readout():
    from grain_research.vector_mechanisms.prefix import prefix_pairing

    rng = np.random.default_rng(936)
    cache = {
        "q": rng.normal(size=(4, 6)),
        "k": rng.normal(size=(19, 2, 6)),
        "v": rng.normal(size=(19, 2, 6)),
    }
    ref, beta = rng.normal(size=(4, 6)), rng.normal(size=24)
    groups = [[1, 2, 3], [8, 9, 10]]
    terms = pairing_query_expansion(cache, groups, ref, beta)
    native = prefix_pairing(cache, groups) @ beta
    shared = prefix_pairing(cache, groups, ref) @ beta
    np.testing.assert_allclose(terms["shared"], shared, atol=1e-14)
    np.testing.assert_allclose(terms["exact_increment"], native - shared, atol=1e-14)
