#!/usr/bin/env python3
"""Finite sampling laws must not masquerade as personality or semantic reasoning."""

import numpy as np
import pytest

from grain_research.construction import math_evidence_sampling as S


def example():
    """Two designed inventories with the same exposure and different target codes."""
    n = np.array([[3, 1, 0, 0, 0, 0, 0, 0], [1, 3, 0, 0, 0, 0, 0, 0]])
    return n, np.tile([[1], [0]], (1, 5))


def events(cells):
    """Give distinct identities even when observation categories repeat."""
    return [{"event_id": str(i), "cell": a} for i, a in enumerate(cells)]


def test_singleton_same_but_repeated_category_changes_reading():
    n, y = example()
    p = [S.replay(n, y, events([0]), mode)[1]["mu"] for mode in S.REGIMES]
    np.testing.assert_allclose(p[0], p[1])
    repeated = [S.replay(n, y, events([0, 0]), mode)[1]["mu"] for mode in S.REGIMES]
    np.testing.assert_allclose(repeated[0], 0.9)
    np.testing.assert_allclose(repeated[1], 1)


def test_distinct_categories_have_identical_posteriors():
    n, y = example()
    p = [S.replay(n, y, events([0, 1]), mode)[1]["mu"] for mode in S.REGIMES]
    np.testing.assert_allclose(p[0], p[1], atol=1e-14)


@pytest.mark.parametrize("regime", S.REGIMES)
def test_recursive_reading_matches_direct_count_law(regime):
    n, y = example()
    for word in ([0], [0, 1], [0, 0], [1, 0, 1]):
        r, end = S.replay(n, y, events(word), regime)
        np.testing.assert_allclose(end["mu"], S.direct_posterior(n, y, word, regime))
        for step in r:
            np.testing.assert_allclose(step["predicted"], step["after"], atol=1e-14)


def test_finite_continuation_is_consumed_not_static():
    n, y = example()
    _, end = S.replay(n, y, events([0, 0]), "finite_inventory")
    np.testing.assert_allclose(end["pi"][:2], [0.5, 0.5])


def test_exhausted_inventory_is_unsupported_not_zero_target():
    n, y = example()
    _, end = S.replay(n, y, events([0, 0, 0, 0]), "finite_inventory")
    assert end["supported"] is False and end["mu"] is None


def test_duplicate_identity_does_not_consume_another_record():
    n, y = example()
    r, end = S.replay(n, y, events([0]) * 2, "finite_inventory")
    assert len(r) == 1
    np.testing.assert_allclose(end["mu"], 0.75)


def test_same_identity_different_cell_rejected():
    n, y = example()
    with pytest.raises(ValueError, match="identity"):
        S.replay(n, y, [{"event_id": "a", "cell": a} for a in (0, 1)], S.REGIMES[0])


def test_bad_inventory_or_unknown_regime_rejected():
    n, y = example()
    unequal = n.copy()
    unequal[0, 0] += 1
    with pytest.raises(ValueError):
        S.validate(unequal, y)
    with pytest.raises(ValueError):
        S.replay(n, y, events([0]), "unknown")


def test_exact_reconstruction_not_a_fit():
    n, _ = example()
    np.testing.assert_array_equal(S.empirical_counts(n / 4, 4), n)
    with pytest.raises(ValueError):
        S.empirical_counts(np.full((2, 8), 1 / 8), 4)
