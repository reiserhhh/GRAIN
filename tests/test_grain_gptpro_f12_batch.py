#!/usr/bin/env python3
"""F12-03 batch invariance on the F12 transport_score route: a new batch of
authors never changes an existing author's score or support."""

import numpy as np
import pytest

from grain_research.gptpro_foundation_v3.f12_transport import (
    fit_transport,
    transport_score,
)

Z = np.tile([-1.0, -1 / 3, 1 / 3, 1.0], 8)


@pytest.mark.parametrize("mode", ["global", "conditional"])
def test_scores_do_not_depend_on_other_authors(mode):
    rng = np.random.default_rng(3)
    old = rng.normal(size=60)
    z = np.resize(Z, 60)
    new = z + (1 + 0.25 * z) * old if mode == "conditional" else 1 + 1.5 * old
    model = fit_transport(old, new, z, mode)
    batch1 = rng.normal(size=(50, 32))
    batch2 = rng.normal(size=(70, 32)) * 3 + 2
    s1, sup1 = transport_score(model, batch1, Z)
    s_all, sup_all = transport_score(model, np.vstack([batch1, batch2]), Z)
    np.testing.assert_array_equal(sup1, sup_all[:50])
    np.testing.assert_array_equal(s1[sup1], s_all[:50][sup1])


def test_cubic_support_is_fixed_by_pilot_range_not_new_batch():
    rng = np.random.default_rng(4)
    old = rng.uniform(-1, 1, 60)
    new = old + 0.1 * old**3
    model = fit_transport(old, new, np.resize(Z, 60), "cubic")
    batch1 = rng.uniform(-1, 1, (20, 32))
    wide = rng.uniform(-5, 5, (20, 32))
    s1, sup1 = transport_score(model, batch1, Z)
    s_all, sup_all = transport_score(model, np.vstack([batch1, wide]), Z)
    np.testing.assert_array_equal(sup1, sup_all[:20])
    np.testing.assert_array_equal(np.isnan(s1), np.isnan(s_all[:20]))
    assert not sup_all[20:].all()


def test_unseen_condition_is_unsupported_not_extrapolated():
    rng = np.random.default_rng(5)
    old = rng.normal(size=60)
    z = np.resize([-1.0, 1.0], 60)
    model = fit_transport(old, z + old, z, "conditional")
    score, support = transport_score(model, np.zeros((1, 2)), np.array([-1.0, 0.0]))
    assert support.tolist() == [[True, False]] and np.isnan(score[0, 1])
