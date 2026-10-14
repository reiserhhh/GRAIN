#!/usr/bin/env python3
"""Test criterion discrimination separately from action-of-average effects."""

import numpy as np

from scripts.describe_grain_target_crossed_evidence_hm232 import accounts


def test_averaging_gap_without_source_discrimination():
    """Equal individual readings can differ from the mean-carrier reading."""
    t = np.ones((2, 5, 5))
    a = accounts(t, np.zeros((2, 5)))
    assert np.count_nonzero(a["source_discrimination"]) == 0
    assert np.all(a["averaging_action_gap"] == 1)
    assert a["energy"]["departure"] == 1


def test_source_tilt_is_not_reader_interaction():
    """A source effect can act identically under every recipient criterion."""
    t = np.broadcast_to(np.arange(5), (2, 5, 5)).copy()
    a = accounts(t, np.full((2, 5), 2))
    assert np.count_nonzero(a["interaction"]) == 0
    assert np.count_nonzero(a["averaging_action_gap"]) == 0
    assert a["energy"]["source_discrimination"] == 2


def test_complete_signed_orthogonal_accounts():
    """Preserve arbitrary signed effects without a diagonal-advantage assumption."""
    rng = np.random.default_rng(232)
    t, c = rng.normal(size=(3, 5, 5)), rng.normal(size=(3, 5))
    a = accounts(t, c)
    assert a["reconstruction_error"] < 1e-14
    assert a["energy_account_error"] < 1e-14
    assert np.max(abs(a["source_discrimination"].sum(2))) < 1e-14
    assert np.max(abs(a["interaction"].sum(1))) < 1e-14
    assert np.max(abs(a["interaction"].sum(2))) < 1e-14
