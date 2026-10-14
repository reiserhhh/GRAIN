#!/usr/bin/env python3
"""Synthetic aligned SOURCE tests, independent of raw/private author material."""

import numpy as np
import pytest

from grain_research.construction import math_criterion_components as C
from grain_research.construction import math_criterion_reader as M
from grain_research.construction import math_source_anchor_exchange as E
from grain_research.construction.math_relational_diagnostic_update import BITS


def fixture():
    """Two full-profile banks and fixed label-blind event weights."""
    y = np.tile(BITS, (2, 1))
    b = np.repeat((0, 1), 32)
    m, e, g = C.criterion_design(y, b)
    rng = np.random.default_rng(279)
    w = rng.normal(size=(6, 2, 64))
    w -= w.mean(-1, keepdims=True)
    r = rng.normal(size=(6, 4, 5))
    return y, b, m, e, g, w, r


def test_components_and_observer_pullback():
    """Direct source contraction and observer-first contraction must agree."""
    y, _, m, e, _, w, r = fixture()
    scores = M.pair_components(w, y, m)
    np.testing.assert_allclose(scores[:, :, 0], scores[:, :, 1:].sum(2), atol=1e-14)
    contacts = M.contacts(scores, r)
    alpha = M.observer_pullback(w, r)
    np.testing.assert_allclose(contacts[2, 0], e.T @ alpha, atol=1e-14)
    np.testing.assert_allclose(contacts[:, 3], contacts[:, :3].sum(1), atol=1e-14)


def test_conditional_reference_and_exact_variance():
    """Validate retained strata and an alternative finite block variance."""
    y, b, m, e, g, w, r = fixture()
    assignments = np.empty((5, 19, 64), int)
    for q in range(5):
        for bank in (0, 1):
            ix = np.flatnonzero(b == bank)
            assignments[q, :, ix] = ix[E.assignment_indices(y[ix], q, 19, 279)].T
    alpha = M.observer_pullback(w, r)[None]
    observed, null, variance, nominal = M.conditional_reference(
        alpha, y, m, g, assignments
    )
    for q in range(5):
        np.testing.assert_allclose(observed[0, q], e[:, q] @ alpha[0], atol=1e-14)
        np.testing.assert_allclose(
            null[:, 0, q], e[assignments[q], q] @ alpha[0], atol=1e-14
        )
        direct = np.zeros(5)
        for group in np.unique(g[:, q]):
            rows = alpha[0, g[:, q] == group]
            n, p = len(rows), y[g[:, q] == group, q].mean()
            direct += p * (1 - p) * n / (n - 1) * ((rows - rows.mean(0)) ** 2).sum(0)
        np.testing.assert_allclose(variance[0, q], direct, atol=1e-14)
    assert np.all((nominal > 0) & (nominal <= 1))


def test_source_assignment_cannot_cross_configuration():
    """A full permutation is insufficient if it changes the held other-four bits."""
    y, _, m, _, g, w, r = fixture()
    ix = np.tile(np.arange(64), (5, 1, 1))
    ix[0, 0, [0, 2]] = ix[0, 0, [2, 0]]
    with pytest.raises(ValueError, match="stratum"):
        M.conditional_reference(M.observer_pullback(w, r)[None], y, m, g, ix)


def test_nonzero_weight_mass_is_rejected():
    """A score-level offset must not be silently accepted as F-J evidence."""
    y, _, m, _, _, w, _ = fixture()
    w[0, 0, 0] += 1
    with pytest.raises(ValueError, match="signed weights"):
        M.pair_components(w, y, m)


def test_cold_pair_rejects_evaluated_labels():
    """The cold record contains vector evidence only, with all source parts kept."""
    rng = np.random.default_rng(279)
    bundle = {
        "prototypes": rng.normal(size=(64, 2, 2, 8)),
        "codes": np.tile(np.arange(32), 2),
        "folds": np.repeat((0, 1), 32),
        "relink": np.tile((0, 1), (64, 1)),
        "variance": 1.0,
        "projection": rng.normal(size=(8, 3)),
    }
    record = {
        "frame_key": "synthetic-test",
        "sampling_regime": "next_cached_record_within_registered_window",
        "events": [
            {
                "event_id": str(k),
                "vector": rng.normal(size=8).tolist(),
                "time": float(k),
                "window": 0,
                "condition": 0,
            }
            for k in (1, 2)
        ],
    }
    result = M.cold_pair(bundle, record, "GRAIN15")
    for values in result["source_readings"].values():
        x = np.asarray(values)
        assert x.shape == (2, 3, 5)
        np.testing.assert_allclose(x[:, 0], x[:, 1:].sum(1), atol=1e-14)
    record["target_label"] = 1
    with pytest.raises(ValueError, match="qualified"):
        M.cold_pair(bundle, record, "GRAIN15")
