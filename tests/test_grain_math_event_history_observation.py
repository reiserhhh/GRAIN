#!/usr/bin/env python3
"""Synthetic checks of complete event supports and dependence-only author laws."""

import numpy as np
import pytest

from grain_research.construction import math_event_history_observation as M


def test_full_basis_has_cross_event_coordinate():
    x, basis, names, mixed = M.design_basis()
    np.testing.assert_allclose(basis @ basis.T, 64 * np.eye(64))
    assert mixed[names.index("yha")]
    f = x[:, 4] * x[:, 1] * x[:, 3]
    coef = M.coefficients(f)
    assert coef[names.index("yha")] == 1
    np.testing.assert_allclose(basis.T @ coef, f)


def test_additive_event_functions_and_interaction():
    x = M.design_basis()[0]
    additive = x[:, 0] * x[:, 2] + x[:, 1] * x[:, 3] + x[:, 4]
    assert M.composition_account(additive)["cross_event_energy"] == 0
    value = additive + 2 * x[:, 4] * x[:, 1] * x[:, 3]
    account = M.composition_account(value)
    assert account["cross_event_energy"] == 4
    assert account["parseval_error"] == 0


def test_author_laws_share_complete_event_marginals():
    plus, minus = M.author_laws()
    p, m = plus.reshape((2,) * 6), minus.reshape((2,) * 6)
    np.testing.assert_allclose(p.sum((1, 3, 5)), m.sum((1, 3, 5)))
    np.testing.assert_allclose(p.sum((0, 2, 4)), m.sum((0, 2, 4)))
    np.testing.assert_allclose(p.sum(4), m.sum(4))
    x = M.design_basis()[0]
    coupling = x[:, 4] * x[:, 1] * x[:, 3]
    assert np.sum((plus - minus) * coupling) == 1
    with pytest.raises(ValueError, match="epsilon"):
        M.author_laws(0)


def test_reader_mean_is_not_discarded_as_noise():
    a = np.arange(36).reshape(6, 6)
    account = M.reader_account(a)
    assert account["battery_common_energy"] > account["reader_specific_energy"] > 0
    assert account["identity_error"] == 0
    with pytest.raises(ValueError, match="six"):
        M.reader_account(a[:, :5])
