#!/usr/bin/env python3
"""Synthetic tests for protocol-aware observation, not predictive success."""

import numpy as np
import pytest

from grain_research.construction import math_acquisition_law as M
from scripts.run_grain_acquisition_law_hm197 import observation_cells


def sample():
    """Supply explicit author cells/selection with all eight-cell support."""
    c = np.tile(np.arange(12) % 8, (16, 2, 1))
    s = np.zeros_like(c, bool)
    s[:8, :, :4] = True
    s[8:, :, 4:8] = True
    y = np.repeat([0, 1], 8)[:, None]
    return c, s, y


def test_mixture_and_equal_author_prior():
    c, s, y = sample()
    law = M.compile_fields(c, s, y)
    assert np.allclose(law["fields"][0], law["mixture"])
    assert np.allclose(law["joint"].sum(1), law["prior"])
    assert np.allclose(law["reference"].sum(1), 1)
    assert np.allclose(law["reference"][0] @ law["selection_probability"], 1 / 3)


def test_selection_adds_information_inside_same_cell():
    c, s, y = sample()
    law = M.compile_fields(c, s, y)
    assert np.allclose(law["fields"][0], 0.5)
    assert np.max(np.abs(law["fields"][1] - law["fields"][2])) > 0.5


def test_screened_selection_does_not_change_target_mean():
    c, s, _ = sample()
    y = np.tile([0, 1], 8)[:, None]
    f = M.compile_fields(c, s, y)["fields"]
    assert np.allclose(f, 0.5)


def test_missing_support_not_zero_or_smoothed():
    c, s, y = sample()
    c[:] = 0
    law = M.compile_fields(c, s, y)
    assert not law["support"][:, 1:].any()
    assert np.isnan(law["fields"][:, 1:]).all()
    assert np.isnan(law["mixture"][1:]).all()


def test_accounts_keep_protocol_and_evidence_separate():
    c = np.arange(12) % 8
    f = np.broadcast_to(np.arange(8)[None, :, None] / 8, (3, 8, 1)).copy()
    f[1] += 0.1
    f[2] -= 0.2
    rows = M.evidence_accounts(c, np.arange(4), f, np.array([0.5]))
    for r in rows:
        assert np.allclose(r["total"], r["fixed_action"] + r["protocol_drift"])
    assert np.allclose(rows[0]["protocol_drift"], 0.1)
    assert np.allclose(rows[1]["protocol_drift"], -0.2)


def test_stable_ties_and_sign_symmetry():
    v = np.tile([1.0, 1], (12, 1))
    poles = np.eye(2)
    ids, signed = M.select_top(v, poles)
    assert np.array_equal(ids, np.arange(4))
    assert np.array_equal(M.select_top(v, poles[::-1])[0], ids)
    assert np.array_equal(M.select_top(v, poles[::-1])[1], -signed)


def test_selection_uses_within_family_comparison():
    v = np.column_stack([np.arange(1, 13), np.ones(12)])
    ids, signed = M.select_top(v, np.array([[0, 1], [1, 0]]))
    assert np.array_equal(ids, np.sort(np.argsort(-np.abs(signed), kind="stable")[:4]))
    assert M.selection_mask(ids).sum() == 4


def test_zero_vectors_and_duplicate_events_rejected():
    with pytest.raises(ValueError, match="Zero"):
        M.select_top(np.zeros((12, 2)), np.eye(2))
    with pytest.raises(ValueError, match="Distinct"):
        M.selection_mask([0, 0, 2, 3])


def test_ragged_author_selection_rejected():
    c, s, y = sample()
    s[0, 0, 0] = False
    with pytest.raises(ValueError, match="fixed-size"):
        M.compile_fields(c, s, y)


def test_cross_transport_can_reverse_without_clipping():
    f = np.zeros((2, 2, 3, 8, 1))
    f[0, :, 1] = 1
    f[1, :, 1] = -1
    result = M.cross_transport(f, np.ones(8) / 8)
    assert result["cross_product"].item() == -1
    assert result["coherence"].item() == -1


def test_inherited_cell_encoder_batch_axes_do_not_change_events():
    x = np.ones((2, 3, 12, 15))
    c = np.ones((2, 3, 12), int)
    assert np.array_equal(observation_cells(x, c, [0.5, 0.5]), np.full(c.shape, 7))
