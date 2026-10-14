#!/usr/bin/env python3
"""Synthetic carrier scope and two-stage interaction checks; no model loading."""

import json

import numpy as np
import pytest

from grain_research.construction import math_target_carrier_closure as M


def test_scopes_preserve_causal_order_and_tail():
    row = {
        "history": list(range(12)),
        "scopes": {"extraversion": [2, 3, 4], "neuroticism": [5, 6, 7, 8]},
    }
    scopes = M.carrier_scopes(row)
    assert scopes["tail"] == [9, 10, 11]
    assert scopes["e_closure"] == list(range(2, 12))
    assert scopes["n_other"] == [2, 3, 4]
    assert max(scopes["n_other"]) < min(scopes["n_local"])


def test_patch_does_not_escape_time_rows():
    base = np.arange(48).reshape(2, 2, 6, 2).astype(float)
    donor = -base
    result = M.replace_slots(base, donor, [1, 5])
    np.testing.assert_array_equal(result[:, :, [1, 5]], donor[:, :, [1, 5]])
    np.testing.assert_array_equal(result[:, :, [0, 2, 3, 4]], base[:, :, [0, 2, 3, 4]])
    np.testing.assert_array_equal(base, np.arange(48).reshape(base.shape))


def test_mixed_character_is_exact_double_flip_defect():
    cube = np.array([[2.0, 8.0], [11.0, 25.0]])
    c = M.memory_characters(cube)
    for ie, inn in np.ndindex(2, 2):
        e, n = 2 * ie - 1, 2 * inn - 1
        reconstructed = c[0] + e * c[1] + n * c[2] + e * n * c[3]
        assert reconstructed == cube[ie, inn]
        defect = cube[1 - ie, 1 - inn] - M.additive_memory(
            cube[ie, inn], cube[1 - ie, inn], cube[ie, 1 - inn]
        )
        assert defect == 4 * e * n * c[3]


def test_encoding_interaction_with_linear_reader():
    account = M.interaction_account(1, 3, 4, 10, 6)
    assert account["total"] == account["encoding"] == 4
    assert account["reader"] == 0


def test_reader_interaction_with_additive_encoding():
    # phi(m)=m**2, m0=0, me=1, mn=2, mboth=madd=3.
    account = M.interaction_account(0, 1, 4, 9, 9)
    assert account["encoding"] == 0
    assert account["total"] == account["reader"] == 4


def test_both_interactions_can_cancel():
    account = M.interaction_account(0, 1, 2, 3, 8)
    assert account["total"] == 0
    assert account["encoding"] == -5
    assert account["reader"] == 5


def test_json_safe_numpy_counts_and_no_nan():
    value = M.json_native({"n": np.int64(3), "r": np.float64(0.2), "null": None})
    assert json.loads(json.dumps(value, allow_nan=False))["n"] == 3
    with pytest.raises(ValueError):
        M.json_native(np.nan)


def test_invalid_memories_and_scope_fail_closed():
    with pytest.raises(ValueError):
        M.memory_characters(np.zeros((2, 3)))
    with pytest.raises(ValueError):
        M.replace_slots(np.zeros((1, 1, 4, 1)), np.zeros((1, 1, 4, 1)), [4])
    with pytest.raises(ValueError):
        M.interaction_account(np.zeros(2), np.zeros(3), 0, 0, 0)
    with pytest.raises(ValueError):
        M.carrier_scopes(
            {"history": [0, 1, 2], "scopes": {"extraversion": [0], "neuroticism": [2]}}
        )


def test_rounded_additive_memory_keeps_explicit_cast():
    b = np.array([1.234567], dtype=np.float32)
    e, n = b + np.float32(0.17), b - np.float32(0.07)
    a = M.additive_memory(b, e, n)
    assert a.dtype == np.float64
    np.testing.assert_array_equal(
        a, e.astype(float) + n.astype(float) - b.astype(float)
    )
