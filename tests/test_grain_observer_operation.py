#!/usr/bin/env python3
"""Operation meaning, keyed observations and a cold fixed-output covector."""

import numpy as np
import pytest

from grain_research.construction import math_observer_operation as M


def bank():
    """Synthetic complete keys, not human personality labels."""
    return [
        {"item": j, "target": j // 10, "pole": 1 if j % 10 < 6 else -1}
        for j in range(50)
    ]


def test_question_keeps_statement_and_declares_object():
    for operation in M.OPERATIONS:
        text = M.question("Worry about things.", operation)
        assert '"I worry about things."' in text
    assert "across occasions" in M.question("Worry about things.", M.OPERATIONS[0])
    assert "within this writing sample" in M.question(
        "Worry about things.", M.OPERATIONS[1]
    )
    with pytest.raises(ValueError):
        M.question("Worry", M.OPERATIONS[0])


def test_cold_covector_and_binary_gauge():
    rng = np.random.default_rng(280)
    h, w = rng.normal(size=(3, 50, 7)), rng.normal(size=(2, 7))
    z = M.head_logits(h, w, np.array([0.1, -0.2]))
    odds = np.einsum("...d,d->...", h, w[1] - w[0]) - 0.3
    np.testing.assert_allclose(z[..., 1] - z[..., 0], odds, atol=1e-14)
    for scale, keys in M.key_values(z, bank()).items():
        shifted = M.key_values(z + 17, bank())[scale]
        for key in keys:
            np.testing.assert_allclose(keys[key], shifted[key], atol=1e-14)


def test_offset_null_and_probability_curvature():
    rng = np.random.default_rng(281)
    z = rng.normal(size=(48, 2, 4, 50, 2))
    z[:, 1] = z[:, 0] + np.array([0, 1.7])
    result = M.operation_accounts(z, bank())
    joint = result["log_odds"]["balanced_key"]["parts"]["operation_difference"]["join"]
    np.testing.assert_allclose(joint, 0, atol=1e-14)
    curved = result["probability"]["balanced_key"]["parts"]["operation_difference"][
        "join"
    ]
    assert np.linalg.norm(curved) > 0.01


def test_context_dependent_operation_and_shape_guards():
    z = np.zeros((48, 2, 4, 50, 2))
    z[:, 0, :2, :6, 1] = 2
    result = M.operation_accounts(z, bank())
    assert np.all(
        result["log_odds"]["balanced_key"]["parts"]["operation_difference"]["join"][
            :, 0
        ]
        == 1
    )
    with pytest.raises(ValueError):
        M.operation_accounts(z[:-1], bank())
    with pytest.raises(ValueError):
        M.head_logits(np.zeros((2, 7)), np.zeros((2, 6)), np.zeros(2))


def test_unequal_poles_official_key():
    logits = np.zeros((50, 2))
    logits[:6, 1] = np.log(4)
    logits[6:10, 1] = -np.log(4)
    keys = M.key_values(logits, bank())["probability"]
    np.testing.assert_allclose(keys["balanced_key"][0], 0.8)
    np.testing.assert_allclose(keys["equal_item_key"][0], 0.8)
    np.testing.assert_allclose(keys["equal_item_key"][1:], 0.5)
