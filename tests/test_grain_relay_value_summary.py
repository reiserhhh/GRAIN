#!/usr/bin/env python3
"""Finite summary and actual tiny-network reader witnesses."""

import numpy as np
import pytest

from grain_research.construction import math_relay_value_summary as M
from tests.test_grain_query_relay_cut import tiny_observer


def unit():
    """Use unequal-length questions followed by exactly the same surface suffix."""
    return {
        "body": [2, 3, 4],
        "input_ids": [
            [1, 2, 3, 4, 5] + [10 + q] * (q + 1) + [30, 31, 32, 33] for q in range(5)
        ],
    }


def test_suffix_and_projection():
    u = unit()
    rows = M.common_suffix(u)
    assert [u["input_ids"][q][r[0] : r[-1] + 1] for q, r in enumerate(rows)] == [
        [30, 31, 32]
    ] * 5
    v = np.random.default_rng(202).normal(size=(3, 5, 1, 3, 8))
    common = M.summary_values(v, "common")
    np.testing.assert_allclose((v - common).sum(1), 0, atol=2e-15)
    np.testing.assert_allclose(
        (v * v).sum(), (common * common).sum() + ((v - common) ** 2).sum(), atol=2e-12
    )
    np.testing.assert_array_equal(
        M.summary_values(v, "foreign", [4, 2, 1, 3, 0])[:, 3], v[:, 3]
    )
    with pytest.raises(ValueError):
        M.common_suffix(
            {"body": [1], "input_ids": [[1, 2, q + 10, 3] for q in range(5)]}
        )


def test_frozen_contraction():
    rng = np.random.default_rng(202)
    p = rng.uniform(size=(5, 3, 12))
    p /= p.sum(-1, keepdims=True)
    v = rng.normal(size=(5, 3, 12, 8))
    suffix = [list(range(5 + q, 8 + q)) for q in range(5)]
    d = rng.normal(size=(5, 3, 3, 8))
    before = p.copy()
    actual = M.fixed_value_output(p, v, suffix, d)
    expected = np.array(
        [
            [
                sum(
                    p[q, h, i]
                    * (d[q, h, suffix[q].index(i)] if i in suffix[q] else v[q, h, i])
                    for i in range(12)
                )
                for h in range(3)
            ]
            for q in range(5)
        ]
    )
    np.testing.assert_allclose(actual, expected, atol=1e-15)
    np.testing.assert_array_equal(p, before)


def test_actual_fixed_reader_self_foreign_and_prefix():
    """All probabilities stay fixed while value substitution changes the endpoint."""
    import transformers.models.qwen2.modeling_qwen2 as qwen

    observer, u = tiny_observer(), unit()
    original = qwen.eager_attention_forward
    native = M.native_batch(observer, u)
    bank = native[5]
    before = {k: v.copy() for k, v in bank.items()}
    identity = M.native_batch(observer, u, bank, "self")
    common = M.native_batch(observer, u, bank, "common")
    foreign = M.native_batch(observer, u, bank, "foreign", [4, 2, 1, 3, 0])
    np.testing.assert_allclose(identity[0], native[0], atol=3e-7, rtol=0)
    np.testing.assert_allclose(foreign[0][3], native[0][3], atol=3e-7, rtol=0)
    assert np.max(abs(common[0] - native[0])) > 1e-8
    assert np.max(abs(foreign[0][1] - native[0][1])) > 1e-8
    for result in (identity, common, foreign):
        assert max(r["fixed_probability_error"] for r in result[6]) == 0
        assert max(r["prefix_V_error"] for r in result[6]) == 0
    for k in bank:
        np.testing.assert_array_equal(bank[k], before[k])
    assert qwen.eager_attention_forward is original
    with pytest.raises(ValueError):
        M.native_batch(observer, u, bank, "foreign", [0] * 5)
    assert qwen.eager_attention_forward is original
