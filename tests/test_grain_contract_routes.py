#!/usr/bin/env python3
"""Finite route worlds and equal-length architecture witnesses, not accuracy."""

import numpy as np
import pytest

from grain_research.construction import math_contract_routes as M
from tests.test_grain_query_relay_cut import tiny_observer


def unit():
    return {
        "input_ids": [[1, 2, 3, 4, 5] + [6 + q] * (q + 3) for q in range(6)],
        "body": [2, 3, 4],
    }


def test_exact_two_route_masks_and_preservation():
    import torch

    u, mask = unit(), torch.zeros(6, 1, 13, 13)
    for r, d in M.ROUTES.values():
        a = M.cut_mask(mask, u, r, d)
        assert torch.isneginf(a[0, 0, 5:7, 2:5]).all().item() == (not r)
        assert torch.isneginf(a[0, 0, 7, 2:5]).all().item() == (not d)
        assert (a[0, 0, :5] == 0).all()
        assert (a[0, 0, :, :2] == 0).all()
    assert (mask == 0).all()
    with pytest.raises(ValueError):
        M.cut_mask(mask, u, 2, 0)
    with pytest.raises(ValueError):
        M.support({**u, "body": [2, 4]})


def test_additive_redundant_synergistic_worlds():
    for values, expected in [
        ([[0, 2], [3, 5]], 0),
        ([[0, 2], [2, 2]], -2),
        ([[0, 0], [0, 4]], 4),
    ]:
        a = np.array([values], float)
        f = M.route_accounts(a)
        assert f["interaction"][0] == expected
        np.testing.assert_allclose(
            f["direct_increment"] + f["relay_increment"] + f["interaction"],
            f["joint_increment"],
        )
    rng = np.random.default_rng(222)
    a = rng.normal(size=(7, 2, 2, 3, 2, 4))
    f = M.route_accounts(a)
    np.testing.assert_allclose(
        f["relay_with_direct"], f["relay_increment"] + f["interaction"], atol=2e-15
    )
    with pytest.raises(ValueError):
        M.route_accounts(np.zeros((3, 4)))


def test_content_blind_closed_reference_and_both_retained_routes():
    observer, u = tiny_observer(), unit()
    changed = {**u, "input_ids": [r[:2] + [23, 24, 25] + r[5:] for r in u["input_ids"]]}
    for name, (r, d) in M.ROUTES.items():
        a = M.native_batch(observer, u, relay_open=r, direct_open=d)
        b = M.native_batch(observer, changed, relay_open=r, direct_open=d)
        if name == "closed":
            np.testing.assert_array_equal(a[0], b[0])
            np.testing.assert_array_equal(a[1], b[1])
        else:
            assert np.max(abs(a[0] - b[0])) > 1e-9
        if not r:
            assert [x["relay_KV_hash"] for x in a[2]] == [
                x["relay_KV_hash"] for x in b[2]
            ]


def test_hook_restored_after_failure_and_body_prefix_identical():
    import transformers.models.qwen2.modeling_qwen2 as qwen

    observer, u = tiny_observer(), unit()
    original = qwen.eager_attention_forward
    a = M.native_batch(observer, u, relay_open=0, direct_open=1)
    b = M.native_batch(observer, u, relay_open=1, direct_open=0)
    assert [(x["body_key_hash"], x["body_value_hash"]) for x in a[2]] == [
        (x["body_key_hash"], x["body_value_hash"]) for x in b[2]
    ]
    with pytest.raises(ValueError):
        M.native_batch(observer, u, relay_open=2, direct_open=0)
    assert qwen.eager_attention_forward is original
