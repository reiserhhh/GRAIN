#!/usr/bin/env python3
"""Test link measurement and its source-target meaning, not model accuracy."""

import copy

import numpy as np
import pytest

from grain_research.construction import math_event_link_witness as M
from scripts.run_grain_event_link_witness_hm247 import literal_accounts


def toy():
    """Supply finite exact-composition windows with alternative linkings."""
    x = np.array([[[1.0], [-1.0], [1.0], [-1.0]], [[1.0], [1.0], [-1.0], [-1.0]]])
    return x, np.zeros((2, 4), int)


def test_linear_literal_covariance():
    x, c = toy()
    g = M.grams(x, c, x, c)
    cov = [np.mean(z[:-1] * z[1:]) - z[:-1].mean(0) * z[1:].mean(0) for z in x]
    np.testing.assert_allclose(
        g[0, 3], np.outer(np.ravel(cov), np.ravel(cov)), atol=1e-14
    )
    assert g[0, 3, 0, 0] > 0


def test_all_signed_accounts_sum_to_joint():
    x, c = toy()
    g = M.grams(x, c, x, c)
    full = (
        M.node_kernels(x.reshape(-1, 1), c.ravel(), x.reshape(-1, 1), c.ravel())
        .reshape(2, 2, 4, 2, 4)
        .transpose(0, 1, 3, 2, 4)
    )
    expected = np.mean(full[..., :-1, :-1] * full[..., 1:, 1:], (-2, -1))
    np.testing.assert_allclose(g.sum(1), expected, atol=1e-14)


def test_linking_changes_with_same_endpoint_marginals():
    x = np.array([[[0.0], [1.0], [-1.0], [0.0], [0.0]]])
    c = np.zeros((1, 5), int)
    a = M.grams(x, c, x, c)
    p = np.array([[2, 3, 1, 4]])
    b = M.grams(x, c, x, c, after_y=p)
    np.testing.assert_allclose(a[:, 0], b[:, 0], atol=1e-14)
    assert np.max(abs(a[:, 3] - b[:, 3])) > 0.01


def test_non_linear_links_with_zero_linear_covariance():
    left = np.array([[-1.0], [0.0], [1.0]])
    right = left**2
    c = np.zeros(3, int)
    ks = M.node_kernels(left, c, left, c)
    kr = M.node_kernels(right, c, right, c)
    g = M.pair_accounts(ks[:, None, None], kr[:, None, None])
    assert abs(g[0, 3, 0, 0]) < 1e-14
    assert g[1, 3, 0, 0] > 0.01


def test_product_grid_is_exact_zero_link_world():
    a = np.repeat(np.array([[-1.0], [1.0]]), 2, axis=0)
    b = np.tile(np.array([[-1.0], [1.0]]), (2, 1))
    c = np.zeros(4, int)
    g = M.pair_accounts(
        M.node_kernels(a, c, a, c)[:, None, None],
        M.node_kernels(b, c, b, c)[:, None, None],
    )
    np.testing.assert_allclose(g[:, 3], 0, atol=1e-14)


def test_distance_kernel_matches_explicit_augmented_distance():
    x, c = toy()
    y = x.reshape(-1, 1)
    z = np.c_[y, np.eye(2)[c.ravel()]]
    direct = (
        np.linalg.norm(z, axis=1)[:, None]
        + np.linalg.norm(z, axis=1)[None]
        - np.linalg.norm(z[:, None] - z[None], axis=-1)
    ) / 2
    np.testing.assert_allclose(
        M.node_kernels(y, c.ravel(), y, c.ravel())[1], direct, atol=1e-14
    )


def test_relink_preserves_condition_pairs_and_multisets():
    c = np.array([[0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0]])
    p = M.relink(c)[0]
    np.testing.assert_array_equal(np.sort(p), np.arange(1, 12))
    np.testing.assert_array_equal(2 * c[0, :-1] + c[0, p], 2 * c[0, :-1] + c[0, 1:])
    assert np.count_nonzero(p != np.arange(1, 12)) > 0


def test_targets_do_not_select_one_favorable_axis():
    w = M.target_weights(np.arange(32))
    assert w.shape == (5, 32)
    np.testing.assert_allclose(w.sum(1), 0, atol=1e-14)
    with pytest.raises(ValueError):
        M.target_weights(np.zeros(8, int))


def test_literal_tensor_coefficients_keep_all_mixed_terms():
    x, c = toy()
    c[1] = [0, 1, 1, 0]
    g = M.grams(x, c, x, c)
    for i in range(2):
        for j in range(2):
            direct = literal_accounts(x[i], c[i], x[j], c[j])
            np.testing.assert_allclose(g[:, :, i, j], direct, atol=1e-14)


def record():
    return {
        "record_id": "u/0",
        "author": "u",
        "regime": "raw_pandora_publication",
        "events": [
            {
                "event_id": f"e{i}",
                "owner": "u",
                "position": i,
                "timestamp": float(i),
                "vector": [float(i)],
                "condition": i % 2,
            }
            for i in range(12)
        ],
    }


def test_display_order_is_not_attested_order():
    r = record()
    a = M.canonical_window(r, 1)
    r["events"].reverse()
    b = M.canonical_window(r, 1)
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)


def test_cold_fixed_reference_rejects_answers_and_keeps_components():
    r = record()
    source = np.array([[[float(t + u)] for t in range(12)] for u in range(32)])
    conditions = np.zeros((32, 12), int)
    w = M.target_weights(np.arange(32))
    a = M.cold_read(source, conditions, w, r)
    r["events"].reverse()
    b = M.cold_read(source, conditions, w, r)
    np.testing.assert_array_equal(a["attested_accounts"], b["attested_accounts"])
    for arm in ("attested", "relinked_audit"):
        np.testing.assert_allclose(
            np.asarray(a[arm + "_accounts"]).sum(1), a[arm + "_total"], atol=1e-14
        )
    assert np.asarray(a["attested_accounts"]).shape == (2, 4, 5)
    assert a["audit_is_new_human_text"] is False
    r["events"][0]["answer"] = 1
    with pytest.raises(ValueError):
        M.cold_read(source, conditions, w, r)


@pytest.mark.parametrize(
    "bad", ("duplicate", "owner", "time", "position", "answer", "regime")
)
def test_qualification_rejects_invalid_evidence(bad):
    r = copy.deepcopy(record())
    if bad == "duplicate":
        r["events"][1]["event_id"] = "e0"
    elif bad == "owner":
        r["events"][1]["owner"] = "v"
    elif bad == "time":
        r["events"][1]["timestamp"] = 0.0
    elif bad == "position":
        r["events"][1]["position"] = 3
    elif bad == "answer":
        r["target_answer"] = 1
    else:
        r["regime"] = "virtual_order"
    with pytest.raises(ValueError):
        M.canonical_window(r, 1)
