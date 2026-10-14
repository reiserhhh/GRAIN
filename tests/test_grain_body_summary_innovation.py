#!/usr/bin/env python3
"""Finite algebra and actual tiny-network body-access contrast witnesses."""

import copy

import numpy as np
import pytest

from grain_research.construction import math_body_summary_innovation as I
from grain_research.construction import math_relay_value_summary as V
from tests.test_grain_query_relay_cut import tiny_observer
from tests.test_grain_relay_value_summary import unit


def test_template_preserving_increment_algebra():
    """Common increments preserve baselines, unlike averaging complete values."""
    rng = np.random.default_rng(203)
    n, b = (rng.normal(size=(3, 5, 1, 3, 8)) for _ in range(2))
    before = [n.copy(), b.copy()]
    np.testing.assert_allclose(I.increment_donor(n, b, "self"), n, atol=1e-15)
    np.testing.assert_array_equal(I.increment_donor(n, b, "zero"), b)
    common = I.increment_donor(n, b, "common")
    np.testing.assert_allclose(
        common - b, np.broadcast_to((n - b).mean(1, keepdims=True), n.shape)
    )
    assert not np.allclose(common, V.summary_values(n, "common"))
    foreign = I.increment_donor(n, b, "foreign", [4, 2, 1, 3, 0])
    np.testing.assert_allclose(foreign[:, 3], n[:, 3], atol=1e-15)
    np.testing.assert_array_equal(n, before[0])
    np.testing.assert_array_equal(b, before[1])
    with pytest.raises(ValueError):
        I.increment_donor(n, b[:, :4], "zero")


def test_actual_blind_content_invariance_and_identity():
    """Blind suffix values ignore body contents at fixed positions, not length."""
    import transformers.models.qwen2.modeling_qwen2 as qwen

    observer, u = tiny_observer(), unit()
    original, original_summary = qwen.eager_attention_forward, V.summary_values
    native = V.native_batch(observer, u)
    blind = I.blind_batch(observer, u)
    other = copy.deepcopy(u)
    for row in other["input_ids"]:
        for ix in other["body"]:
            row[ix] = 24 + ix
    changed = I.blind_batch(observer, other)
    np.testing.assert_array_equal(blind[5]["values"], changed[5]["values"])
    assert max(r["relay_body_weight_max"] for r in blind[7]) == 0
    assert np.max(abs(native[5]["values"] - blind[5]["values"])) > 1e-8
    result = I.reader_batch(observer, u, native[5], blind[5]["values"], "self")
    np.testing.assert_allclose(result[0], native[0], atol=3e-7, rtol=0)
    assert qwen.eager_attention_forward is original
    assert V.summary_values is original_summary


def test_affine_permutation_composition_and_reynolds_projection():
    """The baseline-preserving action composes while its mean is group invariant."""
    rng = np.random.default_rng(204)
    n, b = (rng.normal(size=(2, 5, 1, 4, 6)) for _ in range(2))
    pi, sigma = np.array([4, 2, 1, 3, 0]), np.array([1, 2, 3, 4, 0])
    a = I.increment_donor(n, b, "foreign", pi.tolist())
    composed = I.increment_donor(a, b, "foreign", sigma.tolist())
    expected = I.increment_donor(n, b, "foreign", pi[sigma].tolist())
    np.testing.assert_allclose(composed, expected, atol=2e-15)
    np.testing.assert_allclose(np.linalg.norm(a - b), np.linalg.norm(n - b), atol=2e-15)
    common = I.increment_donor(n, b, "common")
    np.testing.assert_allclose(
        I.increment_donor(common, b, "foreign", pi.tolist()), common, atol=2e-15
    )
    np.testing.assert_allclose(I.increment_donor(a, b, "common"), common, atol=2e-15)


def test_actual_readers_fixed_p_immutable_native_and_C():
    """All conditions keep native p/prefix; foreign C remains an identity."""
    import transformers.models.qwen2.modeling_qwen2 as qwen

    observer, u = tiny_observer(), unit()
    native, blind = V.native_batch(observer, u), I.blind_batch(observer, u)
    before = {k: a.copy() for k, a in native[5].items()}
    bv = blind[5]["values"].copy()
    original, original_summary = qwen.eager_attention_forward, V.summary_values
    results = [
        I.reader_batch(
            observer,
            u,
            native[5],
            bv,
            mode,
            [4, 2, 1, 3, 0] if mode == "foreign" else None,
        )
        for mode in ("zero", "common", "foreign")
    ]
    for r in results:
        assert max(b["fixed_probability_error"] for b in r[6]) == 0
        assert max(b["prefix_V_error"] for b in r[6]) == 0
    np.testing.assert_allclose(results[2][0][3], native[0][3], atol=3e-7, rtol=0)
    assert np.max(abs(results[0][0] - native[0])) > 1e-8
    for k, a in native[5].items():
        np.testing.assert_array_equal(a, before[k])
    np.testing.assert_array_equal(bv, blind[5]["values"])
    assert qwen.eager_attention_forward is original
    assert V.summary_values is original_summary
    with pytest.raises(ValueError):
        I.reader_batch(observer, u, native[5], bv, "foreign", [0] * 5)
    assert qwen.eager_attention_forward is original
    assert V.summary_values is original_summary
