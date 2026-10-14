#!/usr/bin/env python3
"""Device compatibility and bounded recovery, not trained score optimization."""

import numpy as np
import pytest

from grain_research.construction import math_body_summary_innovation as I
from grain_research.construction import math_body_summary_mps_compat as C
from grain_research.construction import math_relay_value_summary as V
from scripts.run_grain_body_summary_mps_recovery_hm203 import recovery_counts
from tests.test_grain_query_relay_cut import tiny_observer
from tests.test_grain_relay_value_summary import unit


@pytest.mark.parametrize("mode", ["self", "zero", "common", "foreign"])
def test_float32_formula_compatibility_and_immutability(mode):
    """The installed bank equals the original float64 formula rounded once."""
    rng = np.random.default_rng(203)
    native, blind = [rng.normal(size=(2, 5, 1, 3, 8)).astype("f4") for _ in range(2)]
    before = native.copy(), blind.copy()
    mapping = [4, 2, 1, 3, 0] if mode == "foreign" else None
    result = C.compatible_donor(native, blind, mode, mapping)
    expected = I.increment_donor(native, blind, mode, mapping).astype("f4")
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(native, before[0])
    np.testing.assert_array_equal(blind, before[1])
    if mode in ("self", "foreign"):
        np.testing.assert_array_equal(result[:, 3], native[:, 3])
    with pytest.raises(ValueError, match="float32"):
        C.compatible_donor(native.astype(float), blind, mode, mapping)


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_actual_device_reader_and_hook_restoration(device):
    """A small untrained model tests the actual tensor installation on MPS."""
    import torch
    import transformers.models.qwen2.modeling_qwen2 as qwen

    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable; no trained fallback or dependency install")
    observer, u = tiny_observer(), unit()
    observer.model.to(device)
    native = V.native_batch(observer, u)
    blind = I.blind_batch(observer, u)
    original, original_summary = qwen.eager_attention_forward, V.summary_values
    bank_before = {k: a.copy() for k, a in native[5].items()}
    for mode in ("self", "zero", "common", "foreign"):
        result = C.reader_batch(
            observer,
            u,
            native[5],
            blind[5]["values"],
            mode,
            [4, 2, 1, 3, 0] if mode == "foreign" else None,
        )
        assert result[7]["installed_dtype"] == "float32"
        assert result[7]["scientific_formula_changed"] is False
        assert max(r["fixed_probability_error"] for r in result[6]) == 0
        assert max(r["prefix_V_error"] for r in result[6]) == 0
        if mode == "self":
            np.testing.assert_allclose(result[0], native[0], atol=3e-7, rtol=0)
        if mode == "foreign":
            np.testing.assert_allclose(result[0][3], native[0][3], atol=3e-7, rtol=0)
    for key, a in bank_before.items():
        np.testing.assert_array_equal(native[5][key], a)
    with pytest.raises(ValueError):
        C.reader_batch(observer, u, native[5], blind[5]["values"], "foreign", [0] * 5)
    assert qwen.eager_attention_forward is original
    assert V.summary_values is original_summary


def test_failed_attempt_not_counted_as_success():
    """One reused capture is not a new call or an erased failed attempt."""
    result = recovery_counts({"attempts": 2, "completed": 1}, 385, 385)
    assert result["combined_attempts"] == 387
    assert result["combined_completed"] == 386
    assert result["original_failed_attempts"] == 1
