#!/usr/bin/env python3
"""Causal support tests using random local CPU weights, without training."""

import numpy as np
import pytest

from grain_research.construction import math_join_encoding as E
from grain_research.construction import math_late_condition as L
from tests.test_grain_continuation_routes import tiny


def test_seam_and_later_rows_cannot_leak_first_comment():
    """Only first-event keys into later-prefix rows are closed."""
    import torch

    mask = torch.zeros(1, 1, 9, 9)
    original = mask.clone()
    changed = E.isolate_mask(mask, 9, 2, 5)
    expected = original.clone()
    expected[:, :, 5:, 2:5] = -torch.inf
    assert torch.equal(changed, expected)
    assert torch.equal(mask, original)
    assert torch.isfinite(changed[:, :, 5:, :2]).all()
    assert torch.isfinite(changed[:, :, 5:, 5:]).all()


@pytest.mark.parametrize("start,end", [(0, 3), (2, 2), (3, 9), (4, 2)])
def test_empty_or_unqualified_blocks_rejected(start, end):
    """A missing event cannot become an implicit zero channel."""
    import torch

    with pytest.raises(ValueError):
        E.isolate_mask(torch.zeros(1, 1, 9, 9), 9, start, end)


def test_isolated_bank_remains_jointly_readable():
    """Later evidence cache is independent, but the native suffix can read both."""
    import transformers.models.qwen2.modeling_qwen2 as qwen

    model = tiny()
    x = [1, 2, 3, 4, 5, 6, 7, 8]
    y = [1, 2, 4, 3, 5, 6, 7, 8]
    hook = qwen.eager_attention_forward
    cx, rx = E.encode(model, x, 2, 4)
    cy, ry = E.encode(model, y, 2, 4)
    a, b = L.cache_arrays(cx), L.cache_arrays(cy)
    assert len(rx) == len(ry) == 2
    assert all(r["closed_weight"] == 0 for r in rx + ry)
    for key in a:
        np.testing.assert_array_equal(a[key][:, :, 4:], b[key][:, :, 4:])
        np.testing.assert_array_equal(a[key][:, :, :2], b[key][:, :, :2])
    native = L.cache_arrays(L.encode_context(model, x))
    for key in a:
        np.testing.assert_array_equal(a[key][:, :, :4], native[key][:, :, :4])
    p = L.observe_suffixes(model, cx, [[9, 10]], [11, 12])
    q = L.observe_suffixes(model, cy, [[9, 10]], [11, 12])
    assert np.max(abs(p["chosen"] - q["chosen"])) > 1e-9
    assert qwen.eager_attention_forward is hook


def test_hook_restored_after_model_error():
    """A failed forward must not modify later observations or other processes."""
    import transformers.models.qwen2.modeling_qwen2 as qwen

    model = tiny()
    old = qwen.eager_attention_forward
    with pytest.raises((IndexError, RuntimeError)):
        E.encode(model, [1, 2, 999, 4, 5, 6], 2, 4)
    assert qwen.eager_attention_forward is old


def test_signed_encoding_increment_not_variance_share():
    """Opposing reading and encoding effects reconstruct the actual native effect."""
    result = E.split(np.array([1.0, -2.0]), np.array([-3.0, 4.0]))
    np.testing.assert_array_equal(result["encoding_increment"], [4.0, -6.0])
    np.testing.assert_array_equal(
        result["native"], result["isolated"] + result["encoding_increment"]
    )


def test_failed_arm_is_not_imputed():
    """No omitted or nonfinite observation is an isolated score of zero."""
    with pytest.raises(ValueError):
        E.split([1.0, 2.0], [np.nan, 3.0])
