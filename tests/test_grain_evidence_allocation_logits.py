#!/usr/bin/env python3
"""Logit-supported conditional composition, including flushed native bank masses."""

import pytest

from grain_research.construction import math_evidence_allocation as A
from grain_research.construction import math_evidence_allocation_logits as C
from grain_research.construction import math_join_encoding as E
from tests.test_grain_continuation_routes import tiny


def test_direct_logits_equal_weight_conditionals_when_supported():
    """Two representations give the SAME operation when normalized mass exists."""
    import torch

    logits = torch.tensor([2.0, 1.0, 3.0, -2.0, 0.5, 1.0, 0.0])[None, None, None]
    weights = logits.softmax(-1)
    torch.testing.assert_close(
        C.conditional_banks(weights, logits, 1, 3, 4, 6),
        A.equal_banks(weights, 1, 3, 4, 6),
    )


def test_finite_logits_define_a_bank_even_when_native_mass_underflows():
    """No epsilon or guessed uniform law is used at a zero normalized bank."""
    import torch

    logits = torch.tensor([0.0, -200.0, -202.0, -4.0, -1.0, -3.0, -2.0])[
        None, None, None
    ]
    weights = logits.softmax(-1)
    assert weights[..., 1:3].sum() == 0
    changed = C.conditional_banks(weights, logits, 1, 3, 4, 6)
    assert changed[..., 1:3].sum() > 0
    torch.testing.assert_close(
        changed[..., 1:3] / changed[..., 1:3].sum(-1, keepdim=True),
        logits[..., 1:3].softmax(-1),
    )
    assert torch.equal(changed[..., [0, 3, 6]], weights[..., [0, 3, 6]])
    torch.testing.assert_close(changed.sum(-1), weights.sum(-1))


def test_structurally_masked_bank_is_not_recovered():
    """Finite score underflow differs from an actually unavailable evidence key."""
    import torch

    logits = torch.zeros(1, 1, 1, 7)
    logits[..., 1:3] = -torch.inf
    with pytest.raises(ValueError):
        C.conditional_banks(logits.softmax(-1), logits, 1, 3, 4, 6)


def test_recursive_reader_receipts_and_hook_restoration():
    """The recovery remains one all-layer operation on the same event banks."""
    import transformers.models.qwen2.modeling_qwen2 as qwen

    model = tiny()
    cache, _ = E.encode(model, [1, 2, 3, 4, 5, 6, 7, 8], 2, 4)
    old = qwen.eager_attention_forward
    _, receipts = C.observe(
        model,
        cache,
        [9, 10],
        [[11, 12], [13, 14, 15]],
        {"body_start": 2, "first_end": 4, "second_start": 5, "prefix_length": 8},
    )
    assert len(receipts) == 2 and all(r["finite_bank_logits"] for r in receipts)
    assert max(r["outside_error"] for r in receipts) == 0
    assert qwen.eager_attention_forward is old
