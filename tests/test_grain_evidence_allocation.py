#!/usr/bin/env python3
"""Finite bank-law and CPU recursive-reader tests, without pretrained fitting."""

import numpy as np
import pytest

from grain_research.construction import math_evidence_allocation as A
from grain_research.construction import math_join_encoding as E
from grain_research.construction import math_late_condition as L
from tests.test_grain_continuation_routes import tiny


def test_bank_composition_preserves_local_laws_and_outside():
    """Equal bank mass is not equal token mass or removal of contextual sinks."""
    import torch

    p = torch.tensor([0.1, 0.08, 0.12, 0.1, 0.2, 0.3, 0.1], dtype=torch.float64)[
        None, None, None
    ]
    before = p.clone()
    a = A.equal_banks(p, 1, 3, 4, 6)
    torch.testing.assert_close(a[..., 1:3].sum(-1), a[..., 4:6].sum(-1))
    torch.testing.assert_close(a.sum(-1), p.sum(-1))
    for lo, hi in ((1, 3), (4, 6)):
        torch.testing.assert_close(
            a[..., lo:hi] / a[..., lo:hi].sum(-1, keepdim=True),
            p[..., lo:hi] / p[..., lo:hi].sum(-1, keepdim=True),
        )
    assert torch.equal(a[..., [0, 3, 6]], p[..., [0, 3, 6]])
    assert torch.equal(p, before)
    v = torch.arange(7, dtype=torch.float64)
    mx, mz = p[..., 1:3].sum(-1), p[..., 4:6].sum(-1)
    fx = (p[..., 1:3] * v[1:3]).sum(-1) / mx
    fz = (p[..., 4:6] * v[4:6]).sum(-1) / mz
    torch.testing.assert_close(((p - a) * v).sum(-1), (mx - (mx + mz) / 2) * (fx - fz))


def test_already_equal_and_common_values_have_no_local_output_effect():
    """Equal bank laws can change weights but not a common value output."""
    import torch

    p = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.3, 0.3])[None, None, None]
    a = A.equal_banks(p, 1, 3, 4, 6)
    torch.testing.assert_close(p.sum(-1), a.sum(-1))
    equal = torch.tensor([0.1, 0.1, 0.2, 0.3, 0.1, 0.2])[None, None, None]
    torch.testing.assert_close(A.equal_banks(equal, 1, 3, 4, 6), equal)


@pytest.mark.parametrize(
    "bounds", [(0, 2, 4, 6), (1, 1, 4, 6), (1, 3, 3, 6), (1, 3, 4, 8)]
)
def test_missing_bank_or_outside_not_imputed(bounds):
    """Declared observation units must exist at their actual addresses."""
    import torch

    with pytest.raises(ValueError):
        A.equal_banks(torch.ones(1, 1, 1, 7) / 7, *bounds)


def test_zero_mass_is_failure_not_epsilon_rescue():
    """Do not fabricate conditional token weights for an unobserved bank."""
    import torch

    p = torch.ones(1, 1, 1, 7) / 7
    p[..., 1:3] = 0
    with pytest.raises(ValueError):
        A.equal_banks(p, 1, 3, 4, 6)


def test_actual_recursive_adapter_uses_all_layers_and_restores_hook():
    """Actual model operation preserves prefix and runs only during the suffix."""
    import transformers.models.qwen2.modeling_qwen2 as qwen

    model = tiny()
    prefix = [1, 2, 3, 4, 5, 6, 7, 8]
    address = {"body_start": 2, "first_end": 4, "second_start": 5, "prefix_length": 8}
    cache, _ = E.encode(model, prefix, 2, 4)
    before = L.cache_arrays(cache)
    original = qwen.eager_attention_forward
    rows, receipts = A.observe(model, cache, [9, 10], [[11, 12, 13], [14, 15]], address)
    assert len(rows) == len(receipts) == 2
    assert [r["layer"] for r in receipts] == [0, 1]
    assert max(r["outside_error"] for r in receipts) == 0
    assert max(r["bank_balance_error"] for r in receipts) < 1e-6
    assert qwen.eager_attention_forward is original
    for key, value in before.items():
        np.testing.assert_array_equal(value, L.cache_arrays(cache)[key])
    with pytest.raises((IndexError, RuntimeError)):
        A.observe(model, cache, [9], [[999, 12]], address)
    assert qwen.eager_attention_forward is original


def test_signed_recursive_effect_is_not_a_psychological_fraction():
    """Opposing allocation and declared effects are retained."""
    r = A.split([1.0, -2.0], [-3.0, 4.0])
    np.testing.assert_array_equal(r["allocation_increment"], [4.0, -6.0])
    np.testing.assert_array_equal(
        r["native"], r["declared"] + r["allocation_increment"]
    )
    with pytest.raises(ValueError):
        A.split([1.0], [np.nan])
