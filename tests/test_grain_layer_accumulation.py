#!/usr/bin/env python3
"""Exact layer-update telescoping and truncated execution on native tiny Qwen2."""

import pytest
import torch

from grain_research.vector_mechanisms.complete_transport import forward_response
from grain_research.vector_mechanisms.context_carrier import cached_suffix_kwargs
from grain_research.vector_mechanisms.layer_accumulation import (
    accumulate,
    accumulation_jvp,
    stage_labels,
    truncated_jvp,
)
from tests.test_grain_complete_transport import tiny_model


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_update_sum_and_exit_really_stops(device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    model = tiny_model(device)
    x = torch.randn(1, 9, 16, device=device)
    d = torch.randn_like(x)
    d[:, -2:] = 0
    kw = cached_suffix_kwargs(model, x)
    primal, dots = accumulation_jvp(model, x, kw, d, start=0, exits=(0, 1))
    native, native_dots = forward_response(model, x, kw, d, start=0)
    torch.testing.assert_close(primal[0][-1], native[0], rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(dots[0][-1], native_dots[0], rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(
        dots[0][-1], dots[1].sum(0) + dots[2], rtol=1e-5, atol=1e-7
    )
    assert dots[2].count_nonzero() == 0
    assert len(stage_labels(0, 2)) == dots[1].shape[0] == 5
    for j, end in enumerate((0, 1)):

        def forbidden(*_args):
            raise AssertionError("Later block executed by truncated program")

        handles = [
            m.register_forward_pre_hook(forbidden)
            for m in model.model.layers[end + 1 :]
        ]
        handles.append(model.lm_head.register_forward_pre_hook(forbidden))
        try:
            short = truncated_jvp(model, x, kw, d, end=end, start=0)
        finally:
            for h in handles:
                h.remove()
        torch.testing.assert_close(short, dots[0][j], rtol=1e-5, atol=1e-7)


def test_bad_exits_fail_closed():
    model = tiny_model()
    x = torch.randn(1, 9, 16)
    kw = cached_suffix_kwargs(model, x)
    for bad in ((1, 0), (1, 1), (2,), (-1,)):
        with pytest.raises(ValueError, match="exits"):
            accumulate(model, x, kw, 0, bad)
