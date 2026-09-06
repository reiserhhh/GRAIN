"""Synthetic mask and causal endpoint checks, without private data."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from grain_research.vector_mechanisms.response_access import (
    CUTS,
    SITES,
    access_forward,
    conditions,
    cut_mask,
    summarize,
)
from grain_research.vector_mechanisms.response_context import ResponseObserver


def test_mask_preserves_all_edges_outside_requested_rectangle():
    m = torch.full((1, 1, 8, 8), torch.finfo(torch.float32).min)
    m[0, 0] = torch.triu(m[0, 0], diagonal=1)
    m[..., :2] = -torch.inf
    before = m.clone()
    cut, n = cut_mask(m, 4, 7)
    assert n == 6
    assert torch.equal(m, before)
    changed = cut != m
    expected = torch.zeros_like(changed)
    expected[..., 4:7, 2:4] = True
    assert torch.equal(changed, expected)
    assert torch.isneginf(cut[..., :2]).all()
    sham, count = cut_mask(m, 4, 7, sham=True)
    assert count == n and torch.equal(sham, m)


@pytest.mark.parametrize(
    "mask",
    [
        None,
        torch.zeros(8, 8),
        torch.zeros(1, 1, 8, 8, dtype=torch.bool),
        torch.zeros(1, 2, 8, 8),
        torch.zeros(1, 1, 8, 7),
    ],
)
def test_reject_unsupported_attention_masks(mask):
    with pytest.raises(ValueError, match="mask"):
        cut_mask(mask, 4, 7)


def test_tiny_actual_decoder_access_algebra_and_hook_cleanup():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(731)
    torch.set_num_threads(2)
    config = Qwen2Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=28,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=2048,
    )
    config._attn_implementation = "eager"
    model = Qwen2ForCausalLM(config).float().eval()
    o = ResponseObserver(
        model,
        SimpleNamespace(pad_token_id=0),
        blocks=SITES,
        capture_sites=[f"block_{i:02}" for i in SITES],
    )
    rows, checks = conditions(
        lambda p, b, **kw: access_forward(o, p, b, **kw),
        [2, 4, 5],
        [3, 6, 7, 8],
        [11, 12, 13, 14],
    )
    assert len(rows) == 18 and o.calls == 18
    assert checks["cut0_tail_error"] <= 1e-5
    assert max(checks["patch_cut_errors"].values()) <= 1e-5
    assert (
        checks["mask_sham_error"]
        == checks["self_sham_error"]
        == checks["first_token_error"]
        == 0
    )
    assert all(len(m._forward_pre_hooks) == 0 for m in model.model.layers)
    # Unsupported compact mode must fail rather than shifting intervention origin.
    with pytest.raises(ValueError, match="origin"):
        access_forward(o, [2], [3, 4], cut=10, compact=True)
    # Fault while a hook is installed must not leak into the next model call.
    with pytest.raises(ValueError, match="span"):
        access_forward(
            o, [2], [3, 4], cut=10, patch_site="block_16", patch=np.zeros((1, 16))
        )
    assert all(len(m._forward_pre_hooks) == 0 for m in model.model.layers)


def test_summary_retains_zero_negative_and_nonmonotonic_cases():
    rows = []
    for person, x in enumerate([0, -0.2, 1.0]):
        for k in CUTS:
            rows.extend(
                [
                    {
                        "participant_code": person,
                        "condition": f"actual_{k}",
                        "tail_logp": x * (k + 1) / 28,
                    },
                    {
                        "participant_code": person,
                        "condition": f"matched_{k}",
                        "tail_logp": 0,
                    },
                ]
            )
        for site in SITES:
            rows.extend(
                [
                    {
                        "participant_code": person,
                        "condition": f"patch_{site}_{label}",
                        "tail_logp": x * f,
                    }
                    for label, f in (("cut", (site + 1) / 28), ("full", 0.5))
                ]
            )
    s = summarize(pd.DataFrame(rows))
    assert s["effects"]["-1"]["retained_context_difference"]["mean"] == 0
    assert s["effects"]["27"]["differential_access_effect"]["mean"] == 0
    assert s["effects"]["10"]["retained_context_difference"]["participants"] == 3
    assert s["patch_effects"]["22"]["full_minus_cut_patch_effect"]["mean"] < 0
