#!/usr/bin/env python3
"""Result-independent tiny-decoder checks; no public/private model loading."""

import numpy as np
import torch

from grain_research.construction import math_expression_response as E
from grain_research.construction import math_marked_response as M


def tiny_case():
    """Supply a deterministic affine/attention/MLP toy, not a personality world."""
    gen = torch.Generator().manual_seed(1201)
    cfg = {
        "hidden_size": 4,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "rms_norm_eps": 1e-6,
    }
    shapes = {
        "input_layernorm.weight": (4,),
        "post_attention_layernorm.weight": (4,),
        "self_attn.q_proj.weight": (4, 4),
        "self_attn.q_proj.bias": (4,),
        "self_attn.k_proj.weight": (2, 4),
        "self_attn.k_proj.bias": (2,),
        "self_attn.v_proj.weight": (2, 4),
        "self_attn.v_proj.bias": (2,),
        "self_attn.o_proj.weight": (4, 4),
        "mlp.gate_proj.weight": (6, 4),
        "mlp.up_proj.weight": (6, 4),
        "mlp.down_proj.weight": (4, 6),
    }
    weights = [
        {
            name: torch.randn(shape, generator=gen) * 0.2
            for name, shape in shapes.items()
        }
        for _ in range(2)
    ]
    for w in weights:
        w["input_layernorm.weight"] = torch.ones(4)
        w["post_attention_layernorm.weight"] = torch.ones(4)
    final = {
        name: weights[0][name]
        for name in (
            "input_layernorm.weight",
            "self_attn.k_proj.weight",
            "self_attn.k_proj.bias",
            "self_attn.v_proj.weight",
            "self_attn.v_proj.bias",
        )
    }
    weights.append(final)
    k, v = [torch.randn(2, 2, 3, 1, 2, generator=gen) for _ in range(2)]
    seeds = torch.randn(2, 2, 4, generator=gen)
    angle = torch.arange(5.0)[:, None] * torch.tensor([[0.1, 0.1]])
    return k, v, seeds, angle.cos(), angle.sin(), weights, cfg


def test_construct_matches_original_on_binary_support():
    k, v, seeds, cos, sin, weights, cfg = tiny_case()
    gate = torch.tensor([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    actual = M.construct(k, v, gate, seeds, cos, sin, weights, cfg)
    expected = E.compile_histories(
        {"k": k.numpy(), "v": v.numpy()},
        seeds.numpy(),
        cos.numpy(),
        sin.numpy(),
        gate.numpy().astype(bool),
        weights,
        cfg,
        "cpu",
    )
    for a, e in zip(actual, expected, strict=True):
        np.testing.assert_allclose(a.numpy(), e, atol=2e-7)


def test_all_three_history_channels_differentiate_and_match_finite_difference():
    k, v, seeds, cos, sin, weights, cfg = tiny_case()
    gate = torch.tensor([[1.0, 0.4, 1.0], [1.0, 1.0, 0.6]])
    channels = tuple(x.clone().requires_grad_(True) for x in (k, v, gate))
    dk, dv, dg = (torch.full_like(x, 0.1) for x in channels)
    target = sum(
        a.square().sum() for a in M.construct(*channels, seeds, cos, sin, weights, cfg)
    )
    gradients = torch.autograd.grad(target, channels)
    differential = sum(
        (a * b).sum() for a, b in zip(gradients, (dk, dv, dg), strict=True)
    ).item()
    assert all(torch.isfinite(a).all() and a.abs().sum() > 0 for a in gradients)
    values = []
    # The frozen RMS primitive uses float32. A 1e-3 difference of total energies
    # suffered cancellation; use a declared macroscopic finite-difference step.
    epsilon = 0.02
    for sign in (-1, 1):
        point = tuple(
            a.detach() + sign * epsilon * d
            for a, d in zip(channels, (dk, dv, dg), strict=True)
        )
        values.append(
            sum(
                a.square().sum()
                for a in M.construct(*point, seeds, cos, sin, weights, cfg)
            ).item()
        )
    np.testing.assert_allclose(
        differential, (values[1] - values[0]) / (2 * epsilon), atol=2e-5, rtol=2e-3
    )
