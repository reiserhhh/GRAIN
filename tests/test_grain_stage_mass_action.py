#!/usr/bin/env python3
"""Stage scheduling and finite interactions can fail in explicit null worlds."""

import numpy as np
import pytest

from grain_research.construction import math_stage_mass_action as M
from grain_research.construction import math_supported_attention_safe as S


def test_49_arms_preserve_every_inherited_edit_on_one_stage_only():
    a = np.arange(2 * 25 * 5).reshape(2, 25, 5) + 1.0
    early, late = M.mass_pairs(a)
    assert early.shape == late.shape == (2, 49, 5)
    for arm, (q, align, sign, route) in enumerate(M.arm_indices(), 1):
        edited = a[:, 1 + q * 4 + align * 2 + sign]
        np.testing.assert_array_equal(early[:, arm], edited if route == 0 else a[:, 0])
        np.testing.assert_array_equal(late[:, arm], a[:, 0] if route == 0 else edited)


def test_route_addressing_does_not_swap_sign_focus_or_target():
    r = np.arange(288 * 49 * 2 * 2 * 6).reshape(288, 49, 2, 2, 6).astype(float)
    routes = M.route_readings(r)
    for arm, (q, align, sign, route) in enumerate(M.arm_indices(), 1):
        np.testing.assert_array_equal(
            routes[route][:, 1 + q * 4 + align * 2 + sign], r[:, arm]
        )
    for route in routes:
        np.testing.assert_array_equal(route[:, 0], r[:, 0])


def test_additive_world_has_zero_interaction():
    b = np.array([2, 4, 1.0])
    e, l = b + [1, -3, 2], b + [-2, 1, 4]
    account = M.finite_account(b, e, l, e + l - b)
    np.testing.assert_array_equal(account["interaction"], 0)
    np.testing.assert_array_equal(account["total"], account["early"] + account["late"])


def test_suppression_and_synergy_are_not_deleted_as_noise():
    b, e, l, t = np.zeros(2), np.array([2, 1]), np.array([-1, 1]), np.array([0, 5])
    account = M.finite_account(b, e, l, t)
    np.testing.assert_array_equal(account["interaction"], [-1, 3])
    np.testing.assert_array_equal(
        sum(account[k] for k in ("early", "late", "interaction")), t
    )
    assert M.cosine([0, 0], [1, 2]) is None


def tiny_inputs():
    """Deterministic small random arithmetic, not a trained measurement model."""
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(123)
    tensor = lambda shape: torch.tensor(
        rng.normal(size=shape) * 0.2, dtype=torch.float32
    )
    cfg = {
        "hidden_size": 4,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "rms_norm_eps": 1e-6,
    }
    weights = []
    for _ in range(4):
        weights.append(
            {
                "input_layernorm.weight": torch.ones(4),
                "post_attention_layernorm.weight": torch.ones(4),
                "self_attn.q_proj.weight": tensor((4, 4)),
                "self_attn.q_proj.bias": tensor((4,)),
                "self_attn.k_proj.weight": tensor((2, 4)),
                "self_attn.k_proj.bias": tensor((2,)),
                "self_attn.v_proj.weight": tensor((2, 4)),
                "self_attn.v_proj.bias": tensor((2,)),
                "self_attn.o_proj.weight": tensor((4, 4)),
                "mlp.gate_proj.weight": tensor((8, 4)),
                "mlp.up_proj.weight": tensor((8, 4)),
                "mlp.down_proj.weight": tensor((4, 8)),
            }
        )
    return (
        tensor((2, 4, 3, 1, 2)),
        tensor((2, 4, 3, 1, 2)),
        torch.ones(2, 3),
        np.arange(3),
        3,
        tensor((2, 2, 4)),
        torch.ones(5, 2),
        torch.zeros(5, 2),
        weights,
        cfg,
    )


def test_constant_schedule_exactly_replays_supported_constructor():
    k, v, m, pos, origin, seeds, cos, sin, w, cfg = tiny_inputs()
    ref = S.construct(k, v, m, pos, origin, seeds, cos, sin, w, cfg)
    actual = M.construct(k, v, m, m, pos, origin, seeds, cos, sin, w, cfg, 1)
    for a, b in zip(ref, actual, strict=True):
        np.testing.assert_array_equal(a.numpy(), b.numpy())


def test_early_edit_is_propagated_not_reset_at_late_boundary():
    k, v, m, pos, origin, seeds, cos, sin, w, cfg = tiny_inputs()
    edited = m.clone()
    edited[:, 0] *= 2
    base = M.construct(k, v, m, m, pos, origin, seeds, cos, sin, w, cfg, 1)
    early = M.construct(k, v, edited, m, pos, origin, seeds, cos, sin, w, cfg, 1)
    assert any(
        np.max(abs(a.numpy() - b.numpy())) > 1e-7
        for a, b in zip(base, early, strict=True)
    )


def test_invalid_cuts_and_nonfinite_corners_rejected():
    k, v, m, pos, origin, seeds, cos, sin, w, cfg = tiny_inputs()
    for cut in (0, 3):
        with pytest.raises(ValueError):
            M.construct(k, v, m, m, pos, origin, seeds, cos, sin, w, cfg, cut)
    with pytest.raises(ValueError):
        M.finite_account([0], [1], [2], [np.nan])
