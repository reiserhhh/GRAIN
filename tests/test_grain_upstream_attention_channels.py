#!/usr/bin/env python3
"""Real tiny-Qwen parity, fixed-query reduction and boundary tests."""

import numpy as np
import pytest
import torch

from grain_research.vector_mechanisms.context_carrier import cached_suffix_kwargs
from grain_research.vector_mechanisms.fixed_seed_attention import decompose_attention
from grain_research.vector_mechanisms.upstream_attention_channels import (
    decompose,
    export_weights,
    normalized_directions,
    pre_attention,
)
from tests.test_grain_complete_transport import tiny_model


@pytest.mark.parametrize("masked", [False, True])
def test_actual_qwen_jvp_with_nonzero_query_gqa_rope_and_mask(masked):
    model = tiny_model()
    x = torch.randn(1, 9, 16)
    d = torch.randn(4, 9, 16)
    kw = cached_suffix_kwargs(model, x)
    if masked:
        kw["attention_mask"] = kw["attention_mask"].clone()
        kw["attention_mask"][..., -1, 3] = -torch.inf
    cos, sin = [v.numpy()[0] for v in kw["position_embeddings"]]
    mask = kw["attention_mask"][0, 0, -1].numpy()
    w = export_weights(model, block=1)
    p = decompose(x.numpy()[0], d.numpy(), cos, sin, mask, w)
    a = model.model.layers[1]
    fn = lambda y: a.self_attn(a.input_layernorm(y), **kw)[0][:, -1]
    for i in range(4):
        value, tangent = torch.func.jvp(fn, (x,), (d[i : i + 1],))
        np.testing.assert_allclose(p["primal"], value.numpy()[0], atol=1e-7, rtol=2e-5)
        np.testing.assert_allclose(
            p["channels"][:, i].sum(0), tangent.numpy()[0], atol=1e-7, rtol=2e-5
        )
    assert np.linalg.norm(p["channels"][0]) > 0
    z = decompose(x.numpy()[0], np.zeros_like(d.numpy()), cos, sin, mask, w)
    np.testing.assert_array_equal(z["channels"], 0)


def test_body_only_reduces_to_old_key_value_pullback():
    model = tiny_model()
    rng = np.random.default_rng(49)
    x, d = rng.normal(size=(9, 16)), rng.normal(size=(4, 9, 16))
    d[:, -1] = 0
    w = export_weights(model, block=1)
    kw = cached_suffix_kwargs(model, torch.tensor(x[None], dtype=torch.float32))
    cos, sin = [v.numpy()[0] for v in kw["position_embeddings"]]
    p = decompose(x, d, cos, sin, np.zeros(9), w)
    np.testing.assert_array_equal(p["channels"][0], 0)
    z, dz = normalized_directions(x, d, w["gamma"], float(w["epsilon"]))
    readout = rng.normal(size=16)
    g = (w["o"].T @ readout).reshape(4, 4)
    old = decompose_attention(
        (z[-1] @ w["q"].T + w["q_bias"]).reshape(4, 4),
        (z @ w["k"].T + w["k_bias"]).reshape(9, 2, 4),
        (z @ w["v"].T + w["v_bias"]).reshape(9, 2, 4),
        g,
        cos,
        sin,
        w["k"],
        w["v"],
    )
    for i, label in ((1, "key_field"), (2, "value_field")):
        np.testing.assert_allclose(
            p["channels"][i] @ readout,
            np.einsum("th,fth->f", old[label], dz),
            atol=1e-15,
        )


def test_full_boundary_stops_before_selected_attention():
    model = tiny_model()
    x, d = torch.randn(1, 9, 16), torch.randn(1, 9, 16)
    kw = cached_suffix_kwargs(model, x)

    def forbidden(*_):
        raise AssertionError("Selected attention or downstream called")

    handles = [
        m.register_forward_pre_hook(forbidden)
        for m in (*model.model.layers[1:], model.lm_head)
    ]
    try:
        primal, tangent = torch.func.jvp(
            lambda h: pre_attention(model, h, kw, 0, 1), (x,), (d,)
        )
    finally:
        for h in handles:
            h.remove()
    assert primal.shape == tangent.shape == x.shape
    torch.testing.assert_close(primal, model.model.layers[0](x, **kw))


def test_invalid_full_sequence_and_mask_rejected():
    w = export_weights(tiny_model(), block=1)
    args = [
        np.ones((9, 16)),
        np.ones((4, 9, 16)),
        np.ones((9, 4)),
        np.zeros((9, 4)),
        np.zeros(9),
        w,
    ]
    with pytest.raises(ValueError):
        decompose(args[0], args[1][:, -1], *args[2:])
    args[4] = np.full(9, -np.inf)
    with pytest.raises(ValueError):
        decompose(*args)
