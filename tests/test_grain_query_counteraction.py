#!/usr/bin/env python3
"""Exact local exchange, limiting cases and actual block-boundary regression."""

import numpy as np
import pytest
import torch

from grain_research.vector_mechanisms.context_carrier import cached_suffix_kwargs
from grain_research.vector_mechanisms.gated_update import export_weights, pre_mlp
from grain_research.vector_mechanisms.query_counteraction import (
    decompose,
    exchange,
    vector_check,
)
from tests.test_grain_complete_transport import tiny_model


def test_full_exchange_matches_actual_swiglu_jvps():
    model = tiny_model()
    layer = model.model.layers[1]
    w = export_weights(model, block=1)
    rng = np.random.default_rng(46)
    x, d = rng.normal(size=(2, 16)), rng.normal(size=(2, 4, 16))
    v = exchange(x, d, w)
    fn = lambda a: layer.mlp(layer.post_attention_layernorm(a))
    for q in range(2):
        for r in range(2):
            native = []
            for seed in d[r]:
                _, tangent = torch.func.jvp(
                    fn,
                    (torch.tensor(x[q], dtype=torch.float32),),
                    (torch.tensor(seed, dtype=torch.float32),),
                )
                native.append(tangent.detach().numpy())
            assert vector_check(native, v[q, r].sum(0))["pass"]
    z = decompose(v)
    np.testing.assert_allclose(z["carried"] + z["operator"], z["contrast"], atol=1e-15)
    common_x = decompose(exchange(np.repeat(x[:1], 2, axis=0), d, w))
    np.testing.assert_array_equal(common_x["operator"], 0)
    common_d = decompose(exchange(x, np.repeat(d[:1], 2, axis=0), w))
    np.testing.assert_array_equal(common_d["carried"], 0)


def test_primal_capture_stops_before_selected_mlp():
    model = tiny_model()
    x = torch.randn(1, 9, 16)
    kw = cached_suffix_kwargs(model, x)

    def forbidden(*_):
        raise AssertionError("Downstream call")

    handles = [
        m.register_forward_pre_hook(forbidden)
        for m in (model.model.layers[1].mlp, model.model.layers[2], model.lm_head)
    ]
    try:
        value = pre_mlp(model, x, kw, start=0, block=1)
    finally:
        for h in handles:
            h.remove()
    assert value.shape == (1, 16) and torch.isfinite(value).all()


def test_checks_reject_invalid_shapes_and_vector_mismatch():
    assert vector_check(np.zeros((4, 3)), np.zeros((4, 3)))["pass"]
    assert not vector_check(np.ones((4, 3)), np.zeros((4, 3)))["pass"]
    with pytest.raises(ValueError):
        decompose(np.zeros((2, 2, 4, 3)))
    with pytest.raises(ValueError):
        exchange(np.zeros((2, 4)), np.zeros((2, 3, 4)), {})
