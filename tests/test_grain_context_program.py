#!/usr/bin/env python3
"""Construction Jacobians, exact cropped replay and contextual chart identities."""

import copy

import numpy as np
import pytest
import torch
from test_grain_upstream_routes import fixture
from transformers import Qwen2Config, Qwen2Model

from grain_research.vector_mechanisms.context_program import (
    CroppedContext,
    capture_context_jacobians,
    cropped_config_dict,
    score_context_charts,
)


def test_cropped_replay_and_context_jacobian():
    obs, f = fixture()
    ids, mask = [1, 3, 6, 9, 2], [False, False, True, True, False]
    charts = capture_context_jacobians(obs, ids, mask, f, cuts=(0, 1), block=2)
    cfg = copy.deepcopy(obs.model.config)
    cfg.num_hidden_layers = 2
    cfg._attn_implementation = "eager"
    cropped = Qwen2Model(cfg).double()
    state = {
        k: v
        for k, v in obs.model.model.state_dict().items()
        if not k.startswith("layers.2.")
    }
    cropped.load_state_dict(state, strict=True)
    run = CroppedContext(cropped, cuts=(0, 1), max_tokens=32)
    states = run.capture(ids)
    for k in (0, 1):
        assert np.allclose(states[k], charts[k]["hidden"], atol=1e-7)
        out = score_context_charts([charts[k], charts[k]], states[k], mask, f.anchor)
        for value in out.values():
            assert value["score"] == pytest.approx(
                f.anchor + charts[k]["projection_origin"].sum(), abs=1e-7
            )
    # A numerical derivative at the last retained cut checks graph continuity.
    delta = torch.randn(1, len(ids), 16, dtype=torch.float64) * 0.001
    values = []
    for sign in (-1, 1):

        def perturb(_module, _inputs, output, sign=sign):
            return output + sign * delta

        handle = obs.model.model.layers[1].register_forward_hook(perturb)
        try:
            result = capture_context_jacobians(obs, ids, mask, f, cuts=(0, 1), block=2)
            values.append(result[1]["projection_origin"].sum())
        finally:
            handle.remove()
    analytic = np.sum(charts[1]["gradients"].sum(0) * delta.numpy()[0])
    assert (values[1] - values[0]) / 2 == pytest.approx(analytic, abs=3e-6)
    assert all(not layer._forward_hooks for layer in cropped.layers)
    assert not hasattr(cropped, "lm_head") and len(cropped.layers) == 2


def test_context_query_mismatch_rejected():
    obs, f = fixture()
    ids, mask = [1, 3, 6, 9, 2], [False, False, True, True, False]
    c = capture_context_jacobians(obs, ids, mask, f, cuts=(0,), block=2)[0]
    with pytest.raises(ValueError, match="Query rows"):
        score_context_charts(
            [c, c], c["hidden"], [False, False, False, True, False], f.anchor
        )


def test_export_config_truncates_strict_layer_metadata():
    obs, _ = fixture()
    original = obs.model.config.to_dict()
    cfg = Qwen2Config.from_dict(cropped_config_dict(original, 2))
    assert cfg.num_hidden_layers == len(cfg.layer_types) == 2
    assert original["num_hidden_layers"] == len(original["layer_types"]) == 3
