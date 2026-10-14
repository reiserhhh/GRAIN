#!/usr/bin/env python3
"""The shape adapter changes admission only, not finite-stage arithmetic."""

import numpy as np
import pytest
from test_grain_stage_mass_action import tiny_inputs

from grain_research.construction import math_stage_shape_adapter as A
from grain_research.construction import math_supported_attention_safe as S


def test_actual_history_layer_count_replays_supported_baseline_exactly():
    k, v, m, pos, origin, seeds, cos, sin, weights, cfg = tiny_inputs()
    k, v = k[:, :3], v[:, :3]
    expected = S.construct(k, v, m, pos, origin, seeds, cos, sin, weights, cfg)
    actual = A.construct(k, v, m, m, pos, origin, seeds, cos, sin, weights, cfg, 1)
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(a.numpy(), b.numpy())


def test_placeholder_is_unused_even_when_its_values_are_changed():
    torch = pytest.importorskip("torch")
    k, v, m, pos, origin, seeds, cos, sin, weights, cfg = tiny_inputs()
    k, v = k[:, :3], v[:, :3]
    edited = m.clone()
    edited[:, 0] *= 2
    adapted = A.construct(
        k, v, edited, m, pos, origin, seeds, cos, sin, weights, cfg, 1
    )
    for sentinel in (1e9, float("nan")):
        unused = torch.full_like(k[:, :1], sentinel)
        original = A.SEALED_CONSTRUCT(
            torch.cat((k, unused), 1),
            torch.cat((v, unused), 1),
            edited,
            m,
            pos,
            origin,
            seeds,
            cos,
            sin,
            weights,
            cfg,
            1,
        )
        for a, b in zip(adapted, original, strict=True):
            np.testing.assert_array_equal(a.numpy(), b.numpy())


def test_missing_or_extra_actual_history_layers_are_not_filled():
    k, v, m, pos, origin, seeds, cos, sin, weights, cfg = tiny_inputs()
    for layers in (2, 4):
        with pytest.raises(ValueError, match="per completed block"):
            A.construct(
                k[:, :layers],
                v[:, :layers],
                m,
                m,
                pos,
                origin,
                seeds,
                cos,
                sin,
                weights,
                cfg,
                1,
            )


def failed_state():
    """Reproduce the zero-output guard failure without any scientific readings."""
    return {
        "state": "FAILED",
        "completed_units": 0,
        "pending": [0, 2],
        "seconds": 0.06596758298110217,
        "attempts": 1,
    }


def test_failed_admission_debits_original_budget_without_reset():
    state = failed_state()
    remaining = A.remaining_budget(state, [], 600)
    assert remaining == 599.9340324170189
    assert remaining + state["seconds"] == 600


@pytest.mark.parametrize(
    "change, files",
    [
        ({"completed_units": 2}, []),
        ({"attempts": 2}, []),
        ({"state": "RUNNING"}, []),
        ({"seconds": float("nan")}, []),
        ({"seconds": 601}, []),
        ({}, ["units_0000.npz"]),
    ],
)
def test_started_or_unknown_scientific_work_cannot_be_restarted(change, files):
    with pytest.raises(ValueError, match="zero-output"):
        A.remaining_budget(failed_state() | change, files, 600)
