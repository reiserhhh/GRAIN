#!/usr/bin/env python3
"""Disjoint route means and mechanistic controls before model execution."""

import numpy as np
import pytest

from grain_research.construction import math_ownership_route_factorization as M


def states():
    """Create matched-length g partners and an independently specified low suffix."""
    a, b = np.random.default_rng(75).normal(size=(2, 7, 5)).astype(np.float32)
    return a, b, a[3:].copy()


@pytest.mark.parametrize("arm", M.ARMS)
def test_region_assignments_and_finite_outputs(arm):
    """Declared operations never silently alter undeclared cut positions."""
    a, b, low = states()
    positions, values = M.assignment(a, b, 3, arm, low, -1, 1)
    out = a.copy()
    out[positions] = values
    if arm == "suffix_even":
        assert np.array_equal(out[:3], a[:3])
    if arm in ("body_even", "random_body"):
        assert np.array_equal(out[3:], a[3:])
    assert values.dtype == np.float32 and np.isfinite(values).all()


def test_both_even_identical_cut_is_algebraic_control():
    """Averaging both regions eliminates g at the full cut, not just the output."""
    a, b, low = states()
    p, v = M.assignment(a, b, 3, "both_even", low, -1, 1)
    q, w = M.assignment(b, a, 3, "both_even", low, 1, 1)
    assert np.array_equal(p, q) and np.array_equal(v, w)
    assert np.array_equal(
        v, ((a.astype(float) + b.astype(float)) / 2).astype(np.float32)
    )


def test_random_pair_mean_norm_and_independent_seed():
    """Random directions retain pair mean and intervention norm, not output tuning."""
    a, b, low = states()
    _, va = M.assignment(a, b, 3, "random_body", low, -1, 81)
    _, vb = M.assignment(b, a, 3, "random_body", low, 1, 81)
    _, ve = M.assignment(a, b, 3, "body_even", low, -1, 81)
    assert np.max(abs(va.astype(float) + vb.astype(float) - a[:3] - b[:3])) < 3e-7
    assert (
        np.max(
            abs(np.linalg.norm(a[:3] - va, axis=1) - np.linalg.norm(a[:3] - ve, axis=1))
        )
        < 3e-7
    )
    _, vd = M.assignment(a, b, 3, "random_body", low, -1, 82)
    assert not np.array_equal(va, vd)


def test_low_suffix_reuses_frozen_payload_exactly():
    """The partial query ablation is not recalibrated by route outcomes."""
    a, b, low = states()
    low += 2
    for arm in ("body_even_suffix_remove_both", "random_body_suffix_remove_both"):
        _, v = M.assignment(a, b, 3, arm, low, 1, 1)
        assert np.array_equal(v[3:], low)


def test_distinct_routes_are_observationally_equal_before_intervention():
    """Native accuracy cannot distinguish three different causal route arrangements."""
    body, suffix, redundant = (M.toy_routes(k) for k in ("body", "suffix", "redundant"))
    assert np.array_equal(body["native"], suffix["native"])
    assert np.array_equal(body["native"], redundant["native"])
    assert np.array_equal(body["body_even"], [0, 0])
    assert np.array_equal(suffix["suffix_even"], [0, 0])
    assert np.array_equal(redundant["body_even"], redundant["native"])
    assert np.array_equal(redundant["suffix_even"], redundant["native"])
    assert np.array_equal(redundant["both_even"], [0, 0])
    assert np.array_equal(
        M.route_interaction(
            *(redundant[k] for k in ("native", "body_even", "suffix_even", "both_even"))
        ),
        -redundant["native"],
    )


def test_noop_exact_and_mismatch_rejected():
    """Exact noops and incompatible cut geometry are separate from scientific claims."""
    a, b, low = states()
    _, v = M.assignment(a, b, 3, "noop", low, -1, 1)
    assert np.array_equal(v, a)
    with pytest.raises(ValueError):
        M.assignment(a, b[:-1], 3, "body_even", low, -1, 1)
    with pytest.raises(ValueError):
        M.assignment(a, b, 3, "undeclared", low, -1, 1)


def test_actual_readout_metadata_interface():
    """Decode the actual native metadata dictionary, not an imaginary state object."""
    from scripts.run_grain_ownership_route_factorization import stage_values

    result = stage_values({"native": {"candidate_logits": list(range(10))}})
    assert 0 < result["score"] < 1
    assert result["log_odds"] == pytest.approx(5)
    assert np.array_equal(result["centered_logits"], np.arange(10) - 4.5)
