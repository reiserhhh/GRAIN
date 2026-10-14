#!/usr/bin/env python3
"""Synthetic checks for source-standardized conditional event measurements."""

import numpy as np
import pytest

from grain_research.construction import math_event_target_conditioning as M
from grain_research.construction import math_target_evidence_routing as T


def laws(weights, rates):
    """Create a complete joint law with a chosen first-target conditional response."""
    p = np.empty((len(weights), 32))
    low, high = T.edges(0)
    p[:, low] = np.asarray(weights) * (1 - np.asarray(rates))
    p[:, high] = np.asarray(weights) * np.asarray(rates)
    return p


def sample():
    """Return positive synthetic paired laws and a SOURCE-only reference."""
    rng = np.random.default_rng(268)
    p = rng.dirichlet(np.ones(32), 6).reshape(3, 2, 32)
    ref = M.reference(np.repeat(np.arange(32), np.arange(1, 33)))
    return p, ref


def test_reconstruction_and_old_interaction():
    """Check the signed whole, including symmetric allocation of old interaction."""
    p, ref = sample()
    d = M.decompose(p, ref["omega"])
    v, old = d["values"], d["one_sided_accounts"]
    np.testing.assert_allclose(v[..., 0], v[..., 1:].sum(-1), atol=1e-14)
    np.testing.assert_allclose(v[..., 1] + v[..., 2], old[..., 0] + old[..., 2] / 2)
    np.testing.assert_allclose(v[..., 3], old[..., 1] + old[..., 2] / 2)


def test_swap_is_antisymmetric():
    """Swapping addressed endpoints reverses every signed measurement part."""
    p, ref = sample()
    a, b = M.decompose(p, ref["omega"]), M.decompose(p[:, ::-1], ref["omega"])
    np.testing.assert_allclose(a["values"], -b["values"], atol=1e-14)


def test_routing_alone_can_move_target():
    """Demonstrate a marginal target change with zero conditional target evidence."""
    w = np.stack((np.arange(1, 17) / 136, np.arange(16, 0, -1) / 136))
    r = np.tile(np.linspace(0.1, 0.9, 16), (2, 1))
    d = M.decompose(laws(w, r)[None], np.full((5, 16), 1 / 16))["values"][0, 0]
    assert abs(d[0]) > 0.1
    np.testing.assert_allclose(d[1:3], 0, atol=1e-14)
    np.testing.assert_allclose(d[0], d[3])


def test_conditional_only():
    """A conditional change at common fixed profile weights has no routing term."""
    w, r = np.full((2, 16), 1 / 16), np.array([[0.7] * 16, [0.3] * 16])
    d = M.decompose(laws(w, r)[None], np.full((5, 16), 1 / 16))["values"][0, 0]
    np.testing.assert_allclose(d, [0.4, 0.4, 0, 0], atol=1e-14)


def test_common_reference_is_source_only():
    """Retain exact profile counts and invariant author-order weighting."""
    _, ref = sample()
    codes = np.repeat(np.arange(32), np.arange(1, 33))
    other = M.reference(codes[::-1])
    np.testing.assert_array_equal(ref["profile_counts"], np.arange(1, 33))
    np.testing.assert_allclose(ref["omega"], other["omega"])
    for q in range(5):
        n = ref["counts"][q]
        omega = n.prod(0) / n.sum(0)
        np.testing.assert_allclose(ref["omega"][q], omega / omega.sum())


@pytest.mark.parametrize("case", ["zero", "unnormalized", "negative_reference"])
def test_invalid_laws_rejected(case):
    """Missing or invalid observation laws are not imputed or smoothed."""
    p, ref = sample()
    o = ref["omega"].copy()
    if case == "zero":
        p[0, 0, 0] = 0
    elif case == "unnormalized":
        p[0, 0] *= 2
    else:
        o[0, 0] = -1
    with pytest.raises(ValueError):
        M.decompose(p, o)


def test_missing_source_profile_rejected():
    """Unsupported source strata cannot be interpreted as neutral evidence."""
    with pytest.raises(ValueError):
        M.reference(np.arange(31))


@pytest.mark.parametrize("extra", ["label", "answer", "questionnaire"])
def test_cold_rejects_outcomes(extra):
    """Current author labels cannot enter the numeric observation contract."""
    with pytest.raises(ValueError):
        M.cold_record({}, {"frame_key": "x", "before": {}, "after": {}, extra: 1})


def test_cold_rejects_repeated_evidence():
    """Duplicating an event must not manufacture a change observation."""
    e = {"event_id": "same", "vector": [1.0, 0.0]}
    with pytest.raises(ValueError):
        M.cold_record({}, {"frame_key": "x", "before": e, "after": e})
