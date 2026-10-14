#!/usr/bin/env python3
"""Finite identities and a null that can distinguish bag from chronology."""

import itertools

import numpy as np
import pytest

from grain_research.construction import math_trait_path as F


def node_law():
    """Return a finite node observer solely for synthetic algebra tests."""
    return {"probabilities": np.arange(40).reshape(8, 5).tolist(), "prior": [0.0] * 5}


def law():
    """Construct a fully supported synthetic source law, not natural evidence."""
    rng = np.random.default_rng(172)
    j = rng.dirichlet(np.ones(64), size=12).reshape(12, 8, 8)
    y = rng.integers(0, 2, (12, 5))
    return F.compile_law(j, y, [0, 0], node_law(), "synthetic")


@pytest.mark.parametrize("p", [[0, 1, 1, 2, 3], [3, 3, 3, 3, 3], [0, 1, 2, 0]])
def test_exact_fixed_end_reference(p):
    masses = [
        F.edge_mass([p[0], *order, p[-1]]) for order in itertools.permutations(p[1:-1])
    ]
    assert np.allclose(F.fixed_end_mean(p), np.mean(masses, axis=0), atol=1e-15)


def test_same_endpoints_zero_residual_gradient():
    p = [0, 1, 2, 3, 1, 5, 7, 0]
    d = F.edge_mass(p) - F.fixed_end_mean(p)
    _, g, c = F.parts(d)
    assert np.max(np.abs(g)) < 1e-15
    assert np.max(np.abs(c.sum(1))) < 1e-15


def test_hodge_complete_and_orthogonal():
    f = np.random.default_rng(2).normal(size=(8, 8, 5))
    s, g, c = F.parts(f)
    assert np.allclose(s + g + c, f)
    assert np.max(np.abs(c.sum(1))) < 1e-14
    assert abs(np.sum(s * g)) < 1e-13
    assert abs(np.sum(g * c)) < 1e-13


def test_source_probability_law():
    b = law()
    F.verify_law(b)
    assert np.allclose(F.contract(b["reference"], b["field"]), b["prior"])


def test_account_composition():
    p = np.random.default_rng(3).integers(0, 8, 18)
    a = F.accounts(F.edge_mass(p), F.fixed_end_mean(p), law())
    assert np.allclose(a["relative"], a["bag_expected"] + a["reorder"])
    assert np.allclose(a["relative"], a["traffic"] + a["endpoint"] + a["cycle"])
    assert np.allclose(a["reorder"], a["reorder_traffic"] + a["reorder_cycle"])
    assert np.max(np.abs(a["reorder_endpoint"])) < 1e-15


def test_node_average_blind_to_cycle():
    node = np.random.default_rng(4).normal(size=(8, 5))
    field = np.repeat(node[:, None], 8, axis=1)
    _, _, cycle = F.parts(field)
    assert np.max(np.abs(cycle)) < 1e-15
    p = np.arange(18) % 8
    assert np.allclose(F.contract(F.edge_mass(p), field), node[p[:-1]].mean(0))


def test_reversal_preserves_traffic_flips_signed_parts():
    j = F.edge_mass([0, 1, 2, 0, 1, 3])
    s, g, c = F.parts(j)
    sr, gr, cr = F.parts(j.T)
    assert np.allclose(s, sr)
    assert np.allclose(g, -gr)
    assert np.allclose(c, -cr)


@pytest.mark.parametrize("p", [[0, 1], [0, 1, 2, 8], [0, 1, 2, 0.2]])
def test_invalid_path(p):
    with pytest.raises(ValueError):
        F.edge_mass(p)


def record():
    """Create opaque event identities; neither identity nor labels are features."""
    return {
        "author_key": "synthetic",
        "events": [
            {"event_id": str(i), "condition": i % 2, "coordinates": [0.2] * 15}
            for i in range(18)
        ],
    }


def test_cold_labels_rejected():
    r = record() | {"y": [1] * 5}
    with pytest.raises(ValueError):
        F.cold_author(law(), r)


def test_cold_duplicate_event_rejected():
    r = record()
    r["events"][1]["event_id"] = "0"
    with pytest.raises(ValueError):
        F.cold_author(law(), r)


def test_cold_operational():
    r = F.cold_author(law(), record())
    assert r["labels_received"] is False
    assert r["edges"] == 17
    assert r["person_posterior"] is False


def test_source_support_missing_is_error():
    with pytest.raises(ValueError):
        F.compile_law(
            np.ones((2, 8, 8)) / 64 * np.eye(8),
            np.ones((2, 5)),
            [0, 0],
            node_law(),
            "bad",
        )
