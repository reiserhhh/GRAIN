#!/usr/bin/env python3
"""Counterexamples and identities for channel pairing; not accuracy acceptance."""

import numpy as np
import pytest

from grain_research.construction import math_channel_boundary_transport as M


def test_harmonic_average_covariance_is_not_scalar_solve():
    t = np.zeros((2, 5, 5))
    t[0, 3, 1], t[1, 3, 2] = 1, 1
    t[0, 4, 3], t[1, 4, 1] = 1, 1
    labels = np.array([2, 0, 1, -1, -1])
    laws = M.channel_laws(t, labels)
    account = M.harmonic_collapse_account(t, laws, labels)
    assert account["identity_error"] < 1e-14
    assert np.allclose(account["mean_law"][4], [1, 0, 0])
    assert np.allclose(account["scalar_law"][4], [0.75, 0.25, 0])


def test_gate_pairing_and_cyclic_orbit():
    logits = np.zeros((2, 2))
    values = np.array([[2, -2], [-2, 2]])
    gate = np.eye(2)
    ids = np.array([0, 1])
    paired = M.measure(logits, values, gate, ids)
    mean = M.measure(logits, values, np.broadcast_to(gate.mean(0), gate.shape), ids)
    wrong = M.measure(logits, values, gate[::-1], ids)
    assert paired["score"] == 4 and mean["score"] == 0 and wrong["score"] == -4
    assert np.allclose(
        (paired["numerator"] + wrong["numerator"]) / 2, mean["numerator"]
    )
    assert M.conditioning_account(paired, mean)["identity_error"] < 1e-14
    synchronous = M.measure(logits[::-1], values[::-1], gate[::-1], ids)
    assert synchronous["score"] == paired["score"]


def test_normalized_orbit_requires_separate_remainder():
    logits = np.array([[0, 1], [2, 0]])
    c = np.array([[1, -2], [3, -1]])
    gate = np.array([[0.8, 0.1], [0.1, 0.7]])
    ids = np.array([0, 1])
    orbit = [M.measure(logits, c, np.roll(gate, i, axis=0), ids) for i in range(2)]
    mean = M.measure(logits, c, np.broadcast_to(gate.mean(0), gate.shape), ids)
    assert np.allclose(np.mean([o["mass"] for o in orbit], 0), mean["mass"])
    assert np.allclose(np.mean([o["numerator"] for o in orbit], 0), mean["numerator"])
    assert not np.isclose(np.mean([o["score"] for o in orbit]), mean["score"])
    for o in orbit:
        assert M.conditioning_account(o, mean)["identity_error"] < 1e-14
        assert np.allclose(o["event_contribution"].sum(1), o["heads"])


def test_graph_coordinate_relabel_and_gqa():
    rng = np.random.default_rng(90)
    q, k = rng.normal(size=(7, 4, 3)), rng.normal(size=(7, 2, 3))
    t, self_mass = M.channel_graphs(q, k)
    assert t.shape == (4, 7, 7) and self_mass.shape == (4, 7)
    assert np.allclose(t[:, 1:].sum(2), 1)
    assert np.allclose(np.triu(t), 0)
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    same, _ = M.channel_graphs(q @ rotation, k @ rotation)
    assert np.allclose(t, same)


def test_no_named_mass_is_not_false_zero():
    with pytest.raises(ValueError):
        M.measure(np.zeros((2, 3)), np.zeros((2, 3)), np.zeros((2, 3)), np.arange(3))
