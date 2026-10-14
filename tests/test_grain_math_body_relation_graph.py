#!/usr/bin/env python3
"""Mechanism and counterexample checks, not personality accuracy tests."""

import itertools

import numpy as np
import pytest

from grain_research.construction import math_body_relation_graph as M


def test_harmonic_inverse_and_root_equivariance():
    rng = np.random.default_rng(89)
    t = np.tril(rng.random((9, 9)), -1)
    t[1:] /= t[1:].sum(1, keepdims=True)
    labels = np.array([2, 0, -1, 1, -1, -1, 0, -1, -1])
    h = M.boundary_law(t, labels)
    interior, boundary = np.flatnonzero(labels < 0), np.flatnonzero(labels >= 0)
    expected = np.linalg.solve(
        np.eye(len(interior)) - t[np.ix_(interior, interior)],
        t[np.ix_(interior, boundary)] @ h[boundary],
    )
    assert np.allclose(h[interior], expected)
    assert np.allclose(h.sum(1), 1)
    swap = np.where(labels == 0, 1, np.where(labels == 1, 0, labels))
    assert np.allclose(M.boundary_law(t, swap), h[:, [1, 0, 2]])


def test_untyped_graph_cannot_determine_role_and_typed_example():
    t = np.zeros((4, 4))
    t[3, 1:3] = 0.5
    labels = np.array([2, 0, 1, -1])
    h = M.boundary_law(t, labels)
    assert np.allclose(h[3], [0.5, 0.5, 0])
    typed = t.copy()
    typed[3, 1:3] = [0, 1]
    assert np.allclose(M.boundary_law(typed, labels)[3], [0, 1, 0])
    # A perfect solve of an untyped graph still cannot choose the correct role.
    assert h[3, 0] == h[3, 1]


def test_direct_mass_preserves_unknown_and_path_effect():
    t = np.zeros((5, 5))
    t[3, 1] = 1
    t[4, 3] = 1
    labels = np.array([2, 0, 1, -1, -1])
    assert np.allclose(M.direct_law(t, labels)[4], [0, 0, 1])
    assert np.allclose(M.boundary_law(t, labels)[4], [1, 0, 0])


def test_graph_scope_and_orthogonal_frame():
    rng = np.random.default_rng(6)
    q, k = rng.normal(size=(8, 4, 6)), rng.normal(size=(8, 2, 6))
    ids = np.repeat([0, 1], 4)
    t, _ = M.causal_graph(q, k, ids)
    assert np.allclose(t[4], 0)
    assert np.allclose(t[4:, :4], 0)
    rotation, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    same, _ = M.causal_graph(q @ rotation, k @ rotation, ids)
    assert np.allclose(t, same)


def test_event_accounts_and_gauge():
    rng = np.random.default_rng(2)
    logits = rng.normal(size=(3, 7))
    v = rng.normal(size=(7, 3, 5))
    ids = np.array([0, 0, 0, 1, 1, 2, 2])
    gate = np.array([1, 0.2, 0, 0.4, 0.1, 0.3, 0])
    r = M.gated_reading(logits, v, gate, ids)
    assert np.allclose(
        r["reading"], np.einsum("hg,hgd->hd", r["event_mass"], r["event_read"])
    )
    assert np.allclose(
        r["reading"], M.gated_reading(logits + 100, v, gate, ids)["reading"]
    )


def test_author_law_marginals():
    x = np.array(list(itertools.product((-1, 1), repeat=4)))
    a, b = M.author_laws(x)
    assert np.allclose(a @ x, b @ x)
    assert np.isclose(a.sum(), 1) and np.isclose(b.sum(), 1)
    assert (a - b) @ (x[:, 0] * x[:, 2]) > 0


@pytest.mark.parametrize("labels", [[2, 0, -1], [2, 1, -1], [2, 3, 0]])
def test_missing_or_invalid_boundary(labels):
    with pytest.raises(ValueError):
        M.boundary_law(np.zeros((3, 3)), np.array(labels))


def test_no_evidence_is_not_zero():
    with pytest.raises(ValueError):
        M.gated_reading(
            np.zeros((1, 3)), np.zeros((3, 1, 2)), np.zeros(3), np.arange(3)
        )
