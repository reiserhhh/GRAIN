#!/usr/bin/env python3
"""Identification, likelihood and event-composition tests for HM142."""

import itertools

import numpy as np
import pytest

from grain_research.construction import math_multiview_law as M


def fixture():
    """Independent positive fixture, not a research trial."""
    p = np.array([0.2, 0.35, 0.45])
    matrices = (
        np.array([[0.72, 0.18, 0.13], [0.17, 0.67, 0.19], [0.11, 0.15, 0.68]]),
        np.array([[0.65, 0.14, 0.21], [0.22, 0.72, 0.15], [0.13, 0.14, 0.64]]),
        np.array([[0.69, 0.21, 0.12], [0.20, 0.63, 0.22], [0.11, 0.16, 0.66]]),
    )
    return p, matrices


def aligned_error(fit, truth):
    """Oracle permutation matching is test evaluation only, never in fitting."""
    return min(
        max(np.max(abs(a[:, perm] - b)) for a, b in zip(fit, truth, strict=True))
        for perm in itertools.permutations(range(3))
    )


def test_unknown_law_exact_recovery():
    p, matrices = fixture()
    fit = M.fit_law(M.joint_law(p, matrices), 3, [-0.7, 0.2, 1.1])
    assert aligned_error(fit["emissions"], matrices) < 1e-12
    assert fit["reconstruction_TV"] < 1e-12
    assert fit["off_diagonal_max"] < 1e-12


def test_all_pairwise_moments_alias_but_triples_differ():
    p, matrices = fixture()
    q, changed, _ = M.pairwise_alias(p, matrices, 0.12)
    t, s = M.joint_law(p, matrices), M.joint_law(q, changed)
    for axis in range(3):
        assert np.max(abs(t.sum(axis=axis) - s.sum(axis=axis))) < 1e-14
    assert abs(t - s).sum() / 2 > 0.005
    assert aligned_error(matrices, changed) > 0.01


@pytest.mark.parametrize("view", [0, 1, 2])
def test_uninformative_rank_is_not_rescued(view):
    p, original = fixture()
    matrices = list(original)
    matrices[view] = np.full((3, 3), 1 / 3)
    with pytest.raises(M.IdentificationError):
        M.fit_law(M.joint_law(p, matrices), 3, [-0.7, 0.2, 1.1])


def test_repeated_contraction_not_parameter_search():
    p, matrices = fixture()
    with pytest.raises(M.IdentificationError, match="separate"):
        M.fit_law(M.joint_law(p, matrices), 3, [1, 1, 1])


def test_permuting_hidden_labels_does_not_change_observed_fit():
    p, matrices = fixture()
    perm = [2, 0, 1]
    original = M.fit_law(M.joint_law(p, matrices), 3, [-0.7, 0.2, 1.1])
    reordered = M.fit_law(
        M.joint_law(p[perm], [a[:, perm] for a in matrices]), 3, [-0.7, 0.2, 1.1]
    )
    assert np.max(abs(original["weights"] - reordered["weights"])) < 1e-12


def test_cold_posterior_is_direct_bayes_enumeration():
    p, matrices = fixture()
    categories = np.array(list(itertools.product(range(3), repeat=3)))
    scores = M.posterior(categories, p, matrices)
    mass = np.array(
        [p * matrices[0][a] * matrices[1][b] * matrices[2][c] for a, b, c in categories]
    )
    assert np.max(abs(scores - mass / mass.sum(axis=1, keepdims=True))) < 1e-14


def test_event_repeat_does_not_strengthen_score():
    p, matrices = fixture()
    rows = [
        {"event": str(v), "view": v, "category": v, "qualification": "self_actual"}
        for v in range(3)
    ]
    one, repeated = (
        M.qualified_posterior(rows, p, matrices),
        M.qualified_posterior(rows * 5, p, matrices),
    )
    assert np.array_equal(one["posterior"], repeated["posterior"])
    assert repeated["unique_events"] == 3 and repeated["repeated_mentions"] == 12


@pytest.mark.parametrize(
    "change",
    [{"qualification": "unknown"}, {"truth": 1}, {"author_id": "x"}, {"category": 3}],
)
def test_unqualified_or_leaking_record_rejected(change):
    p, matrices = fixture()
    row = {
        "event": "e",
        "view": 0,
        "category": 1,
        "qualification": "self_actual",
        **change,
    }
    with pytest.raises(ValueError):
        M.qualified_posterior([row], p, matrices)


def test_same_event_copies_do_not_form_three_views():
    with pytest.raises(ValueError):
        M.empirical_tensor(np.array([[0, 1, 2]]), (3, 3, 3), ["e", "e", "e"])


def test_same_event_reencoding_requires_explicit_operation():
    p, matrices = fixture()
    rows = [
        {"event": "e", "view": v, "category": 0, "qualification": "self_actual"}
        for v in (0, 1)
    ]
    with pytest.raises(ValueError):
        M.qualified_posterior(rows, p, matrices)


def test_empty_evidence_returns_prior_not_zero_trait():
    p, matrices = fixture()
    result = M.qualified_posterior([], p, matrices)
    assert np.allclose(result["posterior"], p)
    assert result["unique_events"] == 0


def test_vector_alphabet_roundtrip_and_coordinate_rotation():
    rng = np.random.default_rng(42)
    alphabet = rng.normal(size=(3, 3, 15))
    x = np.array([[0, 2, 1], [1, 0, 2]])
    vectors = np.stack([alphabet[v, x[:, v]] for v in range(3)], axis=1)
    assert np.array_equal(M.decode_vectors(vectors, alphabet), x)
    q = np.linalg.qr(rng.normal(size=(15, 15)))[0]
    assert np.array_equal(M.decode_vectors(vectors @ q, alphabet @ q), x)
