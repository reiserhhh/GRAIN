"""Identifiability can fail or succeed depending on the observation design."""

import numpy as np
import pytest

from grain_research.construction import math_action_observability as M


def test_exact_orthogonal_operators_identical_observations():
    """Deleting half a coupled action is not proof of two identifiable mechanisms."""
    a = np.array([[0.0, 1.0], [0.0, 0.0]])
    s, h = (a + a.T) / 2, (a - a.T) / 2
    assert np.sum(s * h) == 0
    x, c = np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]])
    v = np.array([[0.0, (x @ s @ c.T).item(), (x @ h @ c.T).item(), 0.0]])
    stats = M.action_diagnostics(v, [1.0])
    assert stats["sh_cosine"] == 1
    assert stats["sh_rank"] == 1
    assert stats["numerical_null_dimension"] == 1
    assert stats["sh_correlation"] is None


def test_crossed_design_separates_same_operators():
    v = np.array([[0, 0.5, 0.5, 0], [0, 0.5, -0.5, 0]], dtype=float)
    stats = M.action_diagnostics(v, [0.5, 0.5])
    assert stats["sh_rank"] == 2
    assert stats["sh_cosine"] == 0
    assert stats["sh_condition_number"] == 1


def test_near_collinear_is_not_exact_null():
    v = np.array([[0, 0.5, 0.5001, 0], [0, 0.5, 0.4999, 0]], dtype=float)
    stats = M.action_diagnostics(v, [0.5, 0.5])
    assert stats["sh_rank"] == 2
    assert stats["sh_condition_number"] > 1e6


def test_dyad_identity_and_weighted_psd():
    rng = np.random.default_rng(18)
    x, c = rng.normal(size=(7, 4)), rng.normal(size=(7, 4))
    d = np.einsum("ij,ik->ijk", x, c)
    for part, feature in {
        "A": d,
        "S": (d + d.transpose(0, 2, 1)) / 2,
        "H": (d - d.transpose(0, 2, 1)) / 2,
    }.items():
        explicit = feature.reshape(7, -1)
        g = M.dyad_grams(x, c)[part]
        np.testing.assert_allclose(g, explicit @ explicit.T, atol=1e-12)
        M.spectrum(M.weighted_design_gram(g, np.arange(1, 8) / 28))


def test_orthogonal_gauge_preserves_design_grams():
    rng = np.random.default_rng(181)
    x, c = rng.normal(size=(8, 4)), rng.normal(size=(8, 4))
    q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    for part, gram in M.dyad_grams(x, c).items():
        np.testing.assert_allclose(gram, M.dyad_grams(x @ q, c @ q)[part], atol=1e-12)


def test_zero_design_not_perfect_sharing():
    stats = M.action_diagnostics(np.zeros((3, 4)), [1 / 3] * 3)
    assert stats["sh_rank"] == 0
    assert stats["sh_cosine"] is None
    assert stats["sh_condition_number"] is None
    assert stats["transpose_direct_ratio"] is None


def test_distinct_design_measures_change_observed_sharing():
    values = np.array([[1, 1], [1, -1]], dtype=float)
    equal = M.moments(values, [0.5, 0.5])["gram"]
    unequal = M.moments(values, [0.9, 0.1])["gram"]
    assert equal[0, 1] == 0
    assert unequal[0, 1] == pytest.approx(0.8)


def test_family_mass_balanced():
    w = M.family_weights([1, 1, 1, 2])
    assert w[:3].sum() == pytest.approx(0.5)
    assert w[3] == 0.5


@pytest.mark.parametrize("weights", ([1, 1], [1.2, -0.2], [np.nan, 0]))
def test_invalid_measure_rejected(weights):
    with pytest.raises(ValueError):
        M.moments(np.ones((2, 2)), weights)


def test_indefinite_matrix_rejected():
    with pytest.raises(ValueError, match="positive semidefinite"):
        M.spectrum(np.array([[1, 2], [2, 1]]))
