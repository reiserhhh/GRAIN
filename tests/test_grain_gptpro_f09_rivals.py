"""Declared process equivalence and parity-disjoint observations."""

import numpy as np

from grain_research.gptpro_foundation_v3.f09_rivals import (
    BETAS,
    CORNERS,
    EVEN,
    ODD,
    candidates,
    fit_family,
    predict_family,
    probability,
)


def test_parity_design_and_crossed_atoms():
    """Parity refers to condition corners, not alternating event row indices."""
    assert len(EVEN) == len(ODD) == 4
    assert set(map(tuple, EVEN)).isdisjoint(map(tuple, ODD))
    assert len(candidates()["atoms"]) == 9


def test_belief_and_audience_equivalence_at_binary_conditions():
    """Process labels cannot be distinguished from exact observed equivalents."""
    for beta in BETAS:
        np.testing.assert_allclose(
            probability(CORNERS, "belief", beta),
            probability(CORNERS, "negative_M1", beta),
            atol=1e-14,
        )
        np.testing.assert_allclose(
            probability(CORNERS, "audience", beta),
            probability(CORNERS, "M2", beta),
            atol=1e-14,
        )
        middle = np.array([[1 / 3, 1, -1], [-1 / 3, 1, 1]])
        assert (
            np.max(
                np.abs(
                    probability(middle, "belief", beta)
                    - probability(middle, "negative_M1", beta)
                )
            )
            > 1e-5
        )


def test_class_prior_not_atom_prior():
    """Duplicate descriptions must not get twice the prior mass of unique laws."""
    spec = candidates(True)
    assert len(spec["atoms"]) == 18 and len(spec["class_keys"]) == 12
    np.testing.assert_allclose(
        np.bincount(spec["classes"], weights=spec["prior"]), np.full(12, 1 / 12)
    )


def test_posterior_is_direct_enumerated_bernoulli_product():
    """A separate product likelihood recovers the stable log-space posterior."""
    z = np.repeat(EVEN, 3, axis=0)
    y = np.array([0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1])
    fitted = fit_family(z, y, True)
    masses = []
    for (family, beta), prior in zip(fitted["atoms"], fitted["prior"], strict=True):
        p = probability(z, family, beta)
        masses.append(prior * np.prod(np.where(y, p, 1 - p)))
    np.testing.assert_allclose(fitted["posterior"], np.array(masses) / sum(masses))
    assert np.all((predict_family(fitted, ODD) > 0) & (predict_family(fitted, ODD) < 1))
