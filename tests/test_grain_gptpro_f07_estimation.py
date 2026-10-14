"""Independent finite differences and missing-channel boundary tests."""

import numpy as np
import pytest

from grain_research.gptpro_foundation_v3.f07_estimation import (
    PRIOR_MEAN,
    PRIOR_VARIANCE,
    fit_channel,
    objective_gradient,
)


@pytest.mark.parametrize("variant", ["base", "offtarget", "mnar", "ordinal"])
def test_objective_derivative(variant):
    """Likelihood derivatives agree with two independent central-difference steps."""
    ordinal = variant == "ordinal"
    counts = (
        np.array([[2, 5, 3, 4, 1], [4, 1, 3, 2, 5]])
        if ordinal
        else np.array([[7, 5, 3], [4, 2, 9]])
    )
    args = (counts, 0.6, 0.12, "base" if ordinal else variant, ordinal)
    theta = np.array([-0.4, 0.8])
    _, gradient = objective_gradient(theta, *args)
    for step in (1e-4, 1e-5):
        numerical = np.array(
            [
                (
                    objective_gradient(theta + step * unit, *args)[0]
                    - objective_gradient(theta - step * unit, *args)[0]
                )
                / (2 * step)
                for unit in np.eye(2)
            ]
        )
        np.testing.assert_allclose(gradient, numerical, atol=1e-7, rtol=1e-7)


def test_no_observation_returns_prior():
    """An entirely absent, parameter-independent channel gives no personal data."""
    fit = fit_channel(np.array([[16, 0, 0], [16, 0, 0]]), 0, 0.1)
    np.testing.assert_allclose(fit["estimate"], PRIOR_MEAN)
    np.testing.assert_allclose(fit["covariance"], np.diag(PRIOR_VARIANCE))
    assert fit["success"]


def test_mnar_absence_is_evidence():
    """Missingness may update a under MNAR, unlike the exogenous channel."""
    counts = np.array([[16, 0, 0], [16, 0, 0]])
    independent = fit_channel(counts, 0.6, 0.1)
    dependent = fit_channel(counts, 0.6, 0.1, "mnar")
    np.testing.assert_allclose(independent["estimate"], PRIOR_MEAN)
    assert dependent["estimate"][0] < -0.5


def test_bad_counts_rejected():
    """Malformed or negative categorical counts cannot silently reach fitting."""
    with pytest.raises(ValueError):
        fit_channel(np.array([[1, -1, 0], [2, 1, 0]]), 0.6, 0.1)
