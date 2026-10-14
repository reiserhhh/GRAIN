#!/usr/bin/env python3
"""Known correlated-factor recovery and frozen out-of-sample scoring tests."""

import shutil

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from grain_research.common_factors import FrozenCommonFactors, WordFactorReference


def test_native_common_factors_recover_correlated_nonisotropic_model():
    """Allow correlated factors and unequal uniqueness, rather than assuming PCA."""
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript unavailable")
    rng = np.random.default_rng(902)
    phi = np.array([[1, 0.5, 0.25], [0.5, 1, 0.35], [0.25, 0.35, 1]])
    truth = rng.multivariate_normal(np.zeros(3), phi, 1500)
    loadings = np.zeros((18, 3))
    for i in range(18):
        loadings[i, i // 6] = 0.8
    x = truth @ loadings.T + rng.normal(size=(1500, 18)) * np.linspace(0.3, 0.7, 18)
    model = FrozenCommonFactors(3).fit(x[:1100])
    inferred = model.transform(x[1100:])
    corr = np.corrcoef(truth[1100:].T, inferred.T)[:3, 3:]
    rows, cols = linear_sum_assignment(-abs(corr))
    signs = np.sign(corr[rows, cols])
    aligned_phi = model.phi_[np.ix_(cols, cols)] * signs[:, None] * signs[None, :]
    assert abs(corr[rows, cols]).min() > 0.95
    np.testing.assert_allclose(aligned_phi, phi, atol=0.08)
    assert model.admissible_ and model.diagnostics_["off_diagonal_rms"] < 0.03
    assert np.ptp(model.uniquenesses_) > 0.15
    np.testing.assert_allclose(
        model.transform(x[1100:1101])[0], inferred[0], atol=1e-12
    )
    np.testing.assert_allclose(
        model.transform(np.vstack([x[1100], x[1101] + 10000]))[0],
        inferred[0],
        atol=1e-12,
    )


def test_word_reference_excludes_unseen_vocabulary():
    docs = [
        "music melody garden",
        "music melody ocean",
        "market finance stocks",
        "market finance bank",
        "garden flowers nature",
        "ocean waves nature",
        "books reading library",
        "books library finance",
    ] * 4
    model = WordFactorReference(method="pca", n_factors=3).fit(docs)
    single = model.transform(["music melody neverobserved"])[0]
    batch = model.transform(["music melody neverobserved", "bank stocks"])[0]
    assert "neverobserved" not in model.vectorizer_.vocabulary_
    np.testing.assert_allclose(single, batch)
    np.testing.assert_allclose(np.cov(model.transform(docs).T), np.eye(3), atol=1e-9)


def test_invalid_common_factor_data_fails_before_r():
    with pytest.raises(ValueError, match="Constant"):
        FrozenCommonFactors(2).fit(np.ones((10, 5)))
    with pytest.raises(ValueError, match="finite"):
        FrozenCommonFactors(2).fit(np.array([[np.nan, 1], [2, 3]]))
