"""Synthetic checks for information-preserving coordinate/readout changes."""

import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from grain_research.vector_mechanisms.external_readout_transport import (
    collapse_dense_pipeline,
    landmark_radial_lift,
    pullback_readout,
    transport_readout,
)
from suica_core.v8_bridge import canonical_orbit_distance_signatures


@pytest.mark.parametrize("classification", [False, True])
@pytest.mark.parametrize("pca", [False, True])
def test_exact_fitted_readout_transport_including_pca(classification, pca):
    rng = np.random.default_rng(386)
    x = rng.normal(size=(90, 9)) + 2
    y = (x[:, 0] > 2).astype(int) if classification else rng.normal(size=(90, 5))
    model = Pipeline(
        [("scale", StandardScaler())]
        + ([("pca", PCA(4))] if pca else [])
        + [("head", LogisticRegression() if classification else Ridge())]
    ).fit(x[:60], y[:60])
    u, _, v = np.linalg.svd(rng.normal(size=(9, 9)))
    matrix = u @ np.diag(np.linspace(0.2, 3, 9)) @ v
    offset = rng.normal(size=9)
    readout = collapse_dense_pipeline(model)
    transported = transport_readout(readout, matrix, offset)
    expected = (
        model.predict_proba(x[60:])[:, 1] if classification else model.predict(x[60:])
    )
    np.testing.assert_allclose(readout.predict(x[60:]), expected, atol=2e-14)
    np.testing.assert_allclose(
        transported.predict(x[60:] @ matrix + offset), expected, atol=2e-14
    )
    with pytest.raises(ValueError, match="invertible"):
        transport_readout(readout, np.zeros((9, 9)), offset)


def test_pca_whitening_and_single_output_regression():
    rng = np.random.default_rng(332)
    x, y = rng.normal(size=(50, 8)), rng.normal(size=50)
    model = Pipeline(
        [
            ("scale", StandardScaler(with_mean=False)),
            ("pca", PCA(3, whiten=True)),
            ("head", Ridge()),
        ]
    ).fit(x, y)
    np.testing.assert_allclose(
        collapse_dense_pipeline(model).predict(x), model.predict(x)
    )


def test_radial_lift_is_total_not_perpendicular_radius():
    rng = np.random.default_rng(691)
    q, l = rng.normal(size=(70, 12)), rng.normal(size=(6, 12))
    d, _, _ = canonical_orbit_distance_signatures(q, l)
    lifted, matrix, offset = landmark_radial_lift(q, l)
    np.testing.assert_allclose(lifted @ matrix + offset, d**2, atol=1e-12)
    assert np.linalg.matrix_rank(matrix) == 6
    wrong = lifted.copy()
    wrong[:, -1] -= np.sum(wrong[:, :-1] ** 2, axis=1)
    # Omitting this quadratic term is the previous documentation error.
    assert np.max(abs(wrong @ matrix + offset - d**2)) > 1
    y = rng.normal(size=(70, 5))
    model = Pipeline([("scale", StandardScaler()), ("head", Ridge())]).fit(
        d[:50] ** 2, y[:50]
    )
    moved = pullback_readout(collapse_dense_pipeline(model), matrix, offset)
    np.testing.assert_allclose(
        moved.predict(lifted[50:]), model.predict(d[50:] ** 2), atol=1e-12
    )


def test_generalized_ridge_penalty_transports_but_isotropic_penalty_does_not():
    rng = np.random.default_rng(555)
    x, y = rng.normal(size=(100, 5)), rng.normal(size=100)
    x, y = x - x.mean(0), y - y.mean()
    matrix = np.diag([0.1, 0.4, 1, 3, 10])
    q, alpha = x @ matrix, 30
    original = np.linalg.solve(x.T @ x + alpha * np.eye(5), x.T @ y)
    transported = np.linalg.solve(q.T @ q + alpha * matrix.T @ matrix, q.T @ y)
    naive = np.linalg.solve(q.T @ q + alpha * np.eye(5), q.T @ y)
    np.testing.assert_allclose(q @ transported, x @ original, atol=1e-12)
    assert np.max(abs(q @ naive - x @ original)) > 1e-2


def test_unsupported_dense_estimator_fails_closed():
    with pytest.raises(ValueError, match="Unsupported dense pipeline"):
        collapse_dense_pipeline(Pipeline([("head", Ridge())]))
