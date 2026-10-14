"""Cache recovery controls: source isolation and pair-preserving resampling."""

import numpy as np

from scripts.run_grain_gptpro_f10_uncertainty import direct_fit, resample


def test_source_only_fit_ignores_application_targets():
    rng = np.random.default_rng(13)
    x, y = rng.normal(size=(40, 4)), rng.normal(size=(40, 2))
    source = np.arange(24)
    before = direct_fit(x, y, source)
    y[24:] += 1000
    np.testing.assert_array_equal(before, direct_fit(x, y, source))


def test_ridge_augmented_solution_matches_normal_equations():
    rng = np.random.default_rng(11)
    x, y = rng.normal(size=(36, 4)), rng.normal(size=(36, 2))
    scaled = x / x.std(axis=0)
    beta = np.linalg.solve(scaled.T @ scaled + np.eye(4), scaled.T @ y)
    p = scaled @ beta
    grid = np.array([0, 0.25, 0.5, 0.75, 1.0])
    loss = np.mean((grid[:, None, None] * p - y) ** 2, axis=1)
    np.testing.assert_allclose(
        direct_fit(x, y, np.arange(36)), p * grid[np.argmin(loss, axis=0)], atol=1e-12
    )


def test_resampling_cannot_cross_roles():
    pairs = [{"situation": str(i % 3)} for i in range(20)]
    allowed = np.arange(12)
    for unit in ("pair", "task_then_pair"):
        sample = resample(np.random.default_rng(1), allowed, pairs, unit)
        assert set(sample).issubset(set(allowed))
        assert len(sample) == 12
