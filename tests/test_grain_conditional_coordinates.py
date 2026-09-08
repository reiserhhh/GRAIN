#!/usr/bin/env python3
"""Synthetic identities and held-reference behavior for M23 localization."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.conditional_coordinates import (
    FAMILIES,
    CoordinateReader,
    state_features,
)
from grain_research.vector_mechanisms.nonlinear_readout import grouped_folds


def states(seed=7):
    """Create reproducible paired states, not synthetic evidence of human traits."""
    rng = np.random.default_rng(seed)
    p, q = rng.normal(size=(50, 5, 4)), rng.normal(size=(50, 4))
    y = p[:, 0, 0] - p[:, 1, 0] + q[:, 1] + p.mean(1)[:, 2]
    return p, q, y


def test_mean_configuration_reconstruction():
    """No part is deleted by the algebraic mean/configuration split."""
    p, q, _ = states()
    m = state_features(p, q, "p_mean")
    c = state_features(p, q, "p_config").reshape(p.shape)
    assert np.allclose(c.mean(1), 0)
    assert np.allclose(m[:, None, :] + c, p)


@pytest.mark.parametrize("family", FAMILIES)
def test_frozen_affine_reader_and_batch_invariance(family):
    """All maps are affine, use training references, and commute with view means."""
    p, q, y = states()
    reader = CoordinateReader.fit(p[:35], q[:35], y[:35], family, 1, rank=12)
    assert reader.unroll_error < 1e-10
    assert np.allclose(reader.reference_p, p[:35].mean(0))
    one = reader.predict(p[35:36], q[35:36])
    many = reader.predict(
        np.concatenate([p[35:36], p[36:37] * 1e6]),
        np.concatenate([q[35:36], q[36:37] * 1e6]),
    )[:1]
    assert np.allclose(one, many)
    assert np.allclose(
        reader.predict(p[35:38], q[35:38]).mean(),
        reader.predict(p[35:38].mean(0)[None], q[35:38].mean(0)[None]),
    )
    with pytest.raises(ValueError):
        reader.predict(p[35:36, :-1], q[35:36])


def test_full_prediction_and_token_accounting():
    """Signed mean/configuration/Q parts sum exactly, even when correlated."""
    p, q, y = states()
    reader = CoordinateReader.fit(p[:35], q[:35], y[:35], "p_full_q", 1, 12)
    parts, token, config = reader.full_parts(p[35:], q[35:])
    assert np.allclose(sum(parts.values()), reader.predict(p[35:], q[35:]))
    assert np.allclose(token.sum(1), parts["p_mean"] + parts["p_config"])
    assert np.allclose(config.sum(1), parts["p_config"])
    ref, _, _ = reader.full_parts(reader.reference_p[None], reader.reference_q[None])
    assert all(np.allclose(ref[k], 0) for k in ("p_mean", "p_config", "q"))


def test_views_grouped_and_invalid_states_rejected():
    """Author views never span folds; nonfinite or mismatched states fail."""
    names = np.repeat([f"u{i}" for i in range(10)], 3)
    folds = grouped_folds(names)
    assert all(len(set(folds[names == name])) == 1 for name in set(names))
    p, q, _ = states()
    with pytest.raises(ValueError):
        state_features(p, q[:-1], "p_mean")
    with pytest.raises(ValueError):
        state_features(p * np.nan, q, "p_mean")


def test_observer_compatibility_ignores_only_collection_time():
    """A different layer/template/dtype must not inherit a timestamp exception."""
    from scripts.analyze_grain_conditional_edits import compatible_observers

    a = {"unix_time": 1, "dtype": "float32", "template": "a"}
    assert compatible_observers(a, {**a, "unix_time": 2})
    assert not compatible_observers(a, {**a, "dtype": "float16"})
    assert not compatible_observers(a, {**a, "template": "b"})


def test_response_metrics_do_not_hide_amplitude():
    """Perfect ordering with tiny amplitude is not perfect response reconstruction."""
    from scripts.analyze_grain_conditional_edits import paired_metrics

    y = np.arange(-5, 5, dtype=float)
    m = paired_metrics(y, 0.01 * y, draws=30)
    assert m["pearson_r"] == pytest.approx(1)
    assert m["mse_over_zero"] == pytest.approx(0.99**2)
