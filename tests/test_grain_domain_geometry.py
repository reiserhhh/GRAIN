#!/usr/bin/env python3
"""Synthetic shared-window and length-controlled association tests."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.domain_geometry import (
    association,
    association_bootstrap,
    shared_windows,
)


class Tokens:
    """Simple count-only tokenizer of varying granularity."""

    def __init__(self, multiplier):
        self.multiplier = multiplier

    def __call__(self, text, **kwargs):
        return {"input_ids": [1] * (len(text) * self.multiplier + 2)}


def test_shared_windows_are_complete_and_fit_all_observers():
    text = "a b c " * 300
    tokenizers = {"a": (Tokens(1), "q: "), "b": (Tokens(2), "")}
    spans = shared_windows(text, tokenizers)
    assert "".join(text[s["start"] : s["end"]] for s in spans) == text
    assert all(max(s["tokens"].values()) <= 512 for s in spans)
    assert spans[0]["start"] == 0 and spans[-1]["end"] == len(text)


def test_controlled_length_removes_common_trend():
    z = np.arange(10, dtype=float)
    residual = np.array([1.0, -1, 0, 0, 0, 0, 0, 0, -1, 1])
    a = association(100 * z + residual, 100 * z - residual, z)
    assert a[0] > 0.99 and a[2] < -0.999
    result = association_bootstrap(z, z, np.ones(10))
    assert result["pearson"] > 0.999 and result["partial_length"] > 0.999
    assert result["pearson_undefined_draws"] == 0


def test_undefined_is_not_zero_evidence():
    a = association(np.ones(6), np.arange(6), np.arange(6))
    assert np.isnan(a).all()
    with pytest.raises(ValueError):
        association([1, 2], [1, 2], [1, 2])


def test_independent_correlation_algebra_matches_ols():
    from scripts.audit_grain_domain_geometry import independent_associations

    rng = np.random.default_rng(4)
    x, y, z = rng.normal(size=(3, 31))
    y = y + x + 0.7 * z
    np.testing.assert_allclose(
        independent_associations(x, y, z), association(x, y, z), atol=1e-12
    )
    np.testing.assert_allclose(
        independent_associations(x, y, np.ones(31)),
        association(x, y, np.ones(31)),
        atol=1e-12,
    )


def test_independent_bootstrap_matches_frozen_runner():
    import pandas as pd

    from scripts.audit_grain_domain_geometry import replay_aggregate

    rng = np.random.default_rng(51)
    x, y, z = rng.normal(size=(3, 16))
    frame = pd.DataFrame(
        {
            "model": "synthetic",
            "corpus": "test",
            "language": "en",
            "delta_cosine": x,
            "delta_tail_logp": y,
            "delta_log_context_length": z,
            "within_cell_percentile": rng.random(16),
        }
    )
    replay = replay_aggregate(frame).iloc[0]
    expected = association_bootstrap(x, y, z)
    for key, value in expected.items():
        assert replay[key] == pytest.approx(value, abs=1e-10)
