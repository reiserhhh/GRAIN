#!/usr/bin/env python3
"""Check path heads against frozen scalar online heads on known paths."""

import numpy as np

from grain_research.gptpro_foundation_v3.f08_estimators import QualifiedHistory
from grain_research.gptpro_foundation_v3.f13_path_heads import (
    path_head_scores,
    summarize_path_scores,
)


def test_hard_history_matches_frozen_head():
    """No missing history means sampled-path fitting equals the original head."""
    model = {
        "b1": {"source_p": [0.2, 0.8, 0.7, 0.3]},
        "b2": {"coef": [0.2, 0.1, 0.3, -0.1]},
        "m": {"mean": [0.2, 0.8, 0.7, 0.3], "right": [[1, 0, 0, 0], [0, 1, 0, 0]]},
    }
    paths = np.random.default_rng(45).integers(0, 2, (20, 24))
    scores = path_head_scores(paths, np.ones(24), model)
    for i, path in enumerate(paths):
        history = QualifiedHistory(model, mode="independent")
        for y in path:
            history.observe(float(y), 1)
        original = history.predict()["raw_probabilities"]
        for head in ("b2", "m"):
            np.testing.assert_allclose(scores[head][i], original[head], atol=1e-12)


def test_zero_history_variance_not_zero_outcome_variance():
    """Known reconstructed history cannot establish known future outcome."""
    value = summarize_path_scores(np.full(200, 0.3))
    np.testing.assert_allclose(value["history_refit_variance"], 0, atol=1e-20)
    np.testing.assert_allclose(value["event_variance"], 0.21)
