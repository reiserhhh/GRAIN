#!/usr/bin/env python3
"""Independent score and paired-uncertainty tests for original-dimension contrasts."""

import copy

import numpy as np
import pytest
from scipy.stats import pearsonr

from scripts.analyze_grain_projection_sensitivity import paired_readouts


def packet(seed):
    """Return synthetic five-trait OOF values, not private-person observations."""
    rng = np.random.default_rng(34)
    y = rng.normal(size=(100, 5))
    return {
        "manifest": {"task": "big5"},
        "ids": np.arange(100),
        "folds": np.repeat(range(5), 20),
        "y": y,
        "p": y + np.random.default_rng(seed).normal(size=y.shape),
    }


def test_identical_predictions_have_zero_intervals():
    a = packet(33)
    rows = paired_readouts(a, a, draws=20)
    assert len(rows) == 2
    assert all(r[k] == 0 for r in rows for k in ("delta", "lower", "upper"))


def test_estimates_reproduce_scipy_on_paired_author_folds():
    a, b = packet(33), packet(35)
    rows = paired_readouts(a, b, draws=20)

    def r(p, selected):
        return np.mean(
            [
                pearsonr(p["y"][selected, j], p["p"][selected, j]).statistic
                for j in range(5)
            ]
        )

    all_rows = np.ones(100, dtype=bool)
    assert rows[0]["delta"] == pytest.approx(r(a, all_rows) - r(b, all_rows), abs=1e-12)
    assert rows[1]["delta"] == pytest.approx(
        np.mean([r(a, a["folds"] == f) - r(b, b["folds"] == f) for f in range(5)]),
        abs=1e-12,
    )


def test_mismatched_authors_and_folds_are_rejected():
    a, b = packet(33), packet(35)
    b["ids"] = np.roll(b["ids"], 1)
    with pytest.raises(ValueError, match="paired"):
        paired_readouts(a, b, draws=20)
    b = copy.deepcopy(a)
    b["folds"] = np.roll(b["folds"], 1)
    with pytest.raises(ValueError, match="paired"):
        paired_readouts(a, b, draws=20)
