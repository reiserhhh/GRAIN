#!/usr/bin/env python3
"""Independent metric and pairing checks for completed primary observer triads."""

import numpy as np
import pytest
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

from scripts.summarize_grain_neural_source_primary import (
    paired_triads,
    save_tables,
    score,
)


def test_output_hashes_do_not_include_filesystem_sidecars(tmp_path):
    """Portable output commitments contain analytical tables, not AppleDouble data."""
    (tmp_path / "._observer_scores.csv").write_bytes(b"filesystem metadata")
    hashes = save_tables(tmp_path, [{"value": 1}], [{"value": 2}], [{"value": 3}])
    assert set(hashes) == {
        "observer_scores.csv",
        "metrics_by_trait.csv",
        "paired_observer_comparison.csv",
    }


def test_scores_match_independent_libraries():
    """Centered-dot and tied-rank formulas reproduce standard library scores."""
    rng = np.random.default_rng(77)
    y, p = rng.normal(size=(2, 100, 5))
    assert score(y, p, "big5") == pytest.approx(
        np.mean([pearsonr(y[:, j], p[:, j]).statistic for j in range(5)]), abs=1e-12
    )
    binary, tied = np.tile([0, 1], 50), rng.integers(0, 4, size=100)
    assert score(binary, tied, "EI_cont") == pytest.approx(
        roc_auc_score(binary, tied), abs=1e-12
    )


def test_identical_predictions_have_exact_zero_paired_intervals():
    """Sharing every author draw must preserve identity, including both estimands."""
    rng = np.random.default_rng(78)
    y = rng.normal(size=(100, 5))
    p = y + rng.normal(size=y.shape)
    rows = paired_triads(y, [p, p, p], np.repeat(range(5), 20), "big5", draws=20)
    assert len(rows) == 6
    assert all(r[k] == 0 for r in rows for k in ("delta", "lower", "upper"))


def test_observed_comparisons_reproduce_fold_and_pooled_scores():
    """Fold means are recomputed, not inferred by subtracting confidence bounds."""
    rng = np.random.default_rng(79)
    y, folds = np.tile([0, 1], 250), np.repeat(range(5), 100)
    p = np.asarray([y * weight + rng.normal(size=len(y)) for weight in (0.1, 0.3, 0.6)])
    rows = paired_triads(y, p, folds, "TF_cont", draws=20)
    target = [
        r
        for r in rows
        if r["candidate"] == "qwen_block_10" and r["baseline"] == "bge_large"
    ]
    pooled = roc_auc_score(y, p[2]) - roc_auc_score(y, p[0])
    by_fold = np.mean(
        [
            roc_auc_score(y[folds == f], p[2, folds == f])
            - roc_auc_score(y[folds == f], p[0, folds == f])
            for f in range(5)
        ]
    )
    assert target[0]["delta"] == pytest.approx(pooled, abs=1e-12)
    assert target[1]["delta"] == pytest.approx(by_fold, abs=1e-12)


def test_missing_observer_or_fold_is_not_silently_accepted():
    """A partial family cannot be used to rank only its completed observers."""
    y = np.tile([0, 1], 50)
    with pytest.raises(ValueError, match="three paired"):
        paired_triads(y, [y, y], np.repeat(range(5), 20), "EI_cont", draws=20)
    with pytest.raises(ValueError, match="five complete"):
        paired_triads(y, [y, y, y], np.zeros(100), "EI_cont", draws=20)
