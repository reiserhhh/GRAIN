#!/usr/bin/env python3
"""Independent score and conditional bootstrap checks, with a held-out prior guard."""

from functools import partial

import numpy as np
import pytest
from sklearn.metrics import brier_score_loss, log_loss

from scripts.analyze_grain_source_proper_scores import (
    OBSERVERS,
    proper_scores,
    training_prior,
)


def example():
    """Use unequal fold sizes and nondegenerate class proportions."""
    folds = np.repeat(np.arange(5), [12, 14, 16, 18, 20])
    y = np.arange(len(folds)) % 3 == 0
    p = np.vstack([0.15 + 0.5 * y, np.full(len(y), 0.4), 0.75 - 0.5 * y])
    return y.astype(int), p, folds


def test_independent_scores_and_bootstrap_endpoints():
    """Recompute both losses/estimands and every interval with sklearn scalar calls."""
    y, p, folds = example()
    rows = proper_scores(y, p, folds, draws=30)
    prior = np.array([np.mean(y[folds != f]) for f in folds])
    parts = [np.where(folds == f)[0] for f in range(5)]
    rng = np.random.default_rng(731)
    samples = [
        [rng.choice(ix, len(ix), replace=True) for ix in parts] for _ in range(30)
    ]
    for row in rows:
        pred = p[OBSERVERS.index(row["observer"])]
        loss = (
            partial(log_loss, labels=[0, 1])
            if row["metric"] == "log_loss_nats"
            else brier_score_loss
        )

        def gain(indices, estimand=row["estimand"], loss=loss, pred=pred):
            selections = [np.concatenate(indices)] if estimand == "pooled" else indices
            return np.mean(
                [loss(y[ix], prior[ix]) - loss(y[ix], pred[ix]) for ix in selections]
            )

        assert row["improvement"] == pytest.approx(gain(parts), abs=1e-14)
        endpoints = np.quantile([gain(sample) for sample in samples], [0.025, 0.975])
        assert [row["lower"], row["upper"]] == pytest.approx(endpoints, abs=1e-14)
        assert row["clipped_predictions"] == 0


def test_prior_excludes_own_fold_and_identical_predictions_zero():
    """Changing held-out outcomes cannot alter that fold's prior predictions."""
    y, _, folds = example()
    prior = training_prior(y, folds)
    changed = y.copy()
    changed[folds == 2] = 1 - changed[folds == 2]
    assert np.array_equal(prior[folds == 2], training_prior(changed, folds)[folds == 2])
    rows = proper_scores(y, np.tile(prior, (3, 1)), folds, draws=20)
    assert all(r[k] == 0 for r in rows for k in ("improvement", "lower", "upper"))


def test_invalid_probabilities_support_and_clip_accounting():
    """Reject invalid score spaces instead of hiding them with clipping."""
    y, p, folds = example()
    for invalid in (-0.1, 1.1, np.nan):
        bad = p.copy()
        bad[0, 0] = invalid
        with pytest.raises(ValueError):
            proper_scores(y, bad, folds, draws=20)
    with pytest.raises(ValueError):
        training_prior(np.zeros(len(y)), folds)
    with pytest.raises(ValueError):
        training_prior(y, folds + 1)
    p[0, :2] = [0, 1]
    rows = proper_scores(y, p, folds, draws=20)
    assert all(
        r["clipped_predictions"] == 2 for r in rows if r["observer"] == OBSERVERS[0]
    )
