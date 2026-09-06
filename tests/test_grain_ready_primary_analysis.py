#!/usr/bin/env python3
"""Independent seven-head increments and complete-family/paired-source guards."""

import copy

import numpy as np
import pytest
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

from scripts.analyze_grain_ready_primary import (
    COHORT,
    OBSERVERS,
    PAIRS,
    cases,
    material_hash,
    paired_increments,
    require_complete_batch,
    require_pairing,
)


@pytest.mark.parametrize("task", ("big5", "EI_cont"))
def test_independent_scalar_and_bootstrap_replay(task):
    rng = np.random.default_rng(123)
    y = rng.normal(size=(100, 5)) if task == "big5" else np.tile([0, 1], 50)
    predictions = rng.normal(size=(7, *y.shape))
    if task != "big5":
        predictions = np.round(predictions, 1)
    folds = np.repeat(range(5), 20)
    rows = paired_increments(y, predictions, folds, task, draws=20)

    def metric(y0, p0):
        return (
            np.mean(pearsonr(y0, p0, axis=0).statistic)
            if task == "big5"
            else roc_auc_score(y0, p0)
        )

    parts = [np.flatnonzero(folds == f) for f in range(5)]

    def estimate(indices):
        joined = np.concatenate(indices)
        return np.array(
            [
                [metric(y[joined], p[joined]) for p in predictions],
                np.mean(
                    [[metric(y[ix], p[ix]) for p in predictions] for ix in indices],
                    axis=0,
                ),
            ]
        )

    observed = estimate(parts)
    rng = np.random.default_rng(731)
    draws = np.array(
        [
            estimate([rng.choice(ix, len(ix), replace=True) for ix in parts])
            for _ in range(20)
        ]
    )
    assert len(rows) == 18
    for n, (a, b) in enumerate(PAIRS):
        for k in range(2):
            r = rows[2 * n + k]
            assert r["candidate"] == OBSERVERS[a] and r["baseline"] == OBSERVERS[b]
            assert r["delta"] == pytest.approx(
                observed[k, a] - observed[k, b], abs=1e-12
            )
            assert [r["lower"], r["upper"]] == pytest.approx(
                np.quantile(draws[:, k, a] - draws[:, k, b], [0.025, 0.975]), abs=1e-12
            )


def test_identical_predictions_and_changed_author_guard():
    rng = np.random.default_rng(715)
    y = rng.normal(size=(100, 5))
    p = rng.normal(size=(100, 5))
    f = np.arange(100) % 5
    rows = paired_increments(y, np.tile(p[None], (7, 1, 1)), f, "big5", draws=20)
    assert all(
        r[k] == 0
        for r in rows
        for k in ("delta", "lower", "upper", "positive_fold_count")
    )
    base = {"ids": np.arange(100), "y": y, "folds": f}
    packets = [copy.deepcopy(base) for _ in range(7)]
    require_pairing(packets)
    packets[-1]["ids"] = np.roll(base["ids"], 1)
    with pytest.raises(ValueError):
        require_pairing(packets)
    with pytest.raises(ValueError):
        paired_increments(y, np.tile(p[None], (6, 1, 1)), f, "big5", draws=20)


def test_complete_batch_does_not_accept_partial_or_missing_case():
    m = {
        "protocol": "VM05_RAW_READY_PRIMARY_BATCH_V1",
        "cases": cases(),
        "cohort": COHORT,
        "arm": "raw",
        "pca": True,
        "readout": "body_token",
        "bridge_labels_read": False,
    }
    m["identity"] = material_hash(m)
    s = {
        "status": "COMPLETE",
        "child_pid": None,
        "reused_profiles": 1,
        "completed": [{"case": c["id"]} for c in cases()],
    }
    require_complete_batch(m, s)
    for key, val in (
        ("status", "PAUSED_BUDGET"),
        ("child_pid", 12),
        ("reused_profiles", 0),
        ("completed", s["completed"][:-1]),
    ):
        with pytest.raises(ValueError):
            require_complete_batch(m, s | {key: val})
