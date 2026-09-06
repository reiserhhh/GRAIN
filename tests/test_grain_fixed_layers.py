#!/usr/bin/env python3
"""Independent six-observer scores/intervals and whole-arm completion guards."""

import copy

import numpy as np
import pytest
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

from scripts.analyze_grain_fixed_layers import (
    OBSERVERS,
    PAIRS,
    cases,
    complete_arms,
    paired_family,
    require_paired,
)


@pytest.mark.parametrize("task", ("big5", "EI_cont"))
def test_six_observer_bootstrap_reproduces_independent_libraries(task):
    rng = np.random.default_rng(122)
    y = rng.normal(size=(100, 5)) if task == "big5" else np.tile([0, 1], 50)
    predictions = rng.normal(size=(6, *y.shape))
    if task != "big5":
        predictions = np.round(predictions, 1)
    folds = np.repeat(range(5), 20)
    rows = paired_family(y, predictions, folds, task, draws=20)

    def metric(y, p):
        return (
            np.mean(pearsonr(y, p, axis=0).statistic)
            if task == "big5"
            else roc_auc_score(y, p)
        )

    parts = [np.where(folds == f)[0] for f in range(5)]

    def estimate(ix):
        all_ix = np.concatenate(ix)
        return np.array(
            [
                [metric(y[all_ix], p[all_ix]) for p in predictions],
                np.mean(
                    [[metric(y[i], p[i]) for p in predictions] for i in ix], axis=0
                ),
            ]
        )

    observed = estimate(parts)
    rng = np.random.default_rng(731)
    draws = np.array(
        [
            estimate([rng.choice(i, len(i), replace=True) for i in parts])
            for _ in range(20)
        ]
    )
    assert len(rows) == 18
    for n, (a, b) in enumerate(PAIRS):
        for k in range(2):
            r = rows[n * 2 + k]
            assert r["candidate"] == OBSERVERS[a] and r["baseline"] == OBSERVERS[b]
            assert r["delta"] == pytest.approx(
                observed[k, a] - observed[k, b], abs=1e-12
            )
            np.testing.assert_allclose(
                [r["lower"], r["upper"]],
                np.quantile(draws[:, k, a] - draws[:, k, b], [0.025, 0.975]),
                atol=1e-12,
            )


def test_identical_observers_and_mismatched_packets():
    rng = np.random.default_rng(313)
    y = rng.normal(size=(100, 5))
    p = rng.normal(size=(100, 5))
    rows = paired_family(
        y, np.repeat(p[None], 6, axis=0), np.arange(100) % 5, "big5", draws=20
    )
    assert all(r[k] == 0 for r in rows for k in ("delta", "lower", "upper"))
    a = {"ids": np.arange(100), "y": y, "folds": np.arange(100) % 5}
    packets = [copy.deepcopy(a) for _ in range(6)]
    require_paired(packets)
    packets[-1]["ids"] = np.roll(a["ids"], 1)
    with pytest.raises(ValueError, match="identical authors"):
        require_paired(packets)
    with pytest.raises(ValueError, match="six finite"):
        paired_family(y, np.repeat(p[None], 5, axis=0), a["folds"], "big5", draws=20)


def test_partial_report_does_not_select_individual_completed_tasks(tmp_path):
    raw = [c for c in cases()[9:] if c["arm"] == "raw"]
    for c in raw[:14]:
        p = tmp_path / (c["id"] + "_replay")
        p.mkdir()
        (p / "summary.json").write_text("{}")
    assert complete_arms(tmp_path) == []
    p = tmp_path / (raw[-1]["id"] + "_replay")
    p.mkdir()
    (p / "summary.json").write_text("{}")
    assert complete_arms(tmp_path) == ["raw"]
    cue = next(c for c in cases()[9:] if c["arm"] == "cue_mask")
    p = tmp_path / (cue["id"] + "_replay")
    p.mkdir()
    (p / "summary.json").write_text("{}")
    assert complete_arms(tmp_path) == ["raw"]
