#!/usr/bin/env python3
"""Synthetic coverage/paired-inference tests; no natural outcomes are read."""

from copy import deepcopy

import numpy as np
import pytest
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

from grain_research.vector_mechanisms.controls import material_hash
from scripts.analyze_grain_cached_readouts import (
    paired_readouts,
    readouts,
    require_complete,
    require_pairing,
)
from scripts.run_grain_cached_readout_batch import cases
from scripts.summarize_grain_neural_source_primary import score


def fixture_packet():
    """Supply every non-outcome field needed by the complete-family gate."""
    m = {
        "protocol": "VM05_CACHED_SECONDARY_READOUT_BATCH_V1",
        "cases": cases(),
        "cohort": "full_source_reference_free",
        "pca": True,
        "seed": 731,
        "thread_control": "retain",
        "readout_selection": "none",
        "bridge_labels_read": False,
        "neural_forwards": 0,
    }
    m["identity"] = material_hash(m)
    s = {
        "status": "COMPLETE",
        "planned": 315,
        "child_pid": None,
        "bridge_labels_read": False,
        "program_complete": False,
        "completed": [{"case": c["id"]} for c in cases()],
    }
    return m, s


def test_gate_refuses_partial_duplicates_changed_family_and_live_child():
    m, s = fixture_packet()
    require_complete(m, s)
    for key, value in (
        ("status", "FIT"),
        ("child_pid", 123),
        ("completed", s["completed"][:-1]),
        ("completed", s["completed"][:-1] + s["completed"][:1]),
    ):
        bad = deepcopy(s)
        bad[key] = value
        with pytest.raises(ValueError, match="Complete315"):
            require_complete(m, bad)
    bad = deepcopy(m)
    bad["cases"] = bad["cases"][:-1]
    bad["identity"] = material_hash({k: v for k, v in bad.items() if k != "identity"})
    with pytest.raises(ValueError, match="Complete315"):
        require_complete(bad, s)


def test_pairing_rejects_author_target_or_fold_mismatch():
    a = {"ids": np.array(["a", "b"]), "y": np.array([0, 1]), "folds": np.array([0, 1])}
    require_pairing([a, deepcopy(a)])
    for key in a:
        bad = deepcopy(a)
        bad[key] = bad[key][::-1]
        with pytest.raises(ValueError, match="identical"):
            require_pairing([a, bad])


@pytest.mark.parametrize("task", ["big5", "EI_cont"])
def test_equal_heads_have_zero_intervals_and_both_estimands(task):
    rng = np.random.default_rng(200)
    folds = np.repeat(np.arange(5), 24)
    y = rng.normal(size=(120, 5)) if task == "big5" else np.tile([0, 1], 60)
    p = 0.4 * y + rng.normal(size=y.shape)
    predictions = np.stack([p] * len(readouts("block_10")))
    rows = paired_readouts(y, predictions, folds, task, "block_10", draws=20)
    assert len(rows) == 16
    assert {r["estimand"] for r in rows} == {"pooled", "mean_official_fold"}
    for r in rows:
        assert r["delta"] == r["lower"] == r["upper"] == 0
        assert r["familywise_confirmed"] is False
        expected = (
            score(y, p, task)
            if r["estimand"] == "pooled"
            else np.mean([score(y[folds == f], p[folds == f], task) for f in range(5)])
        )
        assert r["candidate_score"] == pytest.approx(expected)


def test_support_and_site_independent_margin_scope():
    assert "logit_margins" in readouts("block_10")
    assert all(
        "logit_margins" not in readouts(site)
        for site in ("block_04", "block_16", "final_norm")
    )
    with pytest.raises(ValueError, match="Unknown"):
        readouts("block_11")
    y = np.ones(100)
    folds = np.repeat(np.arange(5), 20)
    predictions = np.stack([np.linspace(0, 1, 100)] * 6)
    with pytest.raises(ValueError, match="Undefined source"):
        paired_readouts(y, predictions, folds, "EI_cont", "block_04", draws=20)


@pytest.mark.parametrize("task", ["big5", "TF_cont"])
def test_nonzero_contrasts_reproduce_independent_libraries(task):
    rng = np.random.default_rng(4242)
    y = rng.normal(size=(100, 5)) if task == "big5" else np.tile([0, 1], 50)
    ps = np.round(rng.normal(size=(6, *y.shape)), 1)
    folds = np.repeat(np.arange(5), 20)
    rows = paired_readouts(y, ps, folds, task, "block_04", draws=20)

    def metric(y0, p0):
        return (
            np.mean(pearsonr(y0, p0, axis=0).statistic)
            if task == "big5"
            else roc_auc_score(y0, p0)
        )

    def estimates(parts):
        joined = np.concatenate(parts)
        return np.array(
            [
                [metric(y[joined], p[joined]) for p in ps],
                np.mean([[metric(y[ix], p[ix]) for p in ps] for ix in parts], axis=0),
            ]
        )

    parts = [np.flatnonzero(folds == f) for f in range(5)]
    obs = estimates(parts)
    rng = np.random.default_rng(731)
    samples = np.array(
        [
            estimates([rng.choice(ix, len(ix), replace=True) for ix in parts])
            for _ in range(20)
        ]
    )
    names = readouts("block_04")
    assert len(rows) == 14
    for row in rows:
        c, b = names.index(row["candidate"]), names.index(row["baseline"])
        k = int(row["estimand"] == "mean_official_fold")
        assert row["delta"] == pytest.approx(obs[k, c] - obs[k, b], abs=1e-12)
        np.testing.assert_allclose(
            [row["lower"], row["upper"]],
            np.quantile(samples[:, k, c] - samples[:, k, b], [0.025, 0.975]),
            atol=1e-12,
        )
