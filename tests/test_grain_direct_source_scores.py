#!/usr/bin/env python3
"""Complete-family, role, aggregation and statistical checks without natural labels."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

from grain_research.vector_mechanisms.controls import material_hash
from scripts import analyze_grain_direct_source_scores as module


def fixture_queue():
    """Give the minimum immutable complete-only queue fields, not real model evidence."""
    m = {
        "protocol": "VM05_DIRECT_QUEUE_V1",
        "preflight_only": False,
        "shards": 128,
        "authors": 128,
        "labels_read": False,
        "bridge_outcomes_read": False,
    }
    m["identity"] = material_hash(m)
    s = {
        "status": "COMPLETE",
        "parent_identity": m["identity"],
        "child_pid": None,
        "real_neural_collection": True,
        "labels_read": False,
        "bridge_outcomes_read": False,
        "verified_authors": 128,
        "completed": [{"shard": i, "authors": 1} for i in range(128)],
    }
    return m, s


def test_complete_gate_rejects_partial_preflight_or_changed_membership():
    m, s = fixture_queue()
    module.require_complete(m, s)
    for key, value in (
        ("status", "RUNNING"),
        ("child_pid", 2),
        ("labels_read", True),
        ("completed", s["completed"][:-1]),
        ("completed", s["completed"][:-1] + s["completed"][:1]),
        ("verified_authors", 127),
        ("real_neural_collection", False),
    ):
        bad = deepcopy(s)
        bad[key] = value
        with pytest.raises(ValueError, match="Complete128"):
            module.require_complete(m, bad)
    bad = deepcopy(m)
    bad["preflight_only"] = True
    bad["identity"] = material_hash({k: v for k, v in bad.items() if k != "identity"})
    with pytest.raises(ValueError, match="Complete128"):
        module.require_complete(bad, s | {"parent_identity": bad["identity"]})


def test_partial_refused_before_source_loader_or_output(tmp_path, monkeypatch):
    m, s = fixture_queue()
    s["status"] = "RUNNING"
    module.write_json(tmp_path / "manifest.json", m)
    module.write_json(tmp_path / "queue_status.json", s)

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "must not read source outcomes or material before complete gate"
        )

    monkeypatch.setattr(module, "load_registration", forbidden)
    monkeypatch.setattr(module, "load_oof", forbidden)
    out = tmp_path / "analysis"
    with pytest.raises(ValueError, match="Complete128"):
        module.analyze(SimpleNamespace(queue_dir=tmp_path, output_dir=out))
    assert not out.exists()


def author_record(group="mbti_only"):
    """Include a low-mass noncompliant view which must not be silently filtered."""
    rows = [
        {
            "task": "EI_cont",
            "arm": "raw",
            "view": i,
            "conditional_score": p,
            "candidate_mass": 0.1,
            "conditional_entropy": 0.5,
            "greedy_first_token_is_candidate": i != 0,
        }
        for i, p in enumerate([0.1, 0.4, 0.7])
    ]
    return {
        "author_id": "a",
        "source_group": group,
        "readouts": rows,
        "view_means": {"raw__EI_cont": float(np.mean([0.1, 0.4, 0.7]))},
    }


def test_all_views_retained_and_bridge_prohibited():
    record = author_record()
    p, quality = module.direct_prediction({"a": record}, ["a"], "EI_cont", "raw")
    np.testing.assert_allclose(p, [0.4])
    assert len(quality) == 3
    assert sum(r["greedy_first_token_is_candidate"] for r in quality) == 2
    for group in ("bridge", "big5_only"):
        with pytest.raises(ValueError, match="source role"):
            module.direct_prediction(
                {"a": author_record(group)}, ["a"], "EI_cont", "raw"
            )
    record["readouts"].pop()
    with pytest.raises(ValueError, match="all three"):
        module.direct_prediction({"a": record}, ["a"], "EI_cont", "raw")


@pytest.mark.parametrize("task", ["big5", "EI_cont"])
def test_paired_metrics_and_intervals_match_independent_libraries(task):
    rng = np.random.default_rng(919)
    y = rng.normal(size=(100, 5)) if task == "big5" else np.tile([0, 1], 50)
    predictions = rng.normal(size=(3, *y.shape))
    folds = np.repeat(np.arange(5), 20)
    rows = module.paired_scores(
        y, predictions, folds, task, ("a", "b", "c"), ((0, 1), (0, 2)), draws=20
    )

    def metric(y, p):
        return (
            np.mean(pearsonr(y, p, axis=0).statistic)
            if task == "big5"
            else roc_auc_score(y, p)
        )

    def estimates(parts):
        all_ids = np.concatenate(parts)
        return np.array(
            [
                [metric(y[all_ids], p[all_ids]) for p in predictions],
                np.mean(
                    [[metric(y[ix], p[ix]) for p in predictions] for ix in parts],
                    axis=0,
                ),
            ]
        )

    parts = [np.flatnonzero(folds == f) for f in range(5)]
    observed = estimates(parts)
    rng = np.random.default_rng(731)
    samples = np.array(
        [
            estimates([rng.choice(ix, len(ix), replace=True) for ix in parts])
            for _ in range(20)
        ]
    )
    for row in rows:
        b = ("a", "b", "c").index(row["baseline"])
        k = int(row["estimand"] == "mean_official_fold")
        assert row["delta"] == pytest.approx(observed[k, 0] - observed[k, b])
        np.testing.assert_allclose(
            [row["lower"], row["upper"]],
            np.quantile(samples[:, k, 0] - samples[:, k, b], [0.025, 0.975]),
            atol=1e-12,
        )
        assert (
            row["status"] == "DEFINED_POINTWISE"
            and row["familywise_confirmed"] is False
        )


def test_constant_native_big5_preserved_as_undefined_not_removed():
    rng = np.random.default_rng(5)
    y = rng.normal(size=(100, 5))
    p = np.stack([np.ones_like(y), y])
    rows = module.paired_scores(
        y,
        p,
        np.repeat(np.arange(5), 20),
        "big5",
        ("native", "head"),
        ((0, 1),),
        draws=20,
    )
    assert len(rows) == 2
    for row in rows:
        assert (
            row["candidate_score"] is None
            and row["delta"] is None
            and row["lower"] is None
        )
        assert (
            row["status"] == "UNDEFINED_OR_LOW_SUPPORT" and row["undefined_draws"] == 20
        )


def test_constant_binary_predictor_has_half_auc_without_threshold_rescue():
    y = np.tile([0, 1], 50)
    p = np.stack([np.ones(100) * 0.99, y])
    rows = module.paired_scores(
        y,
        p,
        np.repeat(np.arange(5), 20),
        "EI_cont",
        ("native", "head"),
        ((0, 1),),
        draws=20,
    )
    assert all(r["candidate_score"] == 0.5 and r["delta"] == -0.5 for r in rows)


def test_big5_trait_order_is_bound_to_source_order():
    names = [t.lower() for t in module.TRAITS]
    record = {
        "author_id": "a",
        "source_group": "big5_only",
        "readouts": [],
        "view_means": {},
    }
    for j, name in enumerate(names):
        value = j / 4
        record["view_means"][f"raw__{name}"] = value
        for i in range(3):
            record["readouts"].append(
                {
                    "task": name,
                    "arm": "raw",
                    "view": i,
                    "conditional_score": value,
                    "candidate_mass": 1.0,
                    "conditional_entropy": 0.0,
                    "greedy_first_token_is_candidate": True,
                }
            )
    p, q = module.direct_prediction({"a": record}, ["a"], "big5", "raw")
    np.testing.assert_array_equal(p, [np.arange(5) / 4])
    assert len(q) == 15
