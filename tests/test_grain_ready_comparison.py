#!/usr/bin/env python3
"""Exact same-ready inputs, fold-local transformations and saved-head equivalence."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.external_heads import predict_head, select_head
from grain_research.vector_mechanisms.ready_comparison import (
    digest,
    feature_matrix,
    nested_fit,
    require_observer_completion,
    support_table,
    verified_packet,
)
from grain_research.vector_mechanisms.ready_registry import registry_arrays


def fixture():
    """Create80 source and10 bridge authors with a known finite synthetic profile."""
    source = []
    for i in range(90):
        source.append(
            {
                "author_id": f"author_{i:03d}",
                "source_group": "mbti_only" if i < 80 else "bridge",
                "official_folds": {
                    t: i % 5 for t in ("EI_cont", "SN_cont", "TF_cont", "JP_cont")
                },
                "views": [
                    [
                        {
                            "body": f"synthetic common words sample {i} item {j}",
                            "comment_id": f"{i}:{j}",
                            "thread_id": f"thread_{i}",
                        }
                        for j in range(12)
                    ]
                    for _ in range(3)
                ],
            }
        )
    rng = np.random.default_rng(41)
    profiles = {"author_ids": np.asarray([b["author_id"] for b in source])}
    for arm in ("raw", "cue_mask", "placebo_mask"):
        profiles[arm + "_status"] = np.asarray(["GEOMETRY_PROFILE_READY"] * 90)
        profiles[arm + "_sorted_distances"] = np.sort(rng.normal(size=(90, 16)), axis=1)
    profiles["raw_ready"] = np.ones(90, dtype=bool)
    profiles["all_arms_ready"] = np.ones(90, dtype=bool)
    reg = registry_arrays(source, profiles, {"author_000", "author_001"})
    return source, profiles, reg


def test_exact_source_subset_and_joint_columns():
    source, profiles, reg = fixture()
    called = []

    def loader(bundles, observer, arm):
        called.append((bundles, observer, arm))
        assert all(b["source_group"] == "mbti_only" for b in bundles)
        return np.tile([10.0, 20.0, 30.0], (len(bundles), 1))

    x, bundles, ix, blocks = feature_matrix(
        source,
        reg,
        profiles,
        "raw_ready_reference_free",
        "EI_cont",
        "raw",
        "grain_qwen_block_10",
        loader,
    )
    assert x.shape == (78, 19) and len(bundles) == 78
    np.testing.assert_array_equal(x[:, :16], profiles["raw_sorted_distances"][ix])
    assert blocks == [
        {"name": "sorted_distances", "column_start": 0, "column_stop": 16},
        {"name": "qwen_block_10", "column_start": 16, "column_stop": 19},
    ]
    assert called[0][1:] == ("qwen_block_10", "raw")
    text, _, _, _ = feature_matrix(
        source, reg, profiles, "raw_ready", "EI_cont", "raw", "word_char", loader
    )
    assert len(text) == 80 and text[0].count("synthetic") == 36
    assert len(called) == 1


def test_refused_comparisons_do_not_load_candidate():
    source, profiles, reg = fixture()

    def forbidden(*args):
        raise AssertionError("Must fail before loading neural candidate")

    with pytest.raises(ValueError, match="all-arms-ready"):
        feature_matrix(
            source,
            reg,
            profiles,
            "raw_ready",
            "EI_cont",
            "cue_mask",
            "bge_large",
            forbidden,
        )
    with pytest.raises(ValueError, match="Reuse completed"):
        feature_matrix(
            source,
            reg,
            profiles,
            "raw_ready",
            "EI_cont",
            "raw",
            "sorted_distances",
            forbidden,
        )
    bad = copy.deepcopy(reg)
    bad["author_ids"][[2, 3]] = bad["author_ids"][[3, 2]]
    with pytest.raises(ValueError, match="author order"):
        feature_matrix(
            source, bad, profiles, "raw_ready", "EI_cont", "raw", "bge_large", forbidden
        )
    x, _, _, _ = feature_matrix(
        source,
        reg,
        profiles,
        "raw_ready_reference_free",
        "EI_cont",
        "raw",
        "sorted_distances",
        forbidden,
    )
    assert x.shape == (78, 16)


def test_target_support_cannot_be_silently_fixed_by_resampling():
    n = 80
    ids, folds = np.asarray([str(i) for i in range(n)]), np.arange(n) % 5
    y = (np.arange(n) % 2).astype(float)
    rows = support_table(y, ids, folds, "EI_cont")
    assert len(rows) == 47 and all(r["supported"] for r in rows)
    y[folds == 0] = 0
    assert not all(r["supported"] for r in support_table(y, ids, folds, "EI_cont"))
    with pytest.raises(ValueError, match="Insufficient target support"):
        nested_fit(
            np.zeros((n, 3)), y, ids, folds, "EI_cont", "dense", True, lambda *a: None
        )
    rng = np.random.default_rng(37)
    assert all(
        r["supported"]
        for r in support_table(rng.uniform(size=(n, 5)), ids, folds, "big5")
    )
    with pytest.raises(ValueError, match="five-trait"):
        support_table(np.ones((n, 4)), ids, folds, "big5")


def test_nested_readout_matches_existing_source_head_without_test_fit():
    rng = np.random.default_rng(47)
    n = 80
    x = rng.normal(size=(n, 7))
    ids, folds = np.asarray([str(i) for i in range(n)]), np.arange(n) % 5
    y = (np.arange(n) % 2).astype(float)
    models = {}

    def checkpoint(fold, model, selections):
        models[fold] = model
        tr = folds != fold
        np.testing.assert_allclose(model.named_steps["scale"].mean_, x[tr].mean(0))
        assert selections[-1]["pca_components"] == 6

    with threadpool_limits(limits=1):
        pred, selection, _, full, _, _ = nested_fit(
            x, y, ids, folds, "EI_cont", "dense", True, checkpoint
        )
        tr, te = folds != 0, folds == 0
        direct, _, parameter = select_head(
            x[tr], y[tr], ids[tr], "dense", "EI_cont", pca=True
        )
        np.testing.assert_allclose(
            pred[te], predict_head(direct, x[te], "EI_cont"), atol=1e-12
        )
    assert selection[0]["parameter"] == parameter and len(models) == 5
    np.testing.assert_allclose(full.named_steps["scale"].mean_, x.mean(0))


def test_manifest_mutation_and_path_escape_rejected(tmp_path):
    m = {"protocol": "synthetic"}
    m["identity"] = material_hash(m)
    (tmp_path / "manifest.json").write_text(json.dumps(m))
    data = tmp_path / "data.npz"
    np.savez(data, x=np.arange(3))
    s = {
        "status": "COMPLETE",
        "parent_identity": m["identity"],
        "outputs_sha256": {"data.npz": digest(data)},
    }
    (tmp_path / "summary.json").write_text(json.dumps(s))
    assert verified_packet(tmp_path)[0] == m
    data.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        verified_packet(tmp_path)
    s["outputs_sha256"] = {"../outside.npz": "does-not-matter"}
    (tmp_path / "summary.json").write_text(json.dumps(s))
    with pytest.raises(ValueError, match="checksum"):
        verified_packet(tmp_path)


def test_complete_observer_stage_cannot_substitute_for_full_queue(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps({"shards": 24, "material_sha256": "original"})
    )
    for stage in ("bge_large", "e5_large", "qwen"):
        for shard in range(24):
            folder = tmp_path / stage / f"shard_{shard:02d}"
            folder.mkdir(parents=True)
            (folder / "run_status.json").write_text(json.dumps({"status": "COMPLETE"}))
    assert require_observer_completion(tmp_path, "original")["shards"] == 24
    (tmp_path / "qwen/shard_23/run_status.json").write_text(
        json.dumps({"status": "RUNNING"})
    )
    with pytest.raises(ValueError, match="All72"):
        require_observer_completion(tmp_path, "original")
    with pytest.raises(ValueError, match="common observation"):
        require_observer_completion(tmp_path, "changed")


@pytest.mark.parametrize("task", ["big5", "EI_cont"])
def test_execution_artifacts_replay_with_synthetic_only_labels(
    tmp_path, monkeypatch, task
):
    from scripts import run_grain_same_ready_comparison as runner
    from scripts.summarize_grain_vector_external_links import load_oof

    rng = np.random.default_rng(11)
    n = 80
    ids = np.asarray([f"synthetic_{i:03d}" for i in range(n)])
    source = [
        {"author_id": u, "source_group": "big5_only" if task == "big5" else "mbti_only"}
        for u in ids
    ]
    registry = {"author_ids": ids, task + "_fold": np.arange(n) % 5}
    x = rng.normal(size=(n, 7))
    y = rng.uniform(size=(n, 5)) if task == "big5" else (np.arange(n) % 2).astype(float)
    parent = {"source_tables": {}}
    parent["identity"] = material_hash(parent)
    (tmp_path / "manifest.json").write_text(json.dumps(parent))
    (tmp_path / "summary.json").write_text("{}")
    monkeypatch.setattr(
        runner,
        "load_registry_inputs",
        lambda *a: (
            source,
            {"material_sha256": material_hash(source)},
            registry,
            {},
            {"identity": "synthetic"},
        ),
    )
    monkeypatch.setattr(
        runner,
        "feature_matrix",
        lambda *a: (
            x,
            source,
            np.arange(n),
            [{"name": "synthetic", "column_start": 0, "column_stop": 7}],
        ),
    )
    monkeypatch.setattr(runner, "archive_source", lambda *a: None)

    def labels(root, bundles, actual_task, tables):
        assert actual_task == task and all(
            b["source_group"] != "bridge" for b in bundles
        )
        return y, {"bridge_label_values_converted": False}

    monkeypatch.setattr(runner, "load_source_labels", labels)
    original_bootstrap = runner.bootstrap_score
    monkeypatch.setattr(
        runner, "bootstrap_score", lambda y, p, t: original_bootstrap(y, p, t, draws=20)
    )
    args = SimpleNamespace(
        output_dir=tmp_path,
        views=tmp_path,
        registry_dir=tmp_path,
        registry_audit_dir=tmp_path,
        profile_dir=tmp_path,
        preflight_dir=tmp_path,
        parent_dir=tmp_path,
        prepared_root=tmp_path,
        cohort="raw_ready",
        task=task,
        arm="raw",
        feature="bge_large",
        no_pca=False,
        preflight_only=False,
    )
    runner.execute(args)
    result = load_oof(tmp_path / "fit")
    assert result["summary"]["authors"] == n
    assert result["summary"]["saved_fold_replay_max"] < 1e-12
    assert len(json.loads((tmp_path / "fit/selection.json").read_text())) == 5
    assert verified_packet(tmp_path / "fit")[1]["status"] == "COMPLETE"
