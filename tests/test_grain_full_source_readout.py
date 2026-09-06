#!/usr/bin/env python3
"""Full-source readout boundaries and output replay use only synthetic people."""

import json
from types import SimpleNamespace

import joblib
import numpy as np
import pytest

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.ready_comparison import verified_packet
from scripts import run_grain_full_source_readout as runner
from scripts.summarize_grain_vector_external_links import load_oof


def material(tmp_path, task="EI_cont"):
    """Create source and bridge roles without any real psychological values."""
    source = [
        {
            "author_id": f"synthetic_{i:03d}",
            "source_group": ("big5_only" if task == "big5" else "mbti_only")
            if i < 80
            else "bridge",
            "official_folds": {task: i % 5},
        }
        for i in range(90)
    ]
    common = {
        "material_sha256": material_hash(source),
        "author_ids": [b["author_id"] for b in source],
    }
    parent = {"source_tables": {}}
    parent["identity"] = material_hash(parent)
    (tmp_path / "views.json").write_text(json.dumps(source))
    (tmp_path / "common_material_manifest.json").write_text(json.dumps(common))
    (tmp_path / "manifest.json").write_text(json.dumps(parent))
    a = SimpleNamespace(
        views=tmp_path / "views.json",
        preflight_dir=tmp_path,
        parent_dir=tmp_path,
        vector_dir=tmp_path / "vectors",
        output_dir=tmp_path,
        prepared_root=tmp_path,
        task=task,
        arm="raw",
        site="block_10",
        readout="decision_task_delta",
        exclude_reference=False,
        no_pca=False,
        preflight_only=False,
    )
    return a, source, common, parent


def fake_queue(monkeypatch):
    """Record all cache verifications while not constructing neural test artifacts."""
    verified = []
    monkeypatch.setattr(
        runner, "require_observer_completion", lambda *a: {"code_sha256": "synthetic"}
    )
    monkeypatch.setattr(runner, "digest", lambda *a: "synthetic")
    monkeypatch.setattr(
        runner,
        "verify_completed",
        lambda folder, source, common, stage, shards, i, code: verified.append(
            (stage, shards, i, code)
        ),
    )
    return verified


def test_full_source_selects_exact_rows_and_verifies_all_observers(
    tmp_path, monkeypatch
):
    a, source, _, _ = material(tmp_path)
    verified = fake_queue(monkeypatch)
    calls = []

    def features(bundles, *args):
        calls.append([b["author_id"] for b in bundles])
        assert all(b["source_group"] == "mbti_only" for b in bundles)
        assert args[-3:] == ("raw", "block_10", "decision_task_delta")
        return np.arange(len(bundles) * 6).reshape(len(bundles), 6)

    monkeypatch.setattr(runner, "cached_readout_matrix", features)
    _, _, _, bundles, folds, x, evidence = runner.source_inputs(a)
    assert len(verified) == 72 and len(set(verified)) == 72
    assert len(evidence) == 73 and x.shape == (80, 6)
    assert calls == [[b["author_id"] for b in source[:80]]]
    assert bundles == source[:80] and set(folds) == set(range(5))


def test_incomplete_cache_and_changed_material_fail_before_labels(
    tmp_path, monkeypatch
):
    a, source, common, _ = material(tmp_path)

    def blocked(*args):
        raise ValueError("All72 observer shards must finish")

    def forbidden(*args):
        raise AssertionError("No targets or features may be loaded")

    monkeypatch.setattr(runner, "require_observer_completion", blocked)
    monkeypatch.setattr(runner, "cached_readout_matrix", forbidden)
    monkeypatch.setattr(runner, "load_source_labels", forbidden)
    with pytest.raises(ValueError, match="All72"):
        runner.execute(a)
    assert not (tmp_path / "fit/source_label_audit.json").exists()
    a.readout, a.site = "logit_margins", "block_16"
    with pytest.raises(ValueError, match="site-independent"):
        runner.source_inputs(a)
    a.site = "block_10"
    common["author_ids"] = common["author_ids"][:-1]
    (tmp_path / "common_material_manifest.json").write_text(json.dumps(common))
    with pytest.raises(ValueError, match="common material"):
        runner.source_inputs(a)
    common["author_ids"] = [b["author_id"] for b in source]
    (tmp_path / "common_material_manifest.json").write_text(json.dumps(common))
    source[0]["official_folds"]["EI_cont"] = 4
    a.views.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="common material"):
        runner.source_inputs(a)


def test_preflight_never_converts_labels(tmp_path, monkeypatch):
    a, source, common, parent = material(tmp_path)
    a.preflight_only = True
    monkeypatch.setattr(
        runner,
        "source_inputs",
        lambda *args: (
            source,
            common,
            parent,
            source[:80],
            np.arange(80) % 5,
            np.ones((80, 7)),
            {"queue": "synthetic"},
        ),
    )
    monkeypatch.setattr(runner, "archive_source", lambda *args: None)

    def forbidden(*args):
        raise AssertionError("No target conversion or fitting during preflight")

    monkeypatch.setattr(runner, "load_source_labels", forbidden)
    monkeypatch.setattr(runner, "nested_fit", forbidden)
    runner.execute(a)
    s = json.loads((tmp_path / "fit/run_status.json").read_text())
    assert s["status"] == "PREFLIGHT_COMPLETE" and not s["source_label_values_read"]
    assert not list((tmp_path / "fit").glob("*.joblib"))


@pytest.mark.parametrize("task", ["big5", "EI_cont"])
def test_nested_full_source_outputs_replay_and_reject_mutation(
    tmp_path, monkeypatch, task
):
    a, source, common, parent = material(tmp_path, task)
    rng = np.random.default_rng(39)
    x, folds = rng.normal(size=(80, 7)), np.arange(80) % 5
    y = (
        rng.uniform(size=(80, 5))
        if task == "big5"
        else (np.arange(80) % 2).astype(float)
    )
    monkeypatch.setattr(
        runner,
        "source_inputs",
        lambda *args: (
            source,
            common,
            parent,
            source[:80],
            folds,
            x,
            {"queue": "synthetic"},
        ),
    )
    monkeypatch.setattr(runner, "archive_source", lambda *args: None)

    def labels(root, bundles, actual_task, tables):
        assert len(bundles) == 80 and actual_task == task
        assert all(b["source_group"] != "bridge" for b in bundles)
        return y, {"bridge_label_values_converted": False}

    monkeypatch.setattr(runner, "load_source_labels", labels)
    bootstrap = runner.bootstrap_score
    monkeypatch.setattr(
        runner, "bootstrap_score", lambda y, p, t: bootstrap(y, p, t, draws=20)
    )
    runner.execute(a)
    out = tmp_path / "fit"
    manifest, summary = verified_packet(out)
    result = load_oof(out)
    assert len(result["ids"]) == 80 and manifest["readout"] == "decision_task_delta"
    assert summary["saved_fold_replay_max"] < 1e-12
    assert summary["full_train_authors"] == 80 and not summary["bridge_labels_used"]
    for fold in range(5):
        model = joblib.load(out / f"private_fold_{fold}.joblib")
        np.testing.assert_allclose(
            model.named_steps["scale"].mean_, x[folds != fold].mean(0)
        )
    (out / "private_fold_0.joblib").write_bytes(
        b"deliberately corrupted synthetic artifact"
    )
    with pytest.raises(ValueError, match="checksum"):
        verified_packet(out)
