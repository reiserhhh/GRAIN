#!/usr/bin/env python3
"""Frozen sensitivity grid, actual PCA flags, and bounded fit/replay/resume wiring."""

import json
import zipfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from grain_research.vector_mechanisms.controls import material_hash
from scripts import run_grain_neural_sensitivity_batch as entry


def fixture_packet(case, source, common):
    """Small source-only packet independent of the run's result generator."""
    return {
        "manifest": {
            "protocol": "GRAIN_VM05_SOURCE_OOF_V1",
            "source_group": "big5_only",
            "exclude_reference": True,
            "thread_control": "retain",
            "drop_seed": 731,
            "permutation_seed": None,
            "seed": 731,
            "bridge_labels_used": False,
            "inner_cv": "3 author folds, seed731",
            "outer_cv": "5 source-specific official folds",
            "common_material_sha256": common,
            "source_material_sha256": material_hash(source),
            "pca_applied": case["pca"],
            "identity": "toy-source",
            **{k: case[k] for k in ("task", "arm", "feature", "pca")},
        },
        "summary": {"bridge_labels_used": False, "score": 0.1},
        "ids": np.asarray([r["author_id"] for r in source]),
        "folds": np.arange(5),
    }


def sources():
    return [
        {
            "author_id": str(i),
            "source_group": "big5_only",
            "official_folds": {"big5": i},
        }
        for i in range(5)
    ]


def write_replay(folder, case, *, pca=False):
    """Create a synthetic zero-error five-fold replay, not a fitted scientific result."""
    folder.mkdir(exist_ok=True)
    (folder / "summary.json").write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "fold_models": 5,
                "new_label_values_read": False,
                "bridge_labels_read": False,
                "model_refits": 0,
                "max_absolute_error": 0.0,
            }
        )
    )
    pd.DataFrame(
        {
            "run": [case["id"]] * 5,
            "fold": range(5),
            "max_absolute_error": [0.0] * 5,
            "pca_applied": [pca] * 5,
        }
    ).to_csv(folder / "replay_by_fold.csv", index=False)


def test_exact_predeclared_grid():
    cases = entry.cases()
    assert len(cases) == len({c["id"] for c in cases}) == 54
    assert all(c["task"] == "big5" and not c["pca"] for c in cases[:9])
    assert all(c["pca"] for c in cases[9:])
    assert {c["feature"] for c in cases[9:]} == {
        "qwen_block_04",
        "qwen_block_16",
        "qwen_final_norm",
    }
    assert len({(c["task"], c["arm"]) for c in cases[9:]}) == 15


def test_real_projection_flag_and_folds_are_required(tmp_path):
    case, source = entry.cases()[0], sources()
    p = fixture_packet(case, source, "common")
    entry.validate_result(p, case, source, "common")
    p["manifest"]["pca_applied"] = True
    with pytest.raises(ValueError, match="readout condition"):
        entry.validate_result(p, case, source, "common")
    p["manifest"]["pca_applied"] = False
    p["folds"] = np.roll(p["folds"], 1)
    with pytest.raises(ValueError, match="official folds"):
        entry.validate_result(p, case, source, "common")
    write_replay(tmp_path, case)
    entry.validate_replay(tmp_path, case)
    write_replay(tmp_path, case, pca=True)
    with pytest.raises(ValueError, match="projection"):
        entry.validate_replay(tmp_path, case)


def test_primary_partial_and_changed_condition_are_rejected(tmp_path, monkeypatch):
    primary, resource = tmp_path / "primary", tmp_path / "resource"
    primary.mkdir()
    resource.mkdir()
    resource_path = resource / "batch_manifest.json"
    resource_path.write_text(
        json.dumps({"protocol": "VM05_STRONG_SPARSE_SOURCE_CONTROLS_V1"})
    )
    cases = entry.primary_cases()
    m = {
        "protocol": "VM05_FULL_NEURAL_SOURCE_QUEUE_V1",
        "cases": cases,
        "scoring_code": {},
        "predecessor_manifest_sha256": entry.digest(resource_path),
    }
    m["identity"] = material_hash(m)
    (primary / "batch_manifest.json").write_text(json.dumps(m))
    s = {
        "status": "PAUSED_BUDGET",
        "completed": [{"case": c["id"]} for c in cases],
        "child_pid": None,
    }
    (primary / "batch_status.json").write_text(json.dumps(s))
    am = {"batch_identity": m["identity"], "inputs": {c["id"]: {} for c in cases}}
    audit = {
        "heads": 45,
        "pending_groups": [],
        "accepted_groups": [{}] * 15,
        "bridge_labels_read": False,
        "model_refits": 0,
    }
    monkeypatch.setattr(entry, "verified_packet", lambda _: (am, audit))
    with pytest.raises(ValueError, match="Complete primary"):
        entry.validate_primary(primary, tmp_path, resource)
    s["status"] = "COMPLETE"
    s["completed"][0]["case"] = "wrong-source-head"
    (primary / "batch_status.json").write_text(json.dumps(s))
    with pytest.raises(ValueError, match="Complete primary"):
        entry.validate_primary(primary, tmp_path, resource)


def test_bounded_fit_replay_and_identical_resume(tmp_path, monkeypatch):
    source, case = sources(), entry.cases()[0]
    common = material_hash(source)
    for name in (
        "preflight",
        "parent",
        "vector",
        "primary",
        "resource",
        "audit",
        "scripts",
    ):
        (tmp_path / name).mkdir()
    (tmp_path / "views.json").write_text(json.dumps(source))
    (tmp_path / "preflight/common_material_manifest.json").write_text(
        json.dumps({"material_sha256": common})
    )
    for name in ("preflight", "parent", "vector"):
        (tmp_path / name / "manifest.json").write_text("{}")
    pd.DataFrame({"author_id": [r["author_id"] for r in source]}).to_csv(
        tmp_path / "preflight/budget_by_author.csv", index=False
    )
    for name in (
        "run_grain_neural_sensitivity_batch",
        "run_grain_vector_mechanism_neural_source_queue",
        "run_grain_vector_mechanism_sparse_baselines",
        "summarize_grain_vector_external_replay",
        "summarize_grain_vector_external_links",
    ):
        (tmp_path / "scripts" / (name + ".py")).write_text("# synthetic source\n")
    primary = {
        "identity": "complete-primary",
        "views_sha256": entry.digest(tmp_path / "views.json"),
        "preflight_manifest_sha256": entry.digest(tmp_path / "preflight/manifest.json"),
        "source_parent_sha256": entry.digest(tmp_path / "parent/manifest.json"),
        "vector_manifest_sha256": entry.digest(tmp_path / "vector/manifest.json"),
    }
    monkeypatch.setattr(entry, "ROOT", tmp_path)
    monkeypatch.setattr(
        entry,
        "validate_primary",
        lambda *args: (primary, {"identity": "audited-primary"}),
    )
    monkeypatch.setattr(entry, "cases", lambda: [case])
    monkeypatch.setattr(entry, "scoring_commitment", dict)
    monkeypatch.setattr(entry, "exclude_reference", lambda s, _: s)

    def archive(root, path):
        with zipfile.ZipFile(path, "x"):
            pass

    monkeypatch.setattr(entry, "archive_source", archive)
    packet = fixture_packet(case, source, common)
    calls = []

    def child(cmd, log, seconds, memory, update):
        from pathlib import Path

        target = Path(cmd[cmd.index("--output-dir") + 1])
        update(999999, 1, 1024, 1024)
        if "--runs" in cmd:
            calls.append("replay")
            assert seconds == 600
            write_replay(target, case)
        else:
            calls.append("fit")
            assert "--no-pca" in cmd and "--exclude-reference" in cmd
            assert seconds == 3600 and memory == 8 * 1024**3
            target.mkdir()
            full = target / "private_full_source_head.joblib"
            full.write_bytes(b"synthetic head")
            packet["summary"]["full_head_sha256"] = entry.digest(full)
        return {"seconds": 1, "peak_sampled_rss_bytes": 1024}

    monkeypatch.setattr(entry, "bounded_child", child)
    monkeypatch.setattr(entry, "load_oof", lambda _: packet)
    a = SimpleNamespace(
        views=tmp_path / "views.json",
        preflight_dir=tmp_path / "preflight",
        parent_dir=tmp_path / "parent",
        vector_dir=tmp_path / "vector",
        primary_dir=tmp_path / "primary",
        resource_dir=tmp_path / "resource",
        audit_dir=tmp_path / "audit",
        prepared_root=tmp_path / "unused",
        output_dir=tmp_path / "results/toy",
        max_seconds=28800,
        resume_completed=False,
    )
    entry.run(a)
    assert calls == ["fit", "replay"]
    assert (
        json.loads((a.output_dir / "batch_status.json").read_text())["status"]
        == "COMPLETE"
    )
    a.resume_completed = True
    entry.run(a)
    assert calls == ["fit", "replay"]
    import fcntl

    previous = (a.output_dir / "batch_status.json").read_bytes()
    with (a.resource_dir / ".execution.lock").open("a") as competing:
        fcntl.flock(competing, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            entry.run(a)
    assert (a.output_dir / "batch_status.json").read_bytes() == previous
    assert calls == ["fit", "replay"]
    (a.output_dir / (case["id"] + "_replay") / "summary.json").write_text("{}")
    with pytest.raises(KeyError):
        entry.run(a)
    assert calls == ["fit", "replay"]
    assert (
        json.loads((a.output_dir / "batch_status.json").read_text())["status"]
        == "FAILED_STOP"
    )
