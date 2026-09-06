#!/usr/bin/env python3
"""CPU resource handoff, complete frozen conditions and failure publication."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from grain_research.vector_mechanisms.controls import material_hash
from scripts import run_grain_vector_mechanism_neural_source_queue as entry


def test_full_frozen_condition_grid():
    rows = entry.cases()
    assert len(rows) == 45 and len({r["id"] for r in rows}) == 45
    assert [r["feature"] for r in rows[:3]] == [
        "bge_large",
        "e5_large",
        "qwen_block_10",
    ]
    assert all(r["task"] == "big5" and r["arm"] == "raw" for r in rows[:3])
    assert {r["task"] for r in rows} == {
        "big5",
        "EI_cont",
        "SN_cont",
        "TF_cont",
        "JP_cont",
    }


def test_pid_ownership_and_terminal_file_are_both_required(tmp_path, monkeypatch):
    monkeypatch.setattr(
        entry.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            returncode=0, stdout="python run_grain_strong_source_controls.py"
        ),
    )
    # A live verified owner needs no race-prone reads of its changing status file.
    assert not entry.predecessor_ready(tmp_path, 42)
    monkeypatch.setattr(
        entry.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout="unrelated worker"),
    )
    with pytest.raises(RuntimeError, match="ownership"):
        entry.owner_alive(42)
    monkeypatch.setattr(
        entry.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=1, stdout=""),
    )
    path = tmp_path / "batch_status.json"
    path.write_text(json.dumps({"status": "RUNNING"}), encoding="utf-8")
    with pytest.raises(ValueError, match="terminal"):
        entry.predecessor_ready(tmp_path, 42)
    path.write_text(json.dumps({"status": "PAUSED_BUDGET"}), encoding="utf-8")
    assert entry.predecessor_ready(tmp_path, 42)


def test_owned_exception_is_terminal_but_lock_refusal_preserves_status(tmp_path):
    import fcntl

    with pytest.raises(RuntimeError), entry.execution_lock(tmp_path) as lease:
        lease["started"] = True
        raise RuntimeError("synthetic fit failure")
    path = tmp_path / "batch_status.json"
    assert json.loads(path.read_text())["status"] == "FAILED_STOP"
    before = path.read_bytes()
    with (tmp_path / ".execution.lock").open("a") as other:
        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError), entry.execution_lock(tmp_path):
            pass
    assert path.read_bytes() == before


def test_source_validation_rejects_wrong_membership_direction_and_folds():
    case = entry.cases()[0]
    source = [
        {
            "author_id": str(i),
            "source_group": "big5_only",
            "official_folds": {"big5": i},
        }
        for i in range(5)
    ]
    m = {
        "protocol": "GRAIN_VM05_SOURCE_OOF_V1",
        "task": "big5",
        "arm": "raw",
        "feature": "bge_large",
        "source_group": "big5_only",
        "exclude_reference": True,
        "thread_control": "retain",
        "drop_seed": 731,
        "permutation_seed": None,
        "seed": 731,
        "pca": True,
        "pca_applied": True,
        "inner_cv": "3 author folds, seed731",
        "outer_cv": "5 source-specific official folds",
        "bridge_labels_used": False,
        "common_material_sha256": "c",
        "source_material_sha256": material_hash(source),
    }
    p = {
        "manifest": m,
        "summary": {"bridge_labels_used": False},
        "ids": np.asarray([str(i) for i in range(5)]),
        "folds": np.arange(5),
    }
    entry.validate_result(p, case, source, "c")
    p["folds"] = np.roll(p["folds"], 1)
    with pytest.raises(ValueError, match="folds"):
        entry.validate_result(p, case, source, "c")
    p["folds"] = np.arange(5)
    m["permutation_seed"] = 732
    with pytest.raises(ValueError, match="condition"):
        entry.validate_result(p, case, source, "c")


def test_scheduler_executes_fit_then_replay_only_after_cpu_handoff(
    tmp_path, monkeypatch
):
    """Exercise scheduler wiring with one explicit synthetic case, never real labels/models."""
    import pandas as pd

    source = [
        {
            "author_id": str(i),
            "source_group": "big5_only",
            "official_folds": {"big5": i},
        }
        for i in range(5)
    ]
    common_hash = material_hash(source)
    for name in ("preflight", "parent", "vector", "predecessor"):
        (tmp_path / name).mkdir()
    (tmp_path / "views.json").write_text(json.dumps(source), encoding="utf-8")
    (tmp_path / "preflight/common_material_manifest.json").write_text(
        json.dumps({"material_sha256": common_hash}), encoding="utf-8"
    )
    for name in (
        "preflight/manifest.json",
        "parent/manifest.json",
        "vector/manifest.json",
    ):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    (tmp_path / "predecessor/batch_manifest.json").write_text(
        json.dumps({"protocol": "VM05_STRONG_SPARSE_SOURCE_CONTROLS_V1"}),
        encoding="utf-8",
    )
    pd.DataFrame({"author_id": [b["author_id"] for b in source]}).to_csv(
        tmp_path / "preflight/budget_by_author.csv", index=False
    )
    names = (
        "scripts/run_grain_vector_mechanism_neural_source_queue.py",
        "scripts/run_grain_vector_mechanism_sparse_baselines.py",
        "scripts/summarize_grain_vector_external_replay.py",
        "scripts/summarize_grain_vector_external_links.py",
    )
    (tmp_path / "scripts").mkdir()
    for name in names:
        (tmp_path / name).write_text("# synthetic source\n", encoding="utf-8")
    case = entry.cases()[0]
    monkeypatch.setattr(entry, "ROOT", tmp_path)
    monkeypatch.setattr(entry, "cases", lambda: [case])
    monkeypatch.setattr(entry, "scoring_commitment", dict)
    monkeypatch.setattr(entry, "require_observer_completion", lambda *args: {})
    monkeypatch.setattr(entry, "exclude_reference", lambda rows, budget: rows)
    monkeypatch.setattr(entry, "predecessor_ready", lambda *args: True)
    monkeypatch.setattr(
        entry, "archive_source", lambda root, path: path.write_bytes(b"fixture")
    )
    m = {
        "protocol": "GRAIN_VM05_SOURCE_OOF_V1",
        "task": "big5",
        "arm": "raw",
        "feature": "bge_large",
        "source_group": "big5_only",
        "exclude_reference": True,
        "thread_control": "retain",
        "drop_seed": 731,
        "permutation_seed": None,
        "seed": 731,
        "pca": True,
        "pca_applied": True,
        "inner_cv": "3 author folds, seed731",
        "outer_cv": "5 source-specific official folds",
        "bridge_labels_used": False,
        "common_material_sha256": common_hash,
        "source_material_sha256": common_hash,
        "identity": "fixture",
    }
    packet = {
        "manifest": m,
        "summary": {"bridge_labels_used": False, "authors": 5, "score": 0.1},
        "ids": np.array([str(i) for i in range(5)]),
        "folds": np.arange(5),
    }
    monkeypatch.setattr(entry, "load_oof", lambda path: packet)
    called = []

    def child(cmd, log, seconds, memory, update):
        """Write generated mock results while checking bounded fit/replay order."""
        from pathlib import Path

        folder = Path(cmd[cmd.index("--output-dir") + 1])
        folder.mkdir()
        update(999999, 1, 1024, 1024)
        if "--runs" in cmd:
            called.append("replay")
            (folder / "summary.json").write_text(
                json.dumps(
                    {
                        "status": "COMPLETE",
                        "fold_models": 5,
                        "max_absolute_error": 0.0,
                        "new_label_values_read": False,
                        "bridge_labels_read": False,
                        "model_refits": 0,
                    }
                ),
                encoding="utf-8",
            )
            pd.DataFrame(
                {
                    "run": [case["id"]] * 5,
                    "fold": range(5),
                    "max_absolute_error": [0.0] * 5,
                }
            ).to_csv(folder / "replay_by_fold.csv", index=False)
        else:
            called.append("fit")
            assert "--exclude-reference" in cmd and "--vector-dir" in cmd
        return {"seconds": 1, "peak_sampled_rss_bytes": 1024}

    monkeypatch.setattr(entry, "bounded_child", child)
    a = SimpleNamespace(
        views=tmp_path / "views.json",
        preflight_dir=tmp_path / "preflight",
        parent_dir=tmp_path / "parent",
        prepared_root=tmp_path / "unused",
        vector_dir=tmp_path / "vector",
        predecessor_dir=tmp_path / "predecessor",
        output_dir=tmp_path / "results/test",
        predecessor_pid=42,
        max_wait_seconds=60,
        max_seconds=4260,
        resume_completed=False,
    )
    # Allow the full declared fit+replay allowance after small setup time.
    a.max_seconds = 5000
    entry.run(a)
    assert called == ["fit", "replay"]
    s = json.loads((a.output_dir / "batch_status.json").read_text())
    assert s["status"] == "COMPLETE" and len(s["completed"]) == s["planned"] == 1
