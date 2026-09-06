#!/usr/bin/env python3
"""Raw ready-stage identity, lease handoff and execution orchestration tests."""

import json
import zipfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from grain_research.vector_mechanisms.controls import material_hash
from scripts import run_grain_ready_primary_batch as entry


def test_exact_raw_stage_has_no_duplicate_big5_profile():
    rows = entry.cases()
    assert len(rows) == len({r["id"] for r in rows}) == 34
    assert not any(
        r["task"] == "big5" and r["feature"] == "sorted_distances" for r in rows
    )
    for task in entry.TASKS:
        features = {r["feature"] for r in rows if r["task"] == task}
        assert features == set(entry.FEATURES) | (
            {"sorted_distances"} if task != "big5" else set()
        )


def test_completion_requires_dead_owner_full_family_and_finite_replays(
    tmp_path, monkeypatch
):
    path = tmp_path / "batch_status.json"
    s = {"status": "FIT", "pid": 901, "child_pid": 902}
    path.write_text(json.dumps(s))
    assert not entry.predecessor_done(tmp_path)
    s["status"] = "PAUSED_BUDGET"
    path.write_text(json.dumps(s))
    with pytest.raises(ValueError, match="completion"):
        entry.predecessor_done(tmp_path)
    s.update(
        status="COMPLETE",
        child_pid=None,
        completed=[
            {"case": c["id"], "replayed_folds": 5, "replay_max": 0.0}
            for c in entry.sensitivity_cases()
        ],
    )
    path.write_text(json.dumps(s))
    monkeypatch.setattr(
        entry.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=1, stdout=""),
    )
    assert entry.predecessor_done(tmp_path)
    monkeypatch.setattr(
        entry.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            returncode=0, stdout="python scripts/run_grain_neural_sensitivity_batch.py"
        ),
    )
    assert not entry.predecessor_done(tmp_path)
    s["completed"][0]["replay_max"] = float("nan")
    path.write_text(json.dumps(s))
    with pytest.raises(ValueError, match="fit/replay"):
        entry.predecessor_done(tmp_path)


def test_ready_acceptance_checks_projection_authors_and_replay(tmp_path, monkeypatch):
    case = {"task": "EI_cont", "feature": "qwen_block_10", "id": "test"}
    source = [
        {
            "author_id": str(i),
            "source_group": "mbti_only",
            "official_folds": {"EI_cont": i},
        }
        for i in range(5)
    ]
    m = {
        "protocol": "VM05_SAME_READY_COMPARISON_V1",
        "cohort": entry.COHORT,
        "task": "EI_cont",
        "feature": "qwen_block_10",
        "arm": "raw",
        "registry_identity": "registry",
        "source_material_sha256": material_hash(source),
        "common_material_sha256": "common",
        "scaler_fit": "inner_source_only",
        "pca": True,
        "pca_applied": True,
        "thread_control": "retain",
        "seed": 731,
        "permutation_seed": None,
        "bridge_labels_used": False,
        "preflight_only": False,
        "separate_calibration_fitted": False,
        "readout": "body_token",
        "identity": "toy",
    }
    head = tmp_path / "private_full_source_head.joblib"
    head.write_bytes(b"synthetic stand-in")
    s = {
        "bridge_labels_used": False,
        "saved_fold_replay_max": 0.0,
        "full_head_sha256": entry.digest(head),
        "score": 0.5,
    }
    packet = {
        "manifest": m,
        "summary": s,
        "ids": np.array([str(i) for i in range(5)]),
        "folds": np.arange(5),
    }
    monkeypatch.setattr(entry, "verified_packet", lambda _: (m, s))
    monkeypatch.setattr(entry, "load_oof", lambda _: packet)
    monkeypatch.setattr(entry, "source_indices", lambda *a: np.arange(5))
    pd.DataFrame({"fold": range(5), "max_absolute_error": [0.0] * 5}).to_csv(
        tmp_path / "saved_head_replay.csv", index=False
    )
    args = (
        tmp_path,
        case,
        source,
        {"material_sha256": "common"},
        {},
        {"identity": "registry"},
    )
    assert entry.validate_ready(*args)["authors"] == 5
    m["pca_applied"] = False
    with pytest.raises(ValueError, match="contract"):
        entry.validate_ready(*args)
    m["pca_applied"] = True
    packet["folds"] = np.roll(packet["folds"], 1)
    with pytest.raises(ValueError, match="contract"):
        entry.validate_ready(*args)


def test_bounded_child_wiring_and_resume_do_not_refit(tmp_path, monkeypatch):
    folders = [n for n in entry.INPUT_FLAGS if n not in ("views", "prepared_root")]
    folders += ["predecessor_dir", "primary_dir", "resource_dir", "big5_profile_parent"]
    for n in folders:
        (tmp_path / n).mkdir()
        for f in ("manifest.json", "batch_manifest.json", "summary.json"):
            (tmp_path / n / f).write_text("{}")
    for f in ["private_oof_predictions.csv", "private_full_source_head.joblib"] + [
        f"private_fold_{i}.joblib" for i in range(5)
    ]:
        (tmp_path / "big5_profile_parent" / f).write_text("synthetic")
    views = tmp_path / "views.json"
    views.write_text("[]")
    pred = {
        "protocol": "VM05_FROZEN_NEURAL_SENSITIVITIES_V1",
        "cases": entry.sensitivity_cases(),
    }
    pred["identity"] = material_hash(pred)
    (tmp_path / "predecessor_dir/batch_manifest.json").write_text(json.dumps(pred))
    monkeypatch.setattr(entry, "ROOT", tmp_path)
    monkeypatch.setattr(entry, "CODE", ())
    monkeypatch.setattr(entry, "scoring_commitment", dict)
    monkeypatch.setattr(
        entry, "verified_packet", lambda _: ({"identity": "registry"}, {})
    )
    monkeypatch.setattr(
        entry,
        "load_registry_inputs",
        lambda *a: ([], {}, {}, {}, {"identity": "registry"}),
    )
    monkeypatch.setattr(entry, "reuse_big5_profile", lambda *a: {"new_fits": 0})
    monkeypatch.setattr(entry, "predecessor_done", lambda _: True)
    one = entry.cases()[0]
    monkeypatch.setattr(entry, "cases", lambda: [one])
    calls = []

    def archive(root, path):
        with zipfile.ZipFile(path, "x"):
            pass

    monkeypatch.setattr(entry, "archive_source", archive)

    def child(cmd, log, seconds, rss, update):
        assert seconds == 3600 and rss == 8 * 1024**3
        assert cmd[-1] == "--worker" and "--no-pca" not in cmd
        assert cmd[cmd.index("--cohort") + 1] == entry.COHORT
        calls.append(cmd)
        out = entry.Path(cmd[cmd.index("--output-dir") + 1])
        (out / "fit").mkdir()
        update(987, 0.01, 100, 100)
        return {"seconds": 0.01, "peak_sampled_rss_bytes": 100}

    monkeypatch.setattr(entry, "bounded_child", child)
    monkeypatch.setattr(
        entry, "validate_ready", lambda *a: {"case": one["id"], "score": 0.0}
    )
    opts = {n: tmp_path / n for n in folders}
    opts.update(
        views=views,
        prepared_root=tmp_path,
        output_dir=tmp_path / "results/run",
        max_seconds=4000,
        max_wait_seconds=60,
        resume_completed=False,
    )
    a = SimpleNamespace(**opts)
    entry.run(a)
    assert len(calls) == 1
    with pytest.raises(ValueError, match="explicit resume"):
        entry.run(a)
    a.resume_completed = True
    entry.run(a)
    assert len(calls) == 1
    s = json.loads((a.output_dir / "batch_status.json").read_text())
    assert s["status"] == "COMPLETE" and not s["program_complete"]
    s["child_pid"] = 987
    (a.output_dir / "batch_status.json").write_text(json.dumps(s))
    monkeypatch.setattr(
        entry.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout="987"),
    )
    with pytest.raises(ValueError, match="Previous owned child"):
        entry.run(a)
    assert len(calls) == 1
    assert json.loads((a.output_dir / "batch_status.json").read_text()) == s
