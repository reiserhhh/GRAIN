#!/usr/bin/env python3
"""Secondary batch selection, source membership, acceptance and owned execution guards."""

import json
import zipfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from grain_research.vector_mechanisms.controls import material_hash
from scripts import run_grain_cached_readout_batch as entry


def test_full_secondary_family_excludes_duplicate_primary_and_margin_sites():
    rows = entry.cases()
    assert len(rows) == len({r["id"] for r in rows}) == 315
    assert {r["site"] for r in rows[:90]} == {"block_10"}
    assert "block_10" not in {r["site"] for r in rows[90:]}
    assert not any(r["readout"] == "body_token" for r in rows)
    margins = [r for r in rows if r["readout"] == "logit_margins"]
    assert len(margins) == 15 and {r["site"] for r in margins} == {"block_10"}
    for site in entry.SITES:
        for readout in entry.READOUTS:
            group = [r for r in rows if r["site"] == site and r["readout"] == readout]
            assert len(group) == 15
            assert {(r["task"], r["arm"]) for r in group} == {
                (t, a) for t in entry.TASKS for a in entry.ARMS
            }


def test_predecessor_requires_completed_family_and_dead_owner(tmp_path, monkeypatch):
    (tmp_path / "batch_manifest.json").write_text("{}")
    status = {"status": "FIT", "pid": 901}
    path = tmp_path / "batch_status.json"
    path.write_text(json.dumps(status))
    assert not entry.predecessor_done(tmp_path)
    status["status"] = "PAUSED_BUDGET"
    path.write_text(json.dumps(status))
    with pytest.raises(ValueError, match="needs completion"):
        entry.predecessor_done(tmp_path)
    calls = []
    monkeypatch.setattr(
        entry, "require_complete_batch", lambda m, s: calls.append(s["status"])
    )
    status["status"] = "COMPLETE"
    path.write_text(json.dumps(status))
    monkeypatch.setattr(
        entry.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=1, stdout=""),
    )
    assert entry.predecessor_done(tmp_path) and calls == ["COMPLETE"]
    monkeypatch.setattr(
        entry.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            returncode=0, stdout="python run_grain_ready_primary_batch.py"
        ),
    )
    assert not entry.predecessor_done(tmp_path)
    monkeypatch.setattr(
        entry.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout="unrelated process"),
    )
    with pytest.raises(ValueError, match="cannot be verified"):
        entry.predecessor_done(tmp_path)


def toy_members(tmp_path):
    """Include source, reference and bridge roles with no target values."""
    source = [
        {
            "author_id": f"toy_{i}",
            "source_group": "big5_only"
            if i < 10
            else "mbti_only"
            if i < 20
            else "bridge",
            "official_folds": {t: i % 5 for t in entry.TASKS},
        }
        for i in range(25)
    ]
    common = {
        "author_ids": [b["author_id"] for b in source],
        "material_sha256": material_hash(source),
    }
    parent = {"source_tables": {}}
    parent["identity"] = material_hash(parent)
    for name, value in (
        ("views.json", source),
        ("common_material_manifest.json", common),
        ("manifest.json", parent),
    ):
        (tmp_path / name).write_text(json.dumps(value))
    pd.DataFrame(
        {
            "author_id": common["author_ids"],
            "old_reference_member": [i == 0 for i in range(25)],
        }
    ).to_csv(tmp_path / "budget_by_author.csv", index=False)
    args = SimpleNamespace(
        views=tmp_path / "views.json", preflight_dir=tmp_path, parent_dir=tmp_path
    )
    return args, source, common, parent


def test_members_are_reference_free_exclusive_and_unchanged(tmp_path):
    a, source, common, parent = toy_members(tmp_path)
    members, actual_common, actual_parent = entry.source_members(a)
    assert actual_common == common and actual_parent == parent
    assert len(members["big5"]) == 9 and len(members["EI_cont"]) == 10
    assert all(b["source_group"] == "big5_only" for b in members["big5"])
    assert all(b["source_group"] == "mbti_only" for b in members["EI_cont"])
    assert not any(
        b["source_group"] == "bridge" for rows in members.values() for b in rows
    )
    source[0]["official_folds"]["big5"] = 4
    a.views.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="changed"):
        entry.source_members(a)


def test_result_requires_exact_readout_folds_and_saved_models(tmp_path, monkeypatch):
    c = {
        "task": "EI_cont",
        "arm": "raw",
        "site": "block_10",
        "readout": "body_equal_view",
        "id": "toy",
    }
    source = [
        {
            "author_id": str(i),
            "source_group": "mbti_only",
            "official_folds": {"EI_cont": i},
        }
        for i in range(5)
    ]
    ids, folds = [str(i) for i in range(5)], list(range(5))
    m = {
        "protocol": "VM05_FULL_SOURCE_READOUT_V1",
        "task": "EI_cont",
        "feature": "qwen_block_10",
        "readout": "body_equal_view",
        "arm": "raw",
        "source_group": "mbti_only",
        "authors": 5,
        "source_material_sha256": material_hash(source),
        "source_author_commitment": material_hash(ids),
        "fold_commitment": material_hash(folds),
        "source_table_parent_identity": "parent",
        "common_material_sha256": "common",
        "budget_sha256": "budget",
        "seed": 731,
        "pca": True,
        "pca_applied": True,
        "exclude_reference": True,
        "thread_control": "retain",
        "drop_seed": 731,
        "permutation_seed": None,
        "bridge_labels_used": False,
        "preflight_only": False,
        "inner_cv": "3 author folds, seed731",
        "outer_cv": "5 source-specific official folds",
        "calibration_validated": False,
        "scoring_code": {},
        "identity": "toy",
    }
    head = tmp_path / "private_full_source_head.joblib"
    head.write_bytes(b"synthetic")
    s = {
        "authors": 5,
        "full_train_authors": 5,
        "bridge_labels_used": False,
        "neural_forwards": 0,
        "exclude_reference": True,
        "permutation_seed": None,
        "saved_fold_replay_max": 0.0,
        "full_head_sha256": entry.digest(head),
        "score": 0.5,
    }
    packet = {"ids": ids, "folds": np.array(folds)}
    monkeypatch.setattr(entry, "verified_packet", lambda _: (m, s))
    monkeypatch.setattr(entry, "load_oof", lambda _: packet)
    record = pd.DataFrame({"fold": folds, "max_absolute_error": [0.0] * 5})
    record.to_csv(tmp_path / "saved_head_replay.csv", index=False)
    args = (
        tmp_path,
        c,
        {"EI_cont": source},
        {"material_sha256": "common"},
        {"identity": "parent"},
        "budget",
    )
    assert entry.validate_result(*args)["replayed_folds"] == 5
    m["readout"] = "body_token"
    with pytest.raises(ValueError, match="contract"):
        entry.validate_result(*args)
    m["readout"] = c["readout"]
    packet["folds"] = np.roll(packet["folds"], 1)
    with pytest.raises(ValueError, match="contract"):
        entry.validate_result(*args)
    packet["folds"] = np.array(folds)
    record.loc[0, "max_absolute_error"] = float("nan")
    record.to_csv(tmp_path / "saved_head_replay.csv", index=False)
    with pytest.raises(ValueError, match="contract"):
        entry.validate_result(*args)


def test_queue_bounds_fits_and_preserves_partial_outputs(tmp_path, monkeypatch):
    folders = (
        "preflight_dir",
        "parent_dir",
        "vector_dir",
        "paired_dir",
        *entry.LEASE_FLAGS,
    )
    for name in folders:
        folder = tmp_path / name
        folder.mkdir()
        for file in (
            "manifest.json",
            "batch_manifest.json",
            "summary.json",
            "common_material_manifest.json",
            "budget_by_author.csv",
        ):
            (folder / file).write_text('{"identity":"ready"}')
    views = tmp_path / "views.json"
    views.write_text("[]")
    monkeypatch.setattr(entry, "ROOT", tmp_path)
    monkeypatch.setattr(entry, "CODE", ())
    monkeypatch.setattr(entry, "scoring_commitment", dict)
    monkeypatch.setattr(
        entry,
        "verified_packet",
        lambda _: (
            {
                "protocol": "VM05_RAW_READY_PAIRED_COMPARISON_V1",
                "batch_identity": "ready",
                "identity": "paired",
            },
            {
                "model_refits": 0,
                "bridge_labels_read": False,
                "new_source_heads_accepted": 34,
                "historical_source_heads_reused": 1,
                "saved_folds_accepted": 175,
                "paired_rows": 90,
            },
        ),
    )
    monkeypatch.setattr(entry, "source_members", lambda a: ({"big5": []}, {}, {}))
    monkeypatch.setattr(entry, "predecessor_done", lambda _: True)
    one = entry.cases()[0]
    monkeypatch.setattr(entry, "cases", lambda: [one])

    def archive(root, path):
        with zipfile.ZipFile(path, "x"):
            pass

    monkeypatch.setattr(entry, "archive_source", archive)
    calls = []

    def child(cmd, log, seconds, rss, update):
        assert seconds == 3600 and rss == 8 * 1024**3
        assert cmd[-2:] == ["--exclude-reference", "--worker"]
        assert cmd[cmd.index("--site") + 1] == "block_10"
        assert cmd[cmd.index("--readout") + 1] != "body_token"
        calls.append(cmd)
        (entry.Path(cmd[cmd.index("--output-dir") + 1]) / "fit").mkdir()
        update(123, 0.01, 100, 100)
        return {"seconds": 0.01}

    monkeypatch.setattr(entry, "bounded_child", child)
    monkeypatch.setattr(
        entry, "validate_result", lambda *a: {"case": one["id"], "score": 0.0}
    )
    a = SimpleNamespace(
        **{n: tmp_path / n for n in folders},
        views=views,
        prepared_root=tmp_path,
        output_dir=tmp_path / "results/run",
        max_seconds=4000,
        max_wait_seconds=60,
        resume_completed=False,
    )
    entry.run(a)
    assert len(calls) == 1
    with pytest.raises(ValueError, match="explicit resume"):
        entry.run(a)
    a.resume_completed = True
    entry.run(a)
    assert len(calls) == 1

    def incomplete(*args):
        raise ValueError("Partial fit evidence must be preserved")

    monkeypatch.setattr(entry, "validate_result", incomplete)
    with pytest.raises(ValueError, match="preserved"):
        entry.run(a)
    assert len(calls) == 1 and (a.output_dir / one["id"] / "fit").exists()
