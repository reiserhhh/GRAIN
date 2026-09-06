#!/usr/bin/env python3
"""Re-signed corruption and bounded queue tests with synthetic, nonprivate caches."""

import copy
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from grain_research.vector_mechanisms.context import QUESTIONS
from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.external_material import ARMS, pack_features
from grain_research.vector_mechanisms.history_cache import verify_history_shard
from grain_research.vector_mechanisms.history_collection import (
    HISTORIES,
    KINDS,
    SITES,
    observation_signature,
)
from grain_research.vector_mechanisms.ready_comparison import digest


def test_history_archive_imports_without_unrelated_legacy_scoring_modules(tmp_path):
    """Execute entrypoints only from exported source, not the full development tree."""
    from grain_research.vector_mechanisms.history_instrument import archive_instrument

    archive = tmp_path / "source.zip"
    archive_instrument(Path(__file__).resolve().parents[1], archive)
    isolated = tmp_path / "isolated"
    with zipfile.ZipFile(archive) as z:
        z.extractall(isolated)
    assert not (isolated / "suica_core/v8_bridge.py").exists()
    for script in (
        "qualify_grain_source_history.py",
        "collect_grain_source_history.py",
        "run_grain_history_queue.py",
    ):
        p = subprocess.run(
            [sys.executable, "-I", str(isolated / "scripts" / script), "--help"],
            cwd=isolated,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        assert p.returncode == 0, p.stderr
        assert "--views" in p.stdout


def write(path, value):
    """Create only synthetic JSON fixtures."""
    path.write_text(json.dumps(value))


def seal(folder, manifest, summary):
    """Rehash fixture mutations to test content checks beyond checksum checks."""
    manifest["identity"] = material_hash(
        {k: v for k, v in manifest.items() if k != "identity"}
    )
    summary["parent_identity"] = manifest["identity"]
    summary["outputs_sha256"] = {
        p.relative_to(folder).as_posix(): digest(p)
        for p in folder.rglob("*")
        if p.is_file()
        and p.name
        not in (
            "manifest.json",
            "summary.json",
            "run_status.json",
            "source_snapshot.zip",
        )
    }
    write(folder / "manifest.json", manifest)
    write(folder / "summary.json", summary)
    write(folder / "run_status.json", summary)


def packet(folder, preflight=False):
    """One artificial author with every logical layout; no real tokenizer/model run."""
    folder.mkdir()
    (folder / "private_cache").mkdir()
    (folder / "source_snapshot.zip").write_bytes(b"synthetic archive")
    whole = [{"author_id": "fixture_person", "source_group": "mbti_only"}]
    material, audit, code = {"identity": "m"}, {"identity": "a"}, {"x": "code"}
    key = material_hash(whole[0]["author_id"])
    model = {"kind": "synthetic", "unix_time": 1}
    signature = None if preflight else observation_signature(model)
    manifest = {
        "protocol": "VM05_HISTORY_COLLECTION_V1",
        "mode": "preflight" if preflight else "source_shard",
        "material_identity": "m",
        "audit_identity": "a",
        "qualification_identity": None if preflight else "q",
        "technical_identity": None if preflight else "t",
        "strict_author_commitment": material_hash([whole[0]["author_id"]]),
        "strict_authors": 1,
        "authors": [whole[0]["author_id"]],
        "case_commitment": material_hash(whole),
        "shard_count": 1,
        "shard_index": 0,
        "code_sha256": code,
        "sites": SITES,
        "feature_rows": ["decision", "current_mean", "current_std"],
        "readout_kinds": KINDS,
        "diagnostic_margin_is_direct_trait_score": False,
        "labels_read": False,
        "bridge_outcomes_read": False,
        "source_snapshot_sha256": digest(folder / "source_snapshot.zip"),
        "max_cache_bytes": 1_000_000,
    }
    summary = {
        "status": "COMPLETE",
        "authors": 1,
        "strict_authors": 1,
        "token_layouts_replayed": 54,
        "real_neural_collection": not preflight,
        "qualified_state_replay_comparison_groups": 0,
        "qualified_state_replay_max": None,
        "model_signature": signature,
        "unique_layouts": 18,
        "forwards": 0 if preflight else 19,
        "cache_bytes": 0,
        "labels_read": False,
        "bridge_outcomes_read": False,
    }
    if preflight:
        records = [
            {
                "case_commitment": material_hash(whole[0]),
                "layouts": 54,
                "unique_layouts": 18,
            }
        ]
    else:
        arrays = {
            f"{a}__{t}__{h}__{s}": np.zeros((3, 3, 1536), dtype=np.float32)
            for a in ARMS
            for t in QUESTIONS
            for h in HISTORIES
            for s in SITES
        }
        stored, aliases = pack_features(arrays)
        npz = folder / "private_cache" / (key + ".npz")
        meta_path = npz.with_suffix(".json")
        np.savez_compressed(npz, **stored)
        meta = whole[0] | {
            "case_commitment": material_hash(whole[0]),
            "feature_sha256": digest(npz),
            "feature_aliases": aliases,
            "cache_format": "lossless_history_aliases_v1",
            "logical_layouts": 54,
            "unique_layouts": 18,
            "forwards": 19,
            "replay_max": 0,
            "causal_prefix_max": 0,
            "input_embedding_max": 0,
            "readouts": [
                {
                    "view": v,
                    "text_arm": a,
                    "task": t,
                    "history_arm": h,
                    "layout_sha256": material_hash([v, t, h]),
                    "yes_logit": 2.0,
                    "no_logit": 1.0,
                    "margin": 1.0,
                    "current_tokens": 3,
                    "visible_tokens": 8,
                    "prompt_tokens": 10,
                }
                for v in range(3)
                for a in ARMS
                for t in QUESTIONS
                for h in HISTORIES
            ],
        }
        write(meta_path, meta)
        records = [
            {
                "cache_key": key,
                "author_id": whole[0]["author_id"],
                "npz_sha256": digest(npz),
                "json_sha256": digest(meta_path),
                "unique_layouts": 18,
            }
        ]
        summary["cache_bytes"] = npz.stat().st_size + meta_path.stat().st_size
        write(folder / "actual_observer.json", model)
    write(folder / "private_records.json", records)
    seal(folder, manifest, summary)
    kw = {
        "qualification": None if preflight else "q",
        "technical": None if preflight else "t",
        "model_signature": signature,
        "allow_preflight": preflight,
    }
    return (folder, whole, material, audit, code, 1, 0), kw


def test_full_cache_and_preflight_have_distinct_acceptance(tmp_path):
    args, kw = packet(tmp_path / "neural")
    assert verify_history_shard(*args, **kw)["arrays"] == 90
    args, kw = packet(tmp_path / "tokens", preflight=True)
    assert not verify_history_shard(*args, **kw)["real_neural_collection"]
    with pytest.raises(ValueError, match="protocol"):
        verify_history_shard(*args, **(kw | {"allow_preflight": False}))


@pytest.mark.parametrize(
    "bad", ["role", "margin", "grid", "array", "untracked", "checksum", "status"]
)
def test_corrupted_content_is_not_accepted_by_resigning(tmp_path, bad):
    args, kw = packet(tmp_path / "data")
    folder = args[0]
    m = json.loads((folder / "manifest.json").read_text())
    s = json.loads((folder / "summary.json").read_text())
    records = json.loads((folder / "private_records.json").read_text())
    meta_path = next((folder / "private_cache").glob("*.json"))
    meta = json.loads(meta_path.read_text())
    if bad == "role":
        meta["source_group"] = "bridge"
    elif bad == "margin":
        meta["readouts"][0]["margin"] = 0
    elif bad == "grid":
        meta["readouts"][0] = copy.deepcopy(meta["readouts"][1])
    elif bad == "array":
        npz = meta_path.with_suffix(".npz")
        with np.load(npz) as z:
            stored = {k: z[k] for k in z.files}
        stored[next(iter(stored))] = np.zeros((3, 2, 1536), dtype=np.float32)
        np.savez_compressed(npz, **stored)
        meta["feature_sha256"] = records[0]["npz_sha256"] = digest(npz)
    elif bad == "untracked":
        (folder / "extra.txt").write_text("not declared")
    elif bad == "checksum":
        meta_path.write_text("changed")
        with pytest.raises(ValueError, match="checksum"):
            verify_history_shard(*args, **kw)
        return
    elif bad == "status":
        write(folder / "run_status.json", {"status": "RUNNING"})
        with pytest.raises(ValueError, match="status"):
            verify_history_shard(*args, **kw)
        return
    write(meta_path, meta)
    records[0]["json_sha256"] = digest(meta_path)
    s["cache_bytes"] = sum(
        p.stat().st_size for p in (folder / "private_cache").iterdir()
    )
    write(folder / "private_records.json", records)
    seal(folder, m, s)
    with pytest.raises(ValueError):
        verify_history_shard(*args, **kw)


def test_queue_pauses_before_a_child_without_partial_or_false_completion(
    tmp_path, monkeypatch
):
    from scripts import run_grain_history_queue as runner

    root = tmp_path
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "fingerprint", lambda _: {"fixed": "sha"})
    monkeypatch.setattr(
        runner,
        "registered_cases",
        lambda *a: (
            {},
            [{"author_id": "fixture"}],
            {"identity": "m"},
            {"identity": "a"},
        ),
    )
    monkeypatch.setattr(
        runner, "archive_instrument", lambda r, p: p.write_bytes(b"fixture")
    )
    monkeypatch.setattr(
        runner.subprocess, "Popen", lambda *a, **kw: pytest.fail("No child expected")
    )
    args = SimpleNamespace(
        max_seconds=60,
        shard_seconds=60,
        shard_cache_bytes=100,
        max_cache_bytes=1000,
        views=None,
        material_dir=None,
        audit_dir=None,
        shards=1,
        preflight_only=True,
        output_dir=root / "results" / "queue",
        resume_completed=False,
    )
    runner.run(args)
    status = json.loads((args.output_dir / "queue_status.json").read_text())
    assert (
        status["status"] == "PAUSED_BUDGET_BETWEEN_SHARDS" and not status["completed"]
    )
    assert not (args.output_dir / "queue.lock").exists()
    with pytest.raises(ValueError, match="identical explicit resume"):
        runner.run(args)
    args.resume_completed = True
    runner.run(args)
    (args.output_dir / "queue.lock").write_text("other-owned-fixture")
    with pytest.raises(FileExistsError):
        runner.run(args)
