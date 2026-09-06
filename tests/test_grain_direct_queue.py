#!/usr/bin/env python3
"""Complete-only direct queue, output corruption and semantic-parent boundaries."""

import json
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.direct_material import archive_direct
from grain_research.vector_mechanisms.direct_queue import semantic_parent, verify_shard
from grain_research.vector_mechanisms.direct_semantics import panel, prepare, summarize
from grain_research.vector_mechanisms.direct_traits import native_readout
from grain_research.vector_mechanisms.history_collection import observation_signature
from grain_research.vector_mechanisms.ready_comparison import digest
from tests.test_grain_direct_collection import Observer, author_packet
from tests.test_grain_direct_traits import Tokenizer
from tests.test_grain_history_cache import seal, write


def fake_collection_environment(tmp_path, monkeypatch):
    """Real collector/queue functions with invented texts and no model loaded."""
    from scripts import collect_grain_direct_traits as collector
    from scripts import run_grain_direct_traits_queue as queue

    fixtures = [
        author_packet(g, str(i))
        for i, g in enumerate(("big5_only", "mbti_only", "mbti_only"))
    ]
    whole = [c for _, c, _ in fixtures]
    by_author = {c["author_id"]: arms for arms, c, _ in fixtures}
    material = {
        "identity": "synthetic",
        "alphabets": fixtures[0][2]["metrics"]["alphabets"]
        | fixtures[-1][2]["metrics"]["alphabets"],
    }
    for name in queue.QUEUE_CODE:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# fixture code\n", encoding="utf-8")
    for entry in (queue, collector):
        monkeypatch.setattr(entry, "ROOT", tmp_path)
        monkeypatch.setattr(
            entry, "load_registration", lambda *args: (by_author, whole, material)
        )
        monkeypatch.setattr(entry, "verify_tokenizer", lambda *args: None)
        monkeypatch.setattr(
            entry,
            "archive_direct",
            lambda root, out: out.write_bytes(b"fixture source"),
        )
    monkeypatch.setattr(
        collector.AutoTokenizer, "from_pretrained", lambda *a, **k: Tokenizer()
    )
    monkeypatch.setattr(collector, "bundle_arms", lambda b: b)
    clock = [0.0]
    monkeypatch.setattr(queue.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(queue.time, "sleep", lambda n: None)

    class Child:
        """Execute the actual token-only collector synchronously in a fake subprocess."""

        pid = 999999
        returncode = 0

        def __init__(self, cmd, **kwargs):
            options = dict(zip(cmd[3::2], cmd[4::2]))
            args = SimpleNamespace(
                qualification=False,
                preflight_only=True,
                qualification_dir=None,
                observer_queue_dir=None,
                shard_count=int(options["--shard-count"]),
                shard_index=int(options["--shard-index"]),
                max_seconds=int(options["--max-seconds"]),
                max_cache_bytes=int(options["--max-cache-bytes"]),
            )
            for key in ("views", "material_dir", "tokenizer_dir", "output_dir"):
                setattr(args, key, Path(options["--" + key.replace("_", "-")]))
            collector.run(args)
            clock[0] += 115

        def poll(self):
            return 0

    monkeypatch.setattr(queue.subprocess, "Popen", Child)
    args = SimpleNamespace(
        views=tmp_path / "unused",
        material_dir=tmp_path / "unused",
        tokenizer_dir=tmp_path,
        output_dir=tmp_path / "results/queue",
        qualification_dir=None,
        semantic_dir=None,
        observer_queue_dir=None,
        history_queue_dir=None,
        preflight_only=True,
        resume_completed=False,
        shards=3,
        max_seconds=200,
        shard_seconds=60,
        shard_cache_bytes=200_000_000,
        max_cache_bytes=6_000_000_000,
    )
    return queue, args, clock, whole, material


def test_budget_pause_and_identical_resume_verify_all_authors(tmp_path, monkeypatch):
    queue, a, clock, whole, material = fake_collection_environment(
        tmp_path, monkeypatch
    )
    queue.run(a)
    status = json.loads((a.output_dir / "queue_status.json").read_text())
    assert status["status"] == "PAUSED_BUDGET_BETWEEN_SHARDS"
    assert len(status["completed"]) == 1 and status["verified_authors"] == 1
    assert not (a.output_dir / "queue.lock").exists()
    first_sha = digest(a.output_dir / "shard_000/summary.json")
    a.resume_completed = True
    a.max_seconds = 1000
    clock[0] = 0
    queue.run(a)
    status = json.loads((a.output_dir / "queue_status.json").read_text())
    assert status["status"] == "COMPLETE" and status["verified_authors"] == 3
    assert all(not r["real_neural_collection"] for r in status["completed"])
    assert digest(a.output_dir / "shard_000/summary.json") == first_sha
    with pytest.raises(ValueError, match="mismatch"):
        verify_shard(
            a.output_dir / "shard_000",
            whole,
            material,
            queue.fingerprint(queue.CODE),
            3,
            0,
            preflight=False,
            qualification="real",
            model_signature="real",
        )
    a.shards = 2
    with pytest.raises(ValueError, match="identical"):
        queue.run(a)


def test_resigned_partial_shard_is_not_overwritten_on_resume(tmp_path, monkeypatch):
    queue, a, clock, _, _ = fake_collection_environment(tmp_path, monkeypatch)
    queue.run(a)
    folder = a.output_dir / "shard_000"
    m = json.loads((folder / "manifest.json").read_text())
    s = json.loads((folder / "summary.json").read_text())
    records = json.loads((folder / "private_records.json").read_text())
    records[0]["token_replay"] = False
    write(folder / "private_records.json", records)
    seal(folder, m, s)
    corrupted = digest(folder / "private_records.json")
    a.resume_completed = True
    a.max_seconds = 1000
    clock[0] = 0
    with pytest.raises(ValueError, match="token-only"):
        queue.run(a)
    assert digest(folder / "private_records.json") == corrupted
    assert (
        json.loads((a.output_dir / "queue_status.json").read_text())["status"]
        == "FAILED_STOP"
    )
    assert not (a.output_dir / "shard_001").exists()


def test_active_queue_lock_survives_rejected_second_owner(tmp_path, monkeypatch):
    queue, a, _clock, _, _ = fake_collection_environment(tmp_path, monkeypatch)
    queue.run(a)
    lock = a.output_dir / "queue.lock"
    lock.write_text("other live owner")
    a.resume_completed = True
    with pytest.raises(FileExistsError):
        queue.run(a)
    assert lock.read_text() == "other live owner"


def test_real_shard_checks_native_replay_counts_and_cache_bytes(tmp_path, monkeypatch):
    """Run the source collector with synthetic logits and test the full shard verifier."""
    from scripts import collect_grain_direct_traits as collector

    queue, a, _clock, whole, material = fake_collection_environment(
        tmp_path, monkeypatch
    )
    model = {"revision": "fixture", "dtype": "torch.bfloat16", "device": "cuda"}
    signature = observation_signature(model)

    class SyntheticObserver(Observer):
        def manifest(self):
            return model

    monkeypatch.setattr(collector, "load_observer", lambda *args: SyntheticObserver())
    monkeypatch.setattr(
        collector,
        "qualified_parent",
        lambda *args: ({"identity": "q"}, {"model_signature": signature}),
    )
    a.output_dir = tmp_path / "results/real_fixture"
    a.qualification = False
    a.preflight_only = False
    a.shard_count = 3
    a.shard_index = 1
    a.max_cache_bytes = 200_000_000
    collector.run(a)
    code = queue.fingerprint(collector.CODE)
    result = verify_shard(
        a.output_dir,
        whole,
        material,
        code,
        3,
        1,
        preflight=False,
        qualification="q",
        model_signature=signature,
    )
    assert result["real_neural_collection"] and result["authors"] == 1
    assert result["forwards"] > 1 and result["max_arithmetic_error"] < 1e-12
    m = json.loads((a.output_dir / "manifest.json").read_text())
    s = json.loads((a.output_dir / "summary.json").read_text())
    s["cache_bytes"] += 1
    seal(a.output_dir, m, s)
    with pytest.raises(ValueError, match="cache size"):
        verify_shard(
            a.output_dir,
            whole,
            material,
            code,
            3,
            1,
            preflight=False,
            qualification="q",
            model_signature=signature,
        )


def semantic_fixture(tmp_path):
    """All90 invented rows can be semantically flat yet numerically qualified."""
    folder = tmp_path / "semantic"
    folder.mkdir()
    (folder / "source_snapshot.zip").write_bytes(b"fixture source")
    tok = Tokenizer()
    rows = []
    for case in panel():
        ids, alphabet = prepare(case, tok)
        logits = np.zeros(128)
        rows.append(
            case
            | {
                "prompt_sha256": material_hash(ids),
                "prompt_tokens": len(ids),
                "alphabet": alphabet,
                **native_readout(logits, alphabet),
            }
        )
    contrasts, tasks = summarize(rows)
    model = {"kind": "fixture", "revision": "fixture"}
    signature = observation_signature(model)
    for name, data in (
        ("observations", rows),
        ("semantic_contrasts", contrasts),
        ("task_sensitivity", tasks),
        ("actual_observer", model),
    ):
        write(folder / (name + ".json"), data)
    m = {
        "protocol": "VM05_DIRECT_SEMANTICS_V1",
        "preflight_only": False,
        "panel_sha256": material_hash(panel()),
        "material_identity": "m",
        "qualification_identity": "q",
        "qualified_model_signature": signature,
        "code_sha256": {"x": "y"},
        "source_snapshot_sha256": digest(folder / "source_snapshot.zip"),
        "labels_read": False,
        "bridge_outcomes_read": False,
    }
    s = {
        "status": "COMPLETE",
        "real_neural_collection": True,
        "cases": 90,
        "forwards": 180,
        "labels_read": False,
        "bridge_outcomes_read": False,
    }
    seal(folder, m, s)
    return folder, m, s, signature, tok


def test_semantic_parent_preserves_nulls_but_rejects_bad_arithmetic_and_token_only(
    tmp_path,
):
    folder, m, s, signature, tok = semantic_fixture(tmp_path)
    result = semantic_parent(folder, {"identity": "m"}, "q", signature, {"x": "y"}, tok)
    assert result["positive_pairs_not_a_gate"]
    assert all(t["primary_positive_pairs"] == 0 for t in result["task_sensitivity"])
    rows = json.loads((folder / "observations.json").read_text())
    rows[0]["conditional_probabilities"][0] += 0.1
    write(folder / "observations.json", rows)
    seal(folder, m, s)
    with pytest.raises(ValueError, match="arithmetic"):
        semantic_parent(folder, {"identity": "m"}, "q", signature, {"x": "y"}, tok)
    s["real_neural_collection"] = False
    seal(folder, m, s)
    with pytest.raises(ValueError, match="incomplete"):
        semantic_parent(folder, {"identity": "m"}, "q", signature, {"x": "y"}, tok)


def test_queue_archive_runs_without_full_development_checkout(tmp_path):
    path = tmp_path / "source.zip"
    archive_direct(Path(__file__).resolve().parents[1], path)
    isolated = tmp_path / "isolated"
    with zipfile.ZipFile(path) as z:
        z.extractall(isolated)
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            str(isolated / "scripts/run_grain_direct_traits_queue.py"),
            "--help",
        ],
        cwd=isolated,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    assert b"--semantic-dir" in result.stdout
