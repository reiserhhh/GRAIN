#!/usr/bin/env python3
"""The direct wrapper must not change payloads, collector arguments or fresh checks."""

import json

import pytest

from scripts import run_grain_direct_queue_io_guard as entry


def configure(tmp_path, monkeypatch, *, resume=False):
    """Use a non-neural queue double with private journal/source paths."""
    monkeypatch.setattr(entry, "ROOT", tmp_path)
    out = tmp_path / "results" / "direct"
    for name in entry.CODE:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# synthetic source\n")
    argv = [
        "wrapper",
        "--output-dir",
        str(out),
        "--preflight-only",
        "--max-seconds",
        "60",
    ]
    if resume:
        out.mkdir(parents=True)
        (out / "manifest.json").write_text('{"frozen": true}\n')
        (out / "queue_status.json").write_text(
            '{"status": "PAUSED_BUDGET_BETWEEN_SHARDS"}\n'
        )
        argv.append("--resume-completed")
    monkeypatch.setattr(entry.sys, "argv", argv)
    monkeypatch.setattr(
        entry.queue, "fingerprint", lambda names: {"frozen_queue": "unchanged"}
    )
    return out, argv


def json_writer(path, value):
    """Write test payloads without pretending a model ran."""
    path.write_text(json.dumps(value) + "\n")


def test_fresh_retries_exact_status_and_restores_writer(tmp_path, monkeypatch):
    out, argv = configure(tmp_path, monkeypatch)
    attempts, calls = [], []
    payload = {"status": "PAUSED_BUDGET_BETWEEN_SHARDS", "completed": []}

    def original(path, value):
        if path == out / "queue_status.json":
            attempts.append((path, value))
            if len(attempts) < 3:
                error = PermissionError("synthetic sharing collision")
                error.winerror = 5
                raise error
        json_writer(path, value)

    def main():
        calls.append(list(entry.sys.argv))
        assert not out.exists()
        out.mkdir()
        entry.queue.write_json(out / "manifest.json", {"frozen": True})
        entry.queue.write_json(out / "queue_status.json", payload)

    monkeypatch.setattr(entry.queue, "write_json", original)
    monkeypatch.setattr(entry.queue, "main", main)
    entry.main()
    assert calls == [argv] and entry.queue.write_json is original
    assert len(attempts) == 3 and all(value is payload for _, value in attempts)
    assert json.loads((out / "queue_status.json").read_text()) == payload
    journal = next((out.parent / "direct_io_guard").iterdir())
    events = [
        json.loads(s) for s in (journal / "events.jsonl").read_text().splitlines()
    ]
    assert [r["event"] for r in events] == [
        "retry_status_write",
        "retry_status_write",
        "queue_returned",
    ]
    assert (
        json.loads((journal / "manifest.json").read_text())["original_manifest_sha256"]
        is None
    )


def test_resume_preserves_prior_and_failure_restores_original(tmp_path, monkeypatch):
    out, _ = configure(tmp_path, monkeypatch, resume=True)
    old = (out / "queue_status.json").read_bytes()
    failure = RuntimeError("original queue rejected a changed source")

    def main():
        raise failure

    monkeypatch.setattr(entry.queue, "write_json", json_writer)
    monkeypatch.setattr(entry.queue, "main", main)
    with pytest.raises(RuntimeError) as caught:
        entry.main()
    assert caught.value is failure and entry.queue.write_json is json_writer
    assert (out / "queue_status.json").read_bytes() == old
    journal = next((out.parent / "direct_io_guard").iterdir())
    assert (journal / "prior_queue_status.json").read_bytes() == old
    assert json.loads((journal / "events.jsonl").read_text())["event"] == "queue_failed"


@pytest.mark.parametrize("reason", ["lock", "nonempty", "resume_missing", "outside"])
def test_refusal_precedes_original_queue(tmp_path, monkeypatch, reason):
    out, argv = configure(tmp_path, monkeypatch)
    if reason in ("lock", "nonempty"):
        out.mkdir(parents=True)
        (out / ("queue.lock" if reason == "lock" else "partial.txt")).write_text(
            "preserve"
        )
    elif reason == "resume_missing":
        argv.append("--resume-completed")
    else:
        argv[2] = str(tmp_path / "outside")
    monkeypatch.setattr(
        entry.queue, "main", lambda: pytest.fail("Refusal happened too late")
    )
    with pytest.raises(ValueError):
        entry.main()
    assert not (out.parent / "direct_io_guard").exists()
