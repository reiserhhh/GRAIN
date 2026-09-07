#!/usr/bin/env python3
"""Run the frozen direct queue with bounded, status-only Windows write retries."""

import argparse
import json
import os
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.ready_comparison import digest
from grain_research.vector_mechanisms.status_io_guard import guarded_writer
from scripts import run_grain_direct_traits_queue as queue

CODE = (
    "scripts/run_grain_direct_queue_io_guard.py",
    "grain_research/vector_mechanisms/status_io_guard.py",
)


def main():
    """Keep fresh/resume checks and collector arguments intact; journal outside output."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--resume-completed", action="store_true")
    a, _ = parser.parse_known_args()
    out = a.output_dir.resolve()
    manifest_path = out / "manifest.json"
    if not out.is_relative_to(ROOT / "results") or (out / "queue.lock").exists():
        raise ValueError("Private output and absent queue lock required")
    if a.resume_completed:
        if not manifest_path.is_file():
            raise ValueError("Explicit resume requires an existing frozen manifest")
    elif out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError("Fresh direct queue must not already contain outputs")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    # The original fresh queue rejects nonempty output directories. Keep the
    # operational journal beside it instead of weakening that scientific check.
    evidence = out.parent / (out.name + "_io_guard") / f"{stamp}_{os.getpid()}"
    evidence.mkdir(parents=True, exist_ok=False)
    prior = out / "queue_status.json"
    if prior.exists():
        with (evidence / "prior_queue_status.json").open("xb") as handle:
            handle.write(prior.read_bytes())
    record_manifest = {
        "protocol": "VM05_DIRECT_STATUS_IO_GUARD_V1",
        "pid": os.getpid(),
        "argv": sys.argv[1:],
        "output_dir": str(out),
        "explicit_resume": a.resume_completed,
        "original_manifest_sha256": digest(manifest_path)
        if manifest_path.exists()
        else None,
        "original_code_sha256": queue.fingerprint(queue.QUEUE_CODE),
        "operational_code_sha256": {n: digest(ROOT / n) for n in CODE},
        "scope": "status JSON retry only; original queue and collector unchanged",
        "maximum_retry_delay_seconds_per_write": 1.55,
        "model_weights_changed": False,
        "label_values_read_by_wrapper": False,
    }
    original = queue.write_json
    original(evidence / "manifest.json", record_manifest)
    with zipfile.ZipFile(evidence / "source_snapshot.zip", "x") as archive:
        for name in CODE:
            archive.write(ROOT / name, name)
    with (evidence / "events.jsonl").open("x", encoding="utf-8") as log:

        def record(event):
            """Write retry/exit evidence independently of the queue status replace."""
            log.write(json.dumps(event, allow_nan=False) + "\n")
            log.flush()

        queue.write_json = guarded_writer(original, out / "queue_status.json", record)
        try:
            queue.main()
            if (
                queue.fingerprint(queue.QUEUE_CODE)
                != record_manifest["original_code_sha256"]
            ):
                raise ValueError("Original direct queue code changed during execution")
            state = json.loads((out / "queue_status.json").read_text(encoding="utf-8"))
            record({"event": "queue_returned", "status": state["status"]})
            print(
                json.dumps({"io_guard": "RETURNED", "queue_status": state["status"]}),
                flush=True,
            )
        except BaseException as error:
            record({"event": "queue_failed", "exception_type": type(error).__name__})
            raise
        finally:
            queue.write_json = original


if __name__ == "__main__":
    main()
