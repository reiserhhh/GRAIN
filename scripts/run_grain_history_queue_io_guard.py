#!/usr/bin/env python3
"""Resume the unchanged history queue with recorded status-only Windows I/O retry."""

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
from scripts import run_grain_history_queue as queue

CODE = (
    "scripts/run_grain_history_queue_io_guard.py",
    "grain_research/vector_mechanisms/status_io_guard.py",
)


def main():
    """Archive wrapper/provenance, preserve prior status and delegate identical arguments."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-completed", action="store_true")
    a, _ = parser.parse_known_args()
    out = a.output_dir.resolve()
    if (
        not a.resume_completed
        or not out.is_relative_to(ROOT / "results")
        or not (out / "manifest.json").is_file()
        or (out / "queue.lock").exists()
    ):
        raise ValueError(
            "Guard requires existing manifest, explicit resume and no queue lock"
        )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    evidence = out / "io_guard_invocations" / f"{stamp}_{os.getpid()}"
    evidence.mkdir(parents=True, exist_ok=False)
    prior = out / "queue_status.json"
    if prior.exists():
        with (evidence / "prior_queue_status.json").open("xb") as f:
            f.write(prior.read_bytes())
    manifest = {
        "protocol": "VM05_HISTORY_STATUS_IO_GUARD_V1",
        "pid": os.getpid(),
        "argv": sys.argv[1:],
        "original_manifest_sha256": digest(out / "manifest.json"),
        "original_code_sha256": queue.fingerprint(queue.QUEUE_CODE),
        "operational_code_sha256": {n: digest(ROOT / n) for n in CODE},
        "scope": "status JSON retry only; collector, cases, arrays and metrics unchanged",
        "maximum_retry_delay_seconds_per_write": 1.55,
        "model_weights_changed": False,
        "label_values_read_by_wrapper": False,
    }
    original = queue.write_json
    original(evidence / "manifest.json", manifest)
    with zipfile.ZipFile(evidence / "source_snapshot.zip", "x") as z:
        for n in CODE:
            z.write(ROOT / n, n)
    with (evidence / "events.jsonl").open("x", encoding="utf-8") as log:

        def record(event):
            """Persist retry and exit events in a separate append-only journal."""
            log.write(json.dumps(event, allow_nan=False) + "\n")
            log.flush()

        queue.write_json = guarded_writer(original, out / "queue_status.json", record)
        try:
            queue.main()
            if queue.fingerprint(queue.QUEUE_CODE) != manifest["original_code_sha256"]:
                raise ValueError(
                    "Original history code changed during guarded execution"
                )
            status = json.loads((out / "queue_status.json").read_text(encoding="utf-8"))
            record({"event": "queue_returned", "status": status["status"]})
            print(
                json.dumps({"io_guard": "RETURNED", "queue_status": status["status"]}),
                flush=True,
            )
        except BaseException as exc:
            record({"event": "queue_failed", "exception_type": type(exc).__name__})
            raise
        finally:
            queue.write_json = original


if __name__ == "__main__":
    main()
