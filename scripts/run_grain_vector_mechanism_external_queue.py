#!/usr/bin/env python3
"""Run serial VM05 CUDA shards with bounded time/disk and immutable checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.external_material import shard_material
from grain_research.vector_mechanisms.inventory import archive_source, code_fingerprint
from scripts.run_grain_vector_mechanism_controls import write_json

STAGES = ("bge_large", "e5_large", "qwen")
TECHNICAL_DIRS = dict(
    zip(
        STAGES,
        ("vm05_technical_bge_r1", "vm05_technical_e5_r1", "vm05_technical_qwen_r1"),
    )
)


def verify_completed(folder, source, common, stage, count, index, fingerprint):
    """Reuse only fully verified shard outputs, never an unverified status flag."""
    read = lambda name: json.loads((folder / name).read_text(encoding="utf-8"))
    status, manifest = read("run_status.json"), read("manifest.json")
    expected = shard_material(source, count, index)
    expected_ids = [b["author_id"] for b in expected]
    if (
        status["status"] != "COMPLETE"
        or status["authors"] != len(expected)
        or status["parent_identity"] != manifest["identity"]
        or manifest["authors"] != expected_ids
        or manifest["stage"] != stage
        or manifest["shard_count"] != count
        or manifest["shard_index"] != index
        or manifest["code_sha256"] != fingerprint
        or manifest["full_material_sha256"] != common["material_sha256"]
        or manifest["source_material_sha256"] != material_hash(expected)
    ):
        raise ValueError("Incomplete or different shard; preserve it, do not overwrite")
    if (
        material_hash({k: v for k, v in manifest.items() if k != "identity"})
        != manifest["identity"]
    ):
        raise ValueError("Shard manifest hash mismatch")
    cache = folder / "private_cache"
    if {p.stem for p in cache.glob("*.npz") if not p.name.startswith("._")} != set(
        expected_ids
    ):
        raise ValueError("Shard author cache set mismatch")
    for uid in expected_ids:
        meta = json.loads((cache / (uid + ".json")).read_text(encoding="utf-8"))
        if (
            meta["author_id"] != uid
            or hashlib.sha256((cache / (uid + ".npz")).read_bytes()).hexdigest()
            != meta["feature_sha256"]
        ):
            raise ValueError("Shard feature checksum mismatch")
    return status


def cache_size(root):
    """Count only result caches across all stages against one common disk budget."""
    return sum(
        p.stat().st_size for p in root.glob("*/shard_*/private_cache/*") if p.is_file()
    )


def main():
    """Resume completed shards explicitly; stop on the first actual error."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--views", type=Path, required=True)
    p.add_argument("--preflight-manifest", type=Path, required=True)
    p.add_argument("--qualification", type=Path, required=True)
    p.add_argument("--technical-root", type=Path, required=True)
    p.add_argument("--e5-model-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--shards", type=int, default=24)
    p.add_argument("--max-seconds", type=int, default=28800)
    p.add_argument("--resume-completed", action="store_true")
    a = p.parse_args()
    if not 60 <= a.max_seconds <= 28800:
        raise ValueError("Queue must be bounded to at most eight hours")
    source = json.loads(a.views.read_text(encoding="utf-8"))
    common = json.loads(a.preflight_manifest.read_text(encoding="utf-8"))
    qual = json.loads(a.qualification.read_text(encoding="utf-8"))
    if (
        material_hash(source) != common["material_sha256"]
        or qual["status"] != "COMPLETE"
        or not qual["all_replay_checks_passed"]
        or not qual["cache_10gb_planning_gate"]
        or qual.get("cache_format") != "lossless_aliases_v1"
        or not qual.get("lossless_reconstruction_verified")
        or qual["material_sha256"] != common["material_sha256"]
    ):
        raise ValueError("Unqualified or changed material")
    shard_material(source, a.shards, 0)
    fingerprint = code_fingerprint(ROOT)
    identity_fields = {
        "protocol": "GRAIN_VM05_SERIAL_SHARDS_V1",
        "common_parent_identity": common["parent_identity"],
        "material_sha256": common["material_sha256"],
        "code_sha256": fingerprint,
        "shards": a.shards,
        "stages": list(STAGES),
        "labels_read": False,
        "max_cache_bytes": 10_000_000_000,
    }
    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    queue_manifest = out / "manifest.json"
    if queue_manifest.exists():
        if (
            not a.resume_completed
            or json.loads(queue_manifest.read_text()) != identity_fields
        ):
            raise ValueError("Existing queue requires explicit identical-code resume")
    elif any(out.iterdir()):
        raise ValueError("Use a fresh queue directory")
    else:
        write_json(queue_manifest, identity_fields)
        archive_source(ROOT, out / "source_snapshot.zip")
    lock = out / "queue.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    start = time.monotonic()
    completed = []
    status = {"status": "RUNNING", "pid": os.getpid(), "labels_read": False}
    try:
        # One shard across all observers before the next: early records have exact same evidence.
        for index in range(a.shards):
            for stage in STAGES:
                folder = out / stage / f"shard_{index:02d}"
                if folder.exists():
                    record = verify_completed(
                        folder, source, common, stage, a.shards, index, fingerprint
                    )
                    completed.append(
                        {"stage": stage, "shard": index, "authors": record["authors"]}
                    )
                    continue
                elapsed = time.monotonic() - start
                remaining = a.max_seconds - elapsed
                if remaining < 1800 or cache_size(out) >= 9_800_000_000:
                    write_json(
                        out / "queue_status.json",
                        status
                        | {
                            "status": "PAUSED_BUDGET_BETWEEN_SHARDS",
                            "completed": completed,
                            "seconds": elapsed,
                            "cache_bytes": cache_size(out),
                        },
                    )
                    return
                if code_fingerprint(ROOT) != fingerprint:
                    raise ValueError("Worker source changed during frozen collection")
                folder.parent.mkdir(parents=True, exist_ok=True)
                log = out / f"{stage}_shard_{index:02d}.log"
                command = [
                    sys.executable,
                    "-u",
                    str(
                        ROOT / "scripts/run_grain_vector_mechanism_external_features.py"
                    ),
                    "--views",
                    str(a.views),
                    "--preflight-manifest",
                    str(a.preflight_manifest),
                    "--stage",
                    stage,
                    "--technical-parent",
                    str(a.technical_root / TECHNICAL_DIRS[stage]),
                    "--output-dir",
                    str(folder),
                    "--shard-count",
                    str(a.shards),
                    "--shard-index",
                    str(index),
                    "--max-seconds",
                    str(min(1800, int(remaining) - 60)),
                    "--max-cache-bytes",
                    str(10_000_000_000 - cache_size(out)),
                ]
                if stage == "e5_large":
                    command += ["--model-dir", str(a.e5_model_dir)]
                status.update(
                    stage=stage, shard=index, completed=completed, seconds=elapsed
                )
                write_json(out / "queue_status.json", status)
                print(
                    json.dumps({k: v for k, v in status.items() if k != "completed"}),
                    flush=True,
                )
                env = dict(os.environ, HF_HUB_DISABLE_PROGRESS_BARS="1")
                with log.open("w", encoding="utf-8") as handle:
                    proc = subprocess.run(
                        command,
                        check=False,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        env=env,
                        timeout=remaining,
                    )
                if proc.returncode:
                    raise RuntimeError(
                        f"Observer failed: {stage} shard {index}, exit {proc.returncode}; see scoped log"
                    )
                record = verify_completed(
                    folder, source, common, stage, a.shards, index, fingerprint
                )
                completed.append(
                    {"stage": stage, "shard": index, "authors": record["authors"]}
                )
        write_json(
            out / "queue_status.json",
            status
            | {
                "status": "COMPLETE",
                "completed": completed,
                "seconds": time.monotonic() - start,
                "cache_bytes": cache_size(out),
            },
        )
    except Exception as error:
        write_json(
            out / "queue_status.json",
            status
            | {
                "status": "FAILED_STOP",
                "completed": completed,
                "error_type": type(error).__name__,
                "error": str(error),
                "seconds": time.monotonic() - start,
            },
        )
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
