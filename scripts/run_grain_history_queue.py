#!/usr/bin/env python3
"""Serial, bounded, checksum-verified history extraction; token-only mode stays separate."""

from __future__ import annotations

import argparse
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
from grain_research.vector_mechanisms.history_cache import verify_history_shard
from grain_research.vector_mechanisms.history_collection import (
    observation_signature,
    qualification_packet,
)
from grain_research.vector_mechanisms.history_instrument import (
    archive_instrument,
    registered_cases,
)
from grain_research.vector_mechanisms.ready_comparison import digest
from scripts.collect_grain_source_history import CODE, require_technical
from scripts.run_grain_vector_mechanism_controls import write_json

QUEUE_CODE = (
    *CODE,
    "scripts/run_grain_history_queue.py",
    "grain_research/vector_mechanisms/history_cache.py",
)


def fingerprint(names):
    """Freeze only the declared instrument and supervisor, not unrelated edits."""
    return {name: digest(ROOT / name) for name in names}


def arguments(a, folder, index, seconds):
    """Build a non-shell child invocation without label paths or external services."""
    cmd = [sys.executable, "-u", str(ROOT / "scripts/collect_grain_source_history.py")]
    for key in ("views", "material_dir", "audit_dir", "qwen_tokenizer"):
        cmd.extend(["--" + key.replace("_", "-"), str(getattr(a, key).resolve())])
    cmd += [
        "--output-dir",
        str(folder),
        "--shard-count",
        str(a.shards),
        "--shard-index",
        str(index),
        "--max-seconds",
        str(seconds),
        "--max-cache-bytes",
        str(a.shard_cache_bytes),
    ]
    if a.preflight_only:
        cmd.append("--preflight-only")
    else:
        for key in ("qualification_dir", "technical_parent", "observer_queue_dir"):
            cmd.extend(["--" + key.replace("_", "-"), str(getattr(a, key).resolve())])
    return cmd


def run(a):
    """Resume only complete verified shards; preserve any failure instead of overwriting."""
    if (
        not 60 <= a.max_seconds <= 28800
        or not 60 <= a.shard_seconds <= 3600
        or not 0 < a.shard_cache_bytes <= 1_000_000_000
        or not 0 < a.max_cache_bytes <= 24_000_000_000
    ):
        raise ValueError("History queue must be time/disk bounded")
    _, whole, material, audit = registered_cases(a.views, a.material_dir, a.audit_dir)
    shard_material(whole, a.shards, 0)
    code, queue_code = fingerprint(CODE), fingerprint(QUEUE_CODE)
    qm = technical = model = None
    if not a.preflight_only:
        if any(
            getattr(a, k) is None
            for k in ("qualification_dir", "technical_parent", "observer_queue_dir")
        ):
            raise ValueError(
                "Real queue requires qualification, technical replay and original queue"
            )
        qm, _, model = qualification_packet(a.qualification_dir, material, audit, ROOT)
        technical = require_technical(a.technical_parent, qm, material, code)
        if technical["model_signature"] != observation_signature(model):
            raise ValueError("Technical replay observer changed")
    out = a.output_dir.resolve()
    if not out.is_relative_to(ROOT / "results"):
        raise ValueError("Private result directory required")
    fields = {
        "protocol": "VM05_HISTORY_QUEUE_V1",
        "preflight_only": a.preflight_only,
        "material_identity": material["identity"],
        "audit_identity": audit["identity"],
        "qualification_identity": qm["identity"] if qm else None,
        "technical_identity": technical["identity"] if technical else None,
        "model_signature": observation_signature(model) if model else None,
        "strict_author_commitment": material_hash([c["author_id"] for c in whole]),
        "authors": len(whole),
        "shards": a.shards,
        "code_sha256": queue_code,
        "collector_code_sha256": code,
        "max_cache_bytes": a.max_cache_bytes,
        "shard_cache_bytes": a.shard_cache_bytes,
        "shard_seconds": a.shard_seconds,
        "labels_read": False,
        "bridge_outcomes_read": False,
    }
    fields["identity"] = material_hash(fields)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.json"
    if manifest.exists():
        if not a.resume_completed or json.loads(manifest.read_text()) != fields:
            raise ValueError("Existing history queue needs identical explicit resume")
    elif any(out.iterdir()):
        raise ValueError("Use fresh history queue output")
    else:
        write_json(manifest, fields)
        archive_instrument(ROOT, out / "source_snapshot.zip")
    lock = out / "queue.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    started = time.monotonic()
    completed = []
    cache_bytes = 0
    proc = None
    status = {
        "status": "RUNNING",
        "pid": os.getpid(),
        "parent_identity": fields["identity"],
        "real_neural_collection": not a.preflight_only,
        "labels_read": False,
        "bridge_outcomes_read": False,
    }

    def save(state, **extra):
        """Write only verified completed counts and current owned process status."""
        write_json(
            out / "queue_status.json",
            status
            | {
                "status": state,
                "completed": completed,
                "verified_authors": sum(r["authors"] for r in completed),
                "cache_bytes": cache_bytes,
                "seconds": time.monotonic() - started,
            }
            | extra,
        )

    try:
        for i in range(a.shards):
            if fingerprint(QUEUE_CODE) != queue_code:
                raise ValueError("History queue source changed")
            folder = out / f"shard_{i:03d}"
            if not folder.exists():
                remaining = a.max_seconds - (time.monotonic() - started)
                if (
                    remaining < a.shard_seconds + 30
                    or cache_bytes + a.shard_cache_bytes > a.max_cache_bytes
                ):
                    save("PAUSED_BUDGET_BETWEEN_SHARDS", next_shard=i)
                    return
                log = out / f"shard_{i:03d}.log"
                with log.open("x", encoding="utf-8") as handle:
                    proc = subprocess.Popen(
                        arguments(a, folder, i, a.shard_seconds),
                        cwd=ROOT,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        env=dict(
                            os.environ,
                            TOKENIZERS_PARALLELISM="false",
                            HF_HUB_OFFLINE="1",
                            TRANSFORMERS_OFFLINE="1",
                        ),
                    )
                    child_start = time.monotonic()
                    while proc.poll() is None:
                        if time.monotonic() - child_start > a.shard_seconds + 30:
                            proc.kill()
                            proc.wait()
                            raise TimeoutError(
                                "Owned history child exceeded shard time; preserve partial"
                            )
                        save(
                            "RUNNING",
                            shard=i,
                            child_pid=proc.pid,
                            child_seconds=time.monotonic() - child_start,
                        )
                        time.sleep(2)
                    if proc.returncode:
                        raise RuntimeError(
                            f"History shard {i} failed, exit {proc.returncode}; see scoped log"
                        )
            verified = verify_history_shard(
                folder,
                whole,
                material,
                audit,
                code,
                a.shards,
                i,
                qualification=fields["qualification_identity"],
                technical=fields["technical_identity"],
                model_signature=fields["model_signature"],
                allow_preflight=a.preflight_only,
            )
            completed.append({"shard": i, **verified})
            cache_bytes += verified["cache_bytes"]
            save("RUNNING", verified_shard=i, child_pid=None)
        if sum(r["authors"] for r in completed) != len(whole):
            raise ValueError("History author union is incomplete")
        save("COMPLETE", child_pid=None)
        print(
            json.dumps(
                {
                    "status": "COMPLETE",
                    "authors": len(whole),
                    "shards": len(completed),
                    "real_neural_collection": not a.preflight_only,
                    "cache_bytes": cache_bytes,
                }
            ),
            flush=True,
        )
    except BaseException as error:
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait()
        save(
            "FAILED_STOP",
            exception_type=type(error).__name__,
            error=str(error),
            child_pid=None,
        )
        raise
    finally:
        lock.unlink(missing_ok=True)


def main():
    """Token preflights are allowed on Mac; real history observer remains pinned CUDA."""
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("views", "material-dir", "audit-dir", "qwen-tokenizer", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("qualification-dir", "technical-parent", "observer-queue-dir"):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--resume-completed", action="store_true")
    p.add_argument("--shards", type=int, default=128)
    p.add_argument("--max-seconds", type=int, default=28800)
    p.add_argument("--shard-seconds", type=int, default=1800)
    p.add_argument("--shard-cache-bytes", type=int, default=500_000_000)
    p.add_argument("--max-cache-bytes", type=int, default=24_000_000_000)
    run(p.parse_args())


if __name__ == "__main__":
    main()
