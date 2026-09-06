#!/usr/bin/env python3
"""Bounded serial direct-trait collection with complete-only replay/resume."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.direct_material import (
    archive_direct,
    load_registration,
    verify_tokenizer,
)
from grain_research.vector_mechanisms.direct_queue import semantic_parent, verify_shard
from grain_research.vector_mechanisms.external_material import shard_material
from grain_research.vector_mechanisms.ready_comparison import digest
from scripts.collect_grain_direct_traits import CODE, qualified_parent
from scripts.qualify_grain_direct_traits_semantics import require_idle_history
from scripts.run_grain_vector_mechanism_controls import write_json

SEMANTIC_CODE = (
    *CODE,
    "scripts/qualify_grain_direct_traits_semantics.py",
    "grain_research/vector_mechanisms/direct_semantics.py",
)
QUEUE_CODE = (
    *SEMANTIC_CODE,
    "grain_research/vector_mechanisms/direct_queue.py",
    "scripts/run_grain_direct_traits_queue.py",
)


def fingerprint(names):
    """Only code actually used by this observer/supervisor can invalidate resume."""
    return {name: digest(ROOT / name) for name in names}


def arguments(a, folder, index):
    """Call the unchanged qualified collector with paths and bounded resources only."""
    cmd = [sys.executable, "-u", str(ROOT / "scripts/collect_grain_direct_traits.py")]
    for key in ("views", "material_dir", "tokenizer_dir"):
        cmd += ["--" + key.replace("_", "-"), str(getattr(a, key).resolve())]
    cmd += [
        "--output-dir",
        str(folder),
        "--shard-count",
        str(a.shards),
        "--shard-index",
        str(index),
        "--max-seconds",
        str(a.shard_seconds),
        "--max-cache-bytes",
        str(a.shard_cache_bytes),
    ]
    if a.preflight_only:
        cmd.append("--preflight-only")
    else:
        for key in ("qualification_dir", "observer_queue_dir"):
            cmd += ["--" + key.replace("_", "-"), str(getattr(a, key).resolve())]
    return cmd


def run(a):
    """Never overwrite/retry a partial shard or claim completion on a budget pause."""
    if (
        not 60 <= a.max_seconds <= 28800
        or not 60 <= a.shard_seconds <= 3600
        or not 0 < a.shard_cache_bytes <= 200_000_000
        or not 0 < a.max_cache_bytes <= 6_000_000_000
        or (not a.preflight_only and a.shards != 128)
    ):
        raise ValueError("Require bounded time/cache and128 real source shards")
    _, whole, material = load_registration(a.views, a.material_dir)
    shard_material(whole, a.shards, 0)
    verify_tokenizer(a.tokenizer_dir, material)
    code, queue_code = fingerprint(CODE), fingerprint(QUEUE_CODE)
    qm = qs = semantic = None
    if not a.preflight_only:
        require_idle_history(a.history_queue_dir)
        qm, qs = qualified_parent(a.qualification_dir, material, code, whole)
        tokenizer = AutoTokenizer.from_pretrained(
            str(a.tokenizer_dir), local_files_only=True, trust_remote_code=False
        )
        semantic = semantic_parent(
            a.semantic_dir,
            material,
            qm["identity"],
            qs["model_signature"],
            fingerprint(SEMANTIC_CODE),
            tokenizer,
        )
    out = a.output_dir.resolve()
    if not out.is_relative_to(ROOT / "results"):
        raise ValueError("Private direct results directory required")
    fields = {
        "protocol": "VM05_DIRECT_QUEUE_V1",
        "preflight_only": a.preflight_only,
        "material_identity": material["identity"],
        "authors": len(whole),
        "author_commitment": material_hash([c["author_id"] for c in whole]),
        "shards": a.shards,
        "collector_code_sha256": code,
        "code_sha256": queue_code,
        "qualification_identity": qm["identity"] if qm else None,
        "model_signature": qs["model_signature"] if qs else None,
        "semantic_parent": semantic,
        "shard_seconds": a.shard_seconds,
        "shard_cache_bytes": a.shard_cache_bytes,
        "max_cache_bytes": a.max_cache_bytes,
        "labels_read": False,
        "bridge_outcomes_read": False,
    }
    fields["identity"] = material_hash(fields)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.json"
    if manifest.exists():
        if (
            not a.resume_completed
            or json.loads(manifest.read_text(encoding="utf-8")) != fields
        ):
            raise ValueError("Existing direct queue needs identical explicit resume")
    elif any(out.iterdir()):
        raise ValueError("Use a fresh direct queue directory")
    else:
        write_json(manifest, fields)
        archive_direct(ROOT, out / "source_snapshot.zip")
    lock = out / "queue.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    started, completed, cache_bytes, proc = time.monotonic(), [], 0, None

    def save(state, **extra):
        """Persist only independently verified completed counts and this owned child's PID."""
        write_json(
            out / "queue_status.json",
            {
                "status": state,
                "pid": os.getpid(),
                "parent_identity": fields["identity"],
                "completed": completed,
                "verified_authors": sum(r["authors"] for r in completed),
                "cache_bytes": cache_bytes,
                "seconds": time.monotonic() - started,
                "real_neural_collection": not a.preflight_only,
                "labels_read": False,
                "bridge_outcomes_read": False,
                **extra,
            },
        )

    try:
        for i in range(a.shards):
            if fingerprint(QUEUE_CODE) != queue_code:
                raise ValueError("Direct queue source changed")
            folder = out / f"shard_{i:03d}"
            if not folder.exists():
                remaining = a.max_seconds - (time.monotonic() - started)
                if (
                    remaining < a.shard_seconds + 30
                    or cache_bytes + a.shard_cache_bytes > a.max_cache_bytes
                ):
                    save("PAUSED_BUDGET_BETWEEN_SHARDS", next_shard=i, child_pid=None)
                    return
                if not a.preflight_only:
                    require_idle_history(a.history_queue_dir)
                with (out / f"shard_{i:03d}.log").open("x", encoding="utf-8") as log:
                    proc = subprocess.Popen(
                        arguments(a, folder, i),
                        cwd=ROOT,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env=dict(
                            os.environ,
                            TOKENIZERS_PARALLELISM="false",
                            HF_HUB_OFFLINE="1",
                            TRANSFORMERS_OFFLINE="1",
                            PYTHONUTF8="1",
                        ),
                    )
                    child_start = time.monotonic()
                    while proc.poll() is None:
                        if time.monotonic() - child_start > a.shard_seconds + 30:
                            raise TimeoutError(
                                "Owned direct shard time bound; preserve partial"
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
                            f"Direct shard{i} failed; preserve scoped log and cache"
                        )
            verified = verify_shard(
                folder,
                whole,
                material,
                code,
                a.shards,
                i,
                preflight=a.preflight_only,
                qualification=fields["qualification_identity"],
                model_signature=fields["model_signature"],
            )
            completed.append({"shard": i, **verified})
            cache_bytes += verified["cache_bytes"]
            if cache_bytes > a.max_cache_bytes:
                raise MemoryError("Verified direct caches exceed frozen total bound")
            save("RUNNING", verified_shard=i, child_pid=None)
        if sum(r["authors"] for r in completed) != len(whole):
            raise ValueError("Direct author union incomplete")
        save("COMPLETE", child_pid=None)
        print(
            json.dumps(
                {
                    "status": "COMPLETE",
                    "shards": len(completed),
                    "authors": len(whole),
                    "real_neural_collection": not a.preflight_only,
                    "cache_bytes": cache_bytes,
                }
            ),
            flush=True,
        )
    except BaseException as e:
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait()
        save("FAILED_STOP", error_type=type(e).__name__, error=str(e), child_pid=None)
        raise
    finally:
        lock.unlink(missing_ok=True)


def main():
    """Preflight is token-only; real source extraction requires both qualified controls."""
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("views", "material-dir", "tokenizer-dir", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in (
        "qualification-dir",
        "semantic-dir",
        "observer-queue-dir",
        "history-queue-dir",
    ):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--resume-completed", action="store_true")
    p.add_argument("--shards", type=int, default=128)
    p.add_argument("--max-seconds", type=int, default=28800)
    p.add_argument("--shard-seconds", type=int, default=2400)
    p.add_argument("--shard-cache-bytes", type=int, default=200_000_000)
    p.add_argument("--max-cache-bytes", type=int, default=6_000_000_000)
    run(p.parse_args())


if __name__ == "__main__":
    main()
