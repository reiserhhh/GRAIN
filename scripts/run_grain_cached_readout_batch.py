#!/usr/bin/env python3
"""Run the declared secondary cached readouts with a single bounded CPU owner."""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
import zipfile
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.external_controls import exclude_reference
from grain_research.vector_mechanisms.external_readouts import ARMS, KINDS
from grain_research.vector_mechanisms.inventory import archive_source
from grain_research.vector_mechanisms.ready_comparison import digest, verified_packet
from grain_research.vector_mechanisms.ready_registry import TASKS
from scripts.analyze_grain_ready_primary import require_complete_batch
from scripts.run_grain_vector_mechanism_controls import write_json
from scripts.run_grain_vector_mechanism_neural_source_queue import execution_lock
from scripts.run_grain_vector_mechanism_source_controls import scoring_commitment
from scripts.run_grain_vector_mechanism_sparse_baselines import (
    bounded_child,
    has_substantive_outputs,
)
from scripts.summarize_grain_vector_external_links import load_oof

SITES = ("block_10", "block_04", "block_16", "final_norm")
READOUTS = tuple(k for k in KINDS if k != "body_token")
INPUT_FLAGS = ("views", "preflight_dir", "parent_dir", "prepared_root", "vector_dir")
LEASE_FLAGS = ("ready_dir", "sensitivity_dir", "primary_dir", "resource_dir")
CODE = (
    "scripts/run_grain_cached_readout_batch.py",
    "scripts/run_grain_full_source_readout.py",
    "scripts/analyze_grain_ready_primary.py",
    "scripts/run_grain_ready_primary_batch.py",
    "grain_research/vector_mechanisms/external_readout_cache.py",
    "grain_research/vector_mechanisms/external_readouts.py",
    "grain_research/vector_mechanisms/external_controls.py",
    "grain_research/vector_mechanisms/ready_comparison.py",
    "grain_research/vector_mechanisms/ready_registry.py",
    "scripts/run_grain_vector_mechanism_external_queue.py",
    "scripts/run_grain_vector_mechanism_neural_source_queue.py",
    "scripts/run_grain_vector_mechanism_sparse_baselines.py",
)


def cases():
    """All315 source cases:90 block10 cases, then225 other-site cases, no duplicates."""
    return [
        {
            "id": f"{task}_{arm}_{site}_{readout}",
            "task": task,
            "arm": arm,
            "site": site,
            "readout": readout,
        }
        for site in SITES
        for readout in READOUTS + (("logit_margins",) if site == "block_10" else ())
        for arm in ARMS
        for task in TASKS
    ]


def process_absent(pid, expected=None):
    """A completion JSON is not a process exit; reject unknown or reused PIDs."""
    p = subprocess.run(
        ["ps", "-p", str(pid), "-o", "args="],
        capture_output=True,
        text=True,
        check=False,
    )
    if p.returncode == 1 and not p.stdout.strip():
        return True
    if p.returncode or (expected and expected not in p.stdout):
        raise ValueError("Process owner cannot be verified")
    return False


def predecessor_done(folder):
    """Require the complete34-fit ready stage, its reused profile and exited owner."""
    m = json.loads((folder / "batch_manifest.json").read_text(encoding="utf-8"))
    s = json.loads((folder / "batch_status.json").read_text(encoding="utf-8"))
    if s["status"] in ("FAILED_STOP", "PAUSED_BUDGET", "PAUSED_WAIT_BUDGET"):
        raise ValueError("Ready predecessor needs completion before secondary fits")
    if s["status"] != "COMPLETE":
        return False
    require_complete_batch(m, s)
    return process_absent(s["pid"], "run_grain_ready_primary_batch.py")


def source_members(a):
    """Freeze reference-free source membership without reading psychological values."""
    source = json.loads(a.views.read_text(encoding="utf-8"))
    common = json.loads(
        (a.preflight_dir / "common_material_manifest.json").read_text(encoding="utf-8")
    )
    parent = json.loads((a.parent_dir / "manifest.json").read_text(encoding="utf-8"))
    ids = [b["author_id"] for b in source]
    if (
        len(ids) != len(set(ids))
        or set(ids) != set(common["author_ids"])
        or material_hash(source) != common["material_sha256"]
        or material_hash({k: v for k, v in parent.items() if k != "identity"})
        != parent["identity"]
    ):
        raise ValueError("Source membership or source-table parent changed")
    budget = pd.read_csv(
        a.preflight_dir / "budget_by_author.csv", dtype={"author_id": str}
    )
    members = {}
    for task in TASKS:
        group = "big5_only" if task == "big5" else "mbti_only"
        selected = exclude_reference(
            [b for b in source if b["source_group"] == group], budget
        )
        folds = [b["official_folds"][task] for b in selected]
        if not selected or set(folds) != set(range(5)):
            raise ValueError("All five source-exclusive official folds required")
        members[task] = selected
    return members, common, parent


def validate_result(folder, case, members, common, parent, budget_sha):
    """Audit checksums, recomputed OOF, source membership and all five replay records."""
    m, s = verified_packet(folder)
    packet = load_oof(folder)
    selected = members[case["task"]]
    ids = [b["author_id"] for b in selected]
    folds = [b["official_folds"][case["task"]] for b in selected]
    expected = {
        "protocol": "VM05_FULL_SOURCE_READOUT_V1",
        "task": case["task"],
        "feature": "qwen_" + case["site"],
        "arm": case["arm"],
        "readout": case["readout"],
        "source_group": selected[0]["source_group"],
        "authors": len(ids),
        "source_material_sha256": material_hash(selected),
        "source_author_commitment": material_hash(ids),
        "fold_commitment": material_hash(folds),
        "source_table_parent_identity": parent["identity"],
        "common_material_sha256": common["material_sha256"],
        "budget_sha256": budget_sha,
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
    }
    record = pd.read_csv(folder / "saved_head_replay.csv")
    by_id = dict(zip(ids, folds, strict=True))
    if (
        any(m.get(k) != v for k, v in expected.items())
        or set(packet["ids"]) != set(ids)
        or any(
            int(f) != by_id[u]
            for u, f in zip(packet["ids"], packet["folds"], strict=True)
        )
        or s["authors"] != len(ids)
        or s["full_train_authors"] != len(ids)
        or s["bridge_labels_used"]
        or s["neural_forwards"] != 0
        or not s["exclude_reference"]
        or s["permutation_seed"] is not None
        or not np.isfinite(s["saved_fold_replay_max"])
        or s["saved_fold_replay_max"] > 1e-10
        or digest(folder / "private_full_source_head.joblib") != s["full_head_sha256"]
        or len(record) != 5
        or set(record["fold"]) != set(range(5))
        or not np.isfinite(record["max_absolute_error"]).all()
        or (record["max_absolute_error"] > 1e-10).any()
        or any(digest(ROOT / n) != sha for n, sha in m["scoring_code"].items())
    ):
        raise ValueError(
            "Secondary result differs from the frozen source/readout contract"
        )
    return {
        "case": case["id"],
        "identity": m["identity"],
        "authors": len(ids),
        "score": s["score"],
        "replayed_folds": 5,
        "replay_max": s["saved_fold_replay_max"],
    }


def run(a):
    """Hold all predecessor CPU leases; never refit an existing or partial condition."""
    if not 3660 <= a.max_seconds <= 28800 or not 60 <= a.max_wait_seconds <= 10800:
        raise ValueError("Require bounded fit and wait budgets")
    out = a.output_dir.resolve()
    if not out.is_relative_to(ROOT / "results"):
        raise ValueError("Private result directory required")
    paired, paired_s = verified_packet(a.paired_dir)
    ready = json.loads(
        (a.ready_dir / "batch_manifest.json").read_text(encoding="utf-8")
    )
    if (
        paired["protocol"] != "VM05_RAW_READY_PAIRED_COMPARISON_V1"
        or paired.get("batch_identity") != ready["identity"]
        or paired_s.get("model_refits") != 0
        or paired_s.get("bridge_labels_read") is not False
        or paired_s.get("new_source_heads_accepted") != 34
        or paired_s.get("historical_source_heads_reused") != 1
        or paired_s.get("saved_folds_accepted") != 175
        or paired_s.get("paired_rows") != 90
    ):
        raise ValueError("Complete paired ready acceptance must precede secondary fits")
    members, common, parent = source_members(a)
    code = scoring_commitment() | {n: digest(ROOT / n) for n in CODE}
    paths = {
        "views": a.views,
        "common": a.preflight_dir / "common_material_manifest.json",
        "budget": a.preflight_dir / "budget_by_author.csv",
        "source_parent": a.parent_dir / "manifest.json",
        "vector_parent": a.vector_dir / "manifest.json",
        "ready_paired": a.paired_dir / "summary.json",
        **{n: getattr(a, n) / "batch_manifest.json" for n in LEASE_FLAGS},
    }
    inputs = {n: digest(p) for n, p in paths.items()}

    def unchanged():
        if any(digest(ROOT / n) != sha for n, sha in code.items()) or any(
            digest(paths[n]) != sha for n, sha in inputs.items()
        ):
            raise ValueError("Frozen readout code or parent changed")

    out.mkdir(parents=True, exist_ok=True)
    with execution_lock(out) as lease:
        m = {
            "protocol": "VM05_CACHED_SECONDARY_READOUT_BATCH_V1",
            "cases": cases(),
            "cohort": "full_source_reference_free",
            "authors_by_task": {t: len(b) for t, b in members.items()},
            "pca": True,
            "seed": 731,
            "thread_control": "retain",
            "readout_selection": "none",
            "input_sha256": inputs,
            "scoring_code": code,
            "ready_paired_identity": paired["identity"],
            "fit_seconds": 3600,
            "child_rss_limit": 8 * 1024**3,
            "neural_forwards": 0,
            "bridge_labels_read": False,
            "program_complete": False,
        }
        m["identity"] = material_hash(m)
        path = out / "batch_manifest.json"
        if path.exists():
            if (
                not a.resume_completed
                or json.loads(path.read_text(encoding="utf-8")) != m
            ):
                raise ValueError("Existing batch requires identical explicit resume")
            status_path = out / "batch_status.json"
            if status_path.exists():
                prior = json.loads(status_path.read_text(encoding="utf-8"))
                if prior.get("child_pid") and not process_absent(prior["child_pid"]):
                    raise ValueError("Previous owned child remains alive")
        elif has_substantive_outputs(out):
            raise ValueError("Fresh readout result directory required")
        else:
            write_json(path, m)
            archive_source(ROOT, out / "source_snapshot.zip")
            with zipfile.ZipFile(
                out / "source_snapshot.zip", "a", zipfile.ZIP_DEFLATED
            ) as z:
                for n in code:
                    if n not in z.namelist():
                        z.write(ROOT / n, n)
        lease["started"] = True
        completed = []

        def status(state, **extra):
            write_json(
                out / "batch_status.json",
                {
                    "status": state,
                    "pid": os.getpid(),
                    "planned": len(m["cases"]),
                    "completed": completed,
                    "bridge_labels_read": False,
                    "program_complete": False,
                    **extra,
                },
            )

        started = time.monotonic()
        while not predecessor_done(a.ready_dir):
            unchanged()
            elapsed = time.monotonic() - started
            status("WAITING_FOR_READY_OWNER_EXIT", wait_seconds=elapsed, child_pid=None)
            if elapsed >= a.max_wait_seconds:
                status("PAUSED_WAIT_BUDGET", child_pid=None)
                return
            time.sleep(30)
        with ExitStack() as locks:
            for n in LEASE_FLAGS:
                h = locks.enter_context((getattr(a, n) / ".execution.lock").open("a"))
                fcntl.flock(h, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if not predecessor_done(a.ready_dir):
                raise ValueError("Ready predecessor restarted during handoff")
            unchanged()
            started = time.monotonic()
            (out / "job_logs").mkdir(exist_ok=True)
            for c in m["cases"]:
                unchanged()
                folder = out / c["id"]
                if not folder.exists():
                    if a.max_seconds - (time.monotonic() - started) < 3660:
                        status("PAUSED_BUDGET", next_case=c["id"], child_pid=None)
                        return
                    folder.mkdir()
                    cmd = [
                        sys.executable,
                        "-u",
                        str(ROOT / "scripts/run_grain_full_source_readout.py"),
                    ]
                    for n in INPUT_FLAGS:
                        cmd += [
                            "--" + n.replace("_", "-"),
                            str(getattr(a, n).resolve()),
                        ]
                    for n in ("task", "arm", "site", "readout"):
                        cmd += ["--" + n, c[n]]
                    cmd += [
                        "--output-dir",
                        str(folder),
                        "--exclude-reference",
                        "--worker",
                    ]

                    def update(pid, elapsed, rss, peak, case=c["id"]):
                        status(
                            "FIT",
                            current=case,
                            child_pid=pid,
                            seconds=elapsed,
                            rss_bytes=rss,
                        )

                    with (out / "job_logs" / (c["id"] + ".log")).open(
                        "x", encoding="utf-8"
                    ) as log:
                        resource = bounded_child(cmd, log, 3600, 8 * 1024**3, update)
                    write_json(folder / "supervisor_resource.json", resource)
                accepted = validate_result(
                    folder / "fit", c, members, common, parent, inputs["budget"]
                )
                completed.append(accepted)
                status("CASE_COMPLETE", current=c["id"], child_pid=None)
                print(json.dumps(accepted), flush=True)
            status("COMPLETE", child_pid=None, seconds=time.monotonic() - started)


def main():
    """Execute the full predeclared family without winner selection or new neural work."""
    p = argparse.ArgumentParser(description=__doc__)
    for n in (*INPUT_FLAGS, *LEASE_FLAGS, "paired_dir", "output_dir"):
        p.add_argument("--" + n.replace("_", "-"), type=Path, required=True)
    p.add_argument("--max-seconds", type=int, default=28800)
    p.add_argument("--max-wait-seconds", type=int, default=10800)
    p.add_argument("--resume-completed", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
