#!/usr/bin/env python3
"""Run frozen full-source neural heads after the owned strong CPU batch releases its lock."""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.external_controls import exclude_reference
from grain_research.vector_mechanisms.inventory import archive_source
from grain_research.vector_mechanisms.ready_comparison import (
    digest,
    require_observer_completion,
)
from scripts.run_grain_vector_mechanism_controls import write_json
from scripts.run_grain_vector_mechanism_source_controls import scoring_commitment
from scripts.run_grain_vector_mechanism_sparse_baselines import (
    bounded_child,
    has_substantive_outputs,
)
from scripts.summarize_grain_vector_external_links import load_oof


def cases():
    """All45 frozen primary observer/task/arm conditions, not best-case selection."""
    return [
        {"id": f"{task}_{arm}_{feature}", "task": task, "arm": arm, "feature": feature}
        for arm in ("raw", "cue_mask", "placebo_mask")
        for task in ("big5", "EI_cont", "SN_cont", "TF_cont", "JP_cont")
        for feature in ("bge_large", "e5_large", "qwen_block_10")
    ]


def owner_alive(pid):
    """Verify the named predecessor process, rejecting PID reuse by another command."""
    p = subprocess.run(
        ["ps", "-p", str(pid), "-o", "args="],
        capture_output=True,
        text=True,
        check=False,
    )
    if p.returncode == 1 and not p.stdout.strip():
        return False
    if p.returncode or "run_grain_strong_source_controls.py" not in p.stdout:
        raise RuntimeError("Cannot verify predecessor PID ownership")
    return True


def predecessor_ready(folder, pid):
    """A terminal state file alone is not proof that the resource owner has exited."""
    if owner_alive(pid):
        return False
    s = json.loads((folder / "batch_status.json").read_text(encoding="utf-8"))
    if s["status"] not in ("COMPLETE", "PAUSED_BUDGET"):
        raise ValueError(
            "Predecessor stopped outside a verified terminal budget boundary"
        )
    if s.get("child_pid"):
        p = subprocess.run(
            ["ps", "-p", str(s["child_pid"]), "-o", "pid="],
            capture_output=True,
            text=True,
            check=False,
        )
        if p.returncode != 1 or p.stdout.strip():
            raise ValueError("Predecessor child remains alive or cannot be verified")
    return True


@contextmanager
def execution_lock(out):
    """Publish terminal failures only after this process owns and starts the run."""
    import fcntl

    with (out / ".execution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lease = {"started": False}
        try:
            yield lease
        except BaseException as error:
            if lease["started"]:
                path = out / "batch_status.json"
                previous = (
                    json.loads(path.read_text(encoding="utf-8"))
                    if path.exists()
                    else {}
                )
                write_json(
                    path,
                    previous
                    | {
                        "status": "FAILED_STOP",
                        "pid": os.getpid(),
                        "child_pid": None,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "bridge_labels_read": False,
                    },
                )
            raise


def validate_result(packet, case, source, common):
    """Accept only this observer/arm on the exact independently selected source cohort."""
    m, s = packet["manifest"], packet["summary"]
    group = "big5_only" if case["task"] == "big5" else "mbti_only"
    bundles = [b for b in source if b["source_group"] == group]
    expected = {
        "protocol": "GRAIN_VM05_SOURCE_OOF_V1",
        "task": case["task"],
        "arm": case["arm"],
        "feature": case["feature"],
        "source_group": group,
        "exclude_reference": True,
        "thread_control": "retain",
        "drop_seed": 731,
        "permutation_seed": None,
        "seed": 731,
        "pca": True,
        "pca_applied": True,
        "inner_cv": "3 author folds, seed731",
        "outer_cv": "5 source-specific official folds",
        "bridge_labels_used": False,
        "common_material_sha256": common,
        "source_material_sha256": material_hash(bundles),
    }
    if (
        any(m[k] != v for k, v in expected.items())
        or s["bridge_labels_used"]
        or set(packet["ids"]) != {b["author_id"] for b in bundles}
    ):
        raise ValueError("Result differs from the complete frozen source condition")
    original_folds = {
        b["author_id"]: b["official_folds"][case["task"]] for b in bundles
    }
    if any(
        int(f) != original_folds[u]
        for u, f in zip(packet["ids"], packet["folds"], strict=True)
    ):
        raise ValueError("Source OOF differs from original task-specific folds")


def run(a):
    """Keep one CPU owner; preserve incomplete fits and never overwrite/retry them."""
    import fcntl

    import pandas as pd

    if (
        sys.platform not in ("darwin", "linux")
        or not 60 <= a.max_wait_seconds <= 10800
        or not 4260 <= a.max_seconds <= 28800
        or a.predecessor_pid <= 1
    ):
        raise ValueError("Require bounded Mac/Linux wait and CPU execution")
    common = json.loads(
        (a.preflight_dir / "common_material_manifest.json").read_text(encoding="utf-8")
    )
    source = json.loads(a.views.read_text(encoding="utf-8"))
    if material_hash(source) != common["material_sha256"]:
        raise ValueError("Common evidence changed")
    require_observer_completion(a.vector_dir, common["material_sha256"])
    budget = pd.read_csv(
        a.preflight_dir / "budget_by_author.csv", dtype={"author_id": str}
    )
    safe_source = exclude_reference(source, budget)
    predecessor = json.loads(
        (a.predecessor_dir / "batch_manifest.json").read_text(encoding="utf-8")
    )
    if predecessor["protocol"] != "VM05_STRONG_SPARSE_SOURCE_CONTROLS_V1":
        raise ValueError("Unexpected predecessor type")
    code = scoring_commitment()
    names = (
        "scripts/run_grain_vector_mechanism_neural_source_queue.py",
        "scripts/run_grain_vector_mechanism_sparse_baselines.py",
        "scripts/summarize_grain_vector_external_replay.py",
        "scripts/summarize_grain_vector_external_links.py",
    )
    fingerprint = code | {name: digest(ROOT / name) for name in names}
    out = a.output_dir.resolve()
    if not out.is_relative_to(ROOT / "results"):
        raise ValueError("Private output root required")
    out.mkdir(parents=True, exist_ok=True)
    with execution_lock(out) as lease:
        fields = {
            "protocol": "VM05_FULL_NEURAL_SOURCE_QUEUE_V1",
            "cases": cases(),
            "scoring_code": fingerprint,
            "views_sha256": digest(a.views),
            "vector_manifest_sha256": digest(a.vector_dir / "manifest.json"),
            "preflight_manifest_sha256": digest(a.preflight_dir / "manifest.json"),
            "source_parent_sha256": digest(a.parent_dir / "manifest.json"),
            "predecessor_manifest_sha256": digest(
                a.predecessor_dir / "batch_manifest.json"
            ),
            "predecessor_pid": a.predecessor_pid,
            "fit_seconds": 3600,
            "replay_seconds": 600,
            "child_rss_limit": 8 * 1024**3,
            "bridge_outcomes_used": False,
        }
        fields["identity"] = material_hash(fields)
        manifest = out / "batch_manifest.json"
        if manifest.exists():
            if (
                not a.resume_completed
                or json.loads(manifest.read_text(encoding="utf-8")) != fields
            ):
                raise ValueError(
                    "Existing neural queue needs identical explicit resume"
                )
        elif has_substantive_outputs(out):
            raise ValueError("Fresh neural queue output required")
        else:
            write_json(manifest, fields)
            archive_source(ROOT, out / "source_snapshot.zip")
        lease["started"] = True
        waited = time.monotonic()
        while not predecessor_ready(a.predecessor_dir, a.predecessor_pid):
            seconds = time.monotonic() - waited
            write_json(
                out / "batch_status.json",
                {
                    "status": "WAITING_FOR_VERIFIED_CPU_OWNER",
                    "pid": os.getpid(),
                    "predecessor_pid": a.predecessor_pid,
                    "wait_seconds": seconds,
                    "new_fits": 0,
                    "bridge_labels_read": False,
                },
            )
            if seconds > a.max_wait_seconds:
                write_json(
                    out / "batch_status.json",
                    {
                        "status": "PAUSED_WAIT_BUDGET",
                        "pid": os.getpid(),
                        "predecessor_pid": a.predecessor_pid,
                        "new_fits": 0,
                        "bridge_labels_read": False,
                    },
                )
                return
            time.sleep(30)
        # Hold the predecessor's existing lock too, preventing an overlapping manual resume.
        with (a.predecessor_dir / ".execution.lock").open("a") as resource_lock:
            fcntl.flock(resource_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if not predecessor_ready(a.predecessor_dir, a.predecessor_pid):
                raise RuntimeError("CPU predecessor resumed before lock handoff")
            started, completed = time.monotonic(), []
            (out / "job_logs").mkdir(exist_ok=True)
            for c in fields["cases"]:
                if any(digest(ROOT / name) != sha for name, sha in fingerprint.items()):
                    raise ValueError("Frozen CPU scoring code changed")
                folder = out / c["id"]
                replay = out / (c["id"] + "_replay")
                if not folder.exists():
                    if a.max_seconds - (time.monotonic() - started) < 4260:
                        write_json(
                            out / "batch_status.json",
                            {
                                "status": "PAUSED_BUDGET",
                                "pid": os.getpid(),
                                "completed": completed,
                                "next_case": c["id"],
                                "planned": len(fields["cases"]),
                            },
                        )
                        return
                    cmd = [
                        sys.executable,
                        "-u",
                        str(
                            ROOT
                            / "scripts/run_grain_vector_mechanism_external_links.py"
                        ),
                    ]
                    for key in (
                        "views",
                        "preflight_dir",
                        "parent_dir",
                        "prepared_root",
                        "vector_dir",
                    ):
                        cmd += [
                            "--" + key.replace("_", "-"),
                            str(getattr(a, key).resolve()),
                        ]
                    cmd += [
                        "--task",
                        c["task"],
                        "--feature",
                        c["feature"],
                        "--arm",
                        c["arm"],
                        "--exclude-reference",
                        "--output-dir",
                        str(folder),
                        "--max-seconds",
                        "3600",
                    ]

                    def update(pid, elapsed, rss, peak, case_id=c["id"]):
                        write_json(
                            out / "batch_status.json",
                            {
                                "status": "RUNNING",
                                "pid": os.getpid(),
                                "current": case_id,
                                "child_pid": pid,
                                "seconds": elapsed,
                                "rss_bytes": rss,
                                "completed": completed,
                                "planned": len(fields["cases"]),
                                "bridge_labels_read": False,
                            },
                        )

                    with (out / "job_logs" / (c["id"] + ".log")).open(
                        "x", encoding="utf-8"
                    ) as log:
                        resource = bounded_child(cmd, log, 3600, 8 * 1024**3, update)
                    write_json(folder / "supervisor_resource.json", resource)
                result = load_oof(folder)
                validate_result(result, c, safe_source, common["material_sha256"])
                if not replay.exists():
                    cmd = [
                        sys.executable,
                        "-u",
                        str(ROOT / "scripts/summarize_grain_vector_external_replay.py"),
                        "--runs",
                        str(folder),
                        "--views",
                        str(a.views.resolve()),
                        "--preflight-dir",
                        str(a.preflight_dir.resolve()),
                        "--vector-dir",
                        str(a.vector_dir.resolve()),
                        "--output-dir",
                        str(replay),
                    ]

                    def replay_update(pid, elapsed, rss, peak, case_id=c["id"]):
                        write_json(
                            out / "batch_status.json",
                            {
                                "status": "REPLAYING",
                                "pid": os.getpid(),
                                "current": case_id,
                                "child_pid": pid,
                                "seconds": elapsed,
                                "rss_bytes": rss,
                                "completed": completed,
                                "planned": len(fields["cases"]),
                                "bridge_labels_read": False,
                            },
                        )

                    with (out / "job_logs" / (c["id"] + "_replay.log")).open(
                        "x", encoding="utf-8"
                    ) as log:
                        bounded_child(cmd, log, 600, 8 * 1024**3, replay_update)
                rs = json.loads((replay / "summary.json").read_text(encoding="utf-8"))
                records = pd.read_csv(replay / "replay_by_fold.csv")
                if (
                    rs["status"] != "COMPLETE"
                    or rs["fold_models"] != 5
                    or not math.isfinite(rs["max_absolute_error"])
                    or rs["max_absolute_error"] > 1e-10
                    or rs["new_label_values_read"]
                    or rs["bridge_labels_read"]
                    or rs["model_refits"]
                    or len(records) != 5
                    or not records["max_absolute_error"].map(math.isfinite).all()
                    or (records["max_absolute_error"] > 1e-10).any()
                    or set(records["run"]) != {c["id"]}
                    or set(records["fold"]) != set(range(5))
                ):
                    raise ValueError("Incomplete saved-head replay")
                completed.append(
                    {
                        "case": c["id"],
                        "identity": result["manifest"]["identity"],
                        "score": result["summary"]["score"],
                        "authors": result["summary"]["authors"],
                        "replayed_folds": 5,
                        "replay_max": rs["max_absolute_error"],
                    }
                )
                write_json(
                    out / "batch_status.json",
                    {
                        "status": "RUNNING",
                        "pid": os.getpid(),
                        "completed": completed,
                        "planned": len(fields["cases"]),
                        "child_pid": None,
                        "bridge_labels_read": False,
                    },
                )
            write_json(
                out / "batch_status.json",
                {
                    "status": "COMPLETE",
                    "pid": os.getpid(),
                    "completed": completed,
                    "planned": len(fields["cases"]),
                    "child_pid": None,
                    "bridge_labels_read": False,
                },
            )


def main():
    """Launch a bounded verified wait, then the unchanged source head and independent replays."""
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "views",
        "preflight-dir",
        "parent-dir",
        "prepared-root",
        "vector-dir",
        "predecessor-dir",
        "output-dir",
    ):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--predecessor-pid", type=int, required=True)
    p.add_argument("--max-wait-seconds", type=int, default=10800)
    p.add_argument("--max-seconds", type=int, default=28800)
    p.add_argument("--resume-completed", action="store_true")
    a = p.parse_args()
    try:
        run(a)
    except Exception as e:
        # Do not overwrite another process's status when exclusive locking was refused.
        if not isinstance(e, BlockingIOError):
            print(
                json.dumps(
                    {
                        "status": "FAILED_STOP",
                        "error_type": type(e).__name__,
                        "error": str(e),
                    }
                ),
                flush=True,
            )
        raise


if __name__ == "__main__":
    main()
