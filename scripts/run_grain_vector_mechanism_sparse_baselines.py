#!/usr/bin/env python3
"""Bounded CPU sparse MBTI reference using the unchanged VM05 source-head protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.inventory import archive_source
from scripts.run_grain_vector_mechanism_controls import write_json
from scripts.run_grain_vector_mechanism_source_controls import scoring_commitment
from scripts.summarize_grain_vector_external_links import load_oof


def cases():
    """All four independent axes and all three predeclared text arms."""
    return [
        {"id": f"{task}_{arm}", "task": task, "arm": arm}
        for task in ("EI_cont", "SN_cont", "TF_cont", "JP_cont")
        for arm in ("raw", "cue_mask", "placebo_mask")
    ]


def current_rss(pid):
    """Read only our owned CPU child's RSS on macOS/Linux; no extra dependency."""
    result = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(pid)],
        capture_output=True,
        text=True,
        check=False,
    )
    if not result.stdout.strip():
        return None
    return int(result.stdout.strip()) * 1024


def bounded_child(command, log, seconds, max_rss_bytes, update):
    """Bound time/memory and stop only the subprocess created by this call."""
    start, peak = time.monotonic(), 0
    with subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT) as child:
        try:
            while child.poll() is None:
                elapsed = time.monotonic() - start
                rss = current_rss(child.pid)
                peak = max(peak, rss or 0)
                update(child.pid, elapsed, rss, peak)
                if elapsed > seconds or peak > max_rss_bytes:
                    raise RuntimeError("Owned sparse child exceeded time/RSS budget")
                try:
                    child.wait(timeout=min(10, max(0.05, seconds - elapsed)))
                except subprocess.TimeoutExpired:
                    pass
            if child.returncode:
                raise RuntimeError("Sparse child failed; inspect its scoped log")
        except BaseException:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            raise
    return {"seconds": time.monotonic() - start, "peak_sampled_rss_bytes": peak}


def main():
    """Run serial immutable conditions; resume only complete verified child artifacts."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--views", type=Path, required=True)
    p.add_argument("--preflight-dir", type=Path, required=True)
    p.add_argument("--parent-dir", type=Path, required=True)
    p.add_argument("--prepared-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--exclude-reference", action="store_true")
    p.add_argument("--max-seconds", type=int, default=7200)
    p.add_argument("--max-child-rss-gb", type=float, default=8)
    p.add_argument("--resume-completed", action="store_true")
    a = p.parse_args()
    if not 600 <= a.max_seconds <= 28800 or not 1 <= a.max_child_rss_gb <= 8:
        raise ValueError("Require bounded CPU time and at most 8 GiB per sparse child")
    if sys.platform not in ("darwin", "linux"):
        raise ValueError("This CPU supervisor requires macOS/Linux ps and file locking")
    import fcntl

    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / ".execution.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    fields = {
        "protocol": "VM05_STRONG_SPARSE_MBTI_V1",
        "cases": cases(),
        "scoring_code": scoring_commitment(),
        "supervisor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "views_sha256": hashlib.sha256(a.views.read_bytes()).hexdigest(),
        "preflight_sha256": hashlib.sha256(
            (a.preflight_dir / "manifest.json").read_bytes()
        ).hexdigest(),
        "exclude_reference": a.exclude_reference,
        "max_child_rss_bytes": int(a.max_child_rss_gb * 1024**3),
        "bridge_labels_read": False,
    }
    fields["identity"] = material_hash(fields)
    path = out / "batch_manifest.json"
    if path.exists():
        if not a.resume_completed or json.loads(path.read_text()) != fields:
            raise ValueError("Existing sparse batch requires explicit identical resume")
    elif any(q.name != ".execution.lock" for q in out.iterdir()):
        raise ValueError("Use a new sparse output directory")
    else:
        write_json(path, fields)
        archive_source(ROOT, out / "source_snapshot.zip")
    (out / "job_logs").mkdir(exist_ok=True)
    start, completed, current = time.monotonic(), [], None
    try:
        for case in fields["cases"]:
            current = case["id"]
            folder = out / current
            if scoring_commitment() != fields["scoring_code"]:
                raise ValueError("Source scoring implementation changed during batch")
            if folder.exists():
                result = load_oof(folder)
                resource = json.loads((folder / "supervisor_resource.json").read_text())
            else:
                remaining = a.max_seconds - (time.monotonic() - start)
                # A ten-times-larger sparse corpus is not budgeted as a tiny dense head.
                if remaining < 1800:
                    write_json(
                        out / "batch_status.json",
                        {
                            "status": "PAUSED_BUDGET",
                            "completed": completed,
                            "current": current,
                            "planned": len(fields["cases"]),
                        },
                    )
                    return
                command = [
                    sys.executable,
                    "-u",
                    str(ROOT / "scripts/run_grain_vector_mechanism_external_links.py"),
                    "--feature",
                    "word_char",
                    "--task",
                    case["task"],
                    "--arm",
                    case["arm"],
                    "--output-dir",
                    str(folder),
                    "--max-seconds",
                    str(int(remaining) - 60),
                ]
                for flag in ("views", "preflight_dir", "parent_dir", "prepared_root"):
                    command += ["--" + flag.replace("_", "-"), str(getattr(a, flag))]
                if a.exclude_reference:
                    command += ["--exclude-reference"]

                def update(pid, elapsed, rss, peak, case_id=current):
                    write_json(
                        out / "batch_status.json",
                        {
                            "status": "RUNNING",
                            "current": case_id,
                            "child_pid": pid,
                            "child_seconds": elapsed,
                            "rss_bytes": rss,
                            "peak_sampled_rss_bytes": peak,
                            "completed": completed,
                            "planned": len(fields["cases"]),
                            "bridge_labels_read": False,
                        },
                    )

                with (out / "job_logs" / (current + ".log")).open(
                    "w", encoding="utf-8"
                ) as log:
                    resource = bounded_child(
                        command,
                        log,
                        remaining,
                        fields["max_child_rss_bytes"],
                        update,
                    )
                result = load_oof(folder)
                write_json(folder / "supervisor_resource.json", resource)
            m, s = result["manifest"], result["summary"]
            if (
                m["task"] != case["task"]
                or m["arm"] != case["arm"]
                or m["feature"] != "word_char"
                or m["pca_applied"]
                or m["exclude_reference"] != a.exclude_reference
                or m["permutation_seed"] is not None
                or m["thread_control"] != "retain"
                or s["bridge_labels_used"]
            ):
                raise ValueError(
                    "Completed sparse condition differs from the frozen design"
                )
            completed.append(
                {
                    "case": current,
                    "authors": s["authors"],
                    "auc": s["score"],
                    "mean_fold_auc": result["mean_fold_score"],
                    **resource,
                }
            )
            print(json.dumps(completed[-1] | {"completed": len(completed)}), flush=True)
        write_json(
            out / "batch_status.json",
            {
                "status": "COMPLETE",
                "completed": completed,
                "planned": len(fields["cases"]),
                "seconds": time.monotonic() - start,
            },
        )
    except Exception as error:
        write_json(
            out / "batch_status.json",
            {
                "status": "FAILED_STOP",
                "current": current,
                "completed": completed,
                "error": str(error),
                "planned": len(fields["cases"]),
            },
        )
        raise


if __name__ == "__main__":
    main()
