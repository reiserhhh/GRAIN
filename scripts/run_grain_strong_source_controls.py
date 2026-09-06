#!/usr/bin/env python3
"""Bounded strong sparse MBTI source controls with explicit completed baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.inventory import archive_source
from scripts.run_grain_vector_mechanism_controls import write_json
from scripts.run_grain_vector_mechanism_source_controls import cases as old_cases
from scripts.run_grain_vector_mechanism_source_controls import scoring_commitment
from scripts.run_grain_vector_mechanism_sparse_baselines import (
    bounded_child,
    has_substantive_outputs,
)
from scripts.summarize_grain_vector_external_links import load_oof


def cases():
    """Apply all84 frozen MBTI controls to the strong reference, without selection."""
    return [c | {"feature": "word_char"} for c in old_cases() if c["task"] != "big5"]


def validate_case(m, case, common, *, baseline=False):
    """Reject a wrong model, exposure branch, outcome permutation or source cohort."""
    expected = {
        "task": case["task"],
        "arm": case["arm"],
        "feature": "word_char",
        "source_group": "mbti_only",
        "common_material_sha256": common,
        "exclude_reference": not baseline,
        "thread_control": "retain" if baseline else case["mode"],
        "drop_seed": 731 if baseline else case["seed"],
        "permutation_seed": None,
        "pca_applied": False,
        "bridge_labels_used": False,
    }
    if any(m[k] != value for k, value in expected.items()):
        raise ValueError("Source result differs from frozen strong control design")


def baseline_index(paths, common, source_bundles):
    """Require every original axis/arm on identical source authors before controls."""
    expected = {(c["task"], c["arm"]) for c in cases()}
    source_ids = {b["author_id"] for b in source_bundles}
    source_hash = material_hash(source_bundles)
    if (
        not source_ids
        or len(source_ids) != len(source_bundles)
        or any(b["source_group"] != "mbti_only" for b in source_bundles)
    ):
        raise ValueError("Expected complete MBTI-only source cohort")
    found, references = {}, {}
    for path in paths:
        result = load_oof(path)
        m, s = result["manifest"], result["summary"]
        key = (m["task"], m["arm"])
        if key not in expected or key in found or s["bridge_labels_used"]:
            raise ValueError("Duplicate or unsupported baseline packet")
        validate_case(m, {"task": key[0], "arm": key[1]}, common, baseline=True)
        if (
            m["source_material_sha256"] != source_hash
            or set(result["ids"]) != source_ids
        ):
            raise ValueError("Baseline does not cover the exact complete source cohort")
        if key[0] in references:
            if any(
                not np.array_equal(result[k], references[key[0]][k])
                for k in ("ids", "y", "folds")
            ):
                raise ValueError("Baseline arms use different authors, labels or folds")
        else:
            references[key[0]] = {k: result[k] for k in ("ids", "y", "folds")}
        found[key] = {
            "task": key[0],
            "arm": key[1],
            "path": str(path.resolve()),
            "identity": m["identity"],
            "full_head_sha256": s["full_head_sha256"],
            "summary_sha256": hashlib.sha256(
                (path / "summary.json").read_bytes()
            ).hexdigest(),
        }
    if set(found) != expected:
        raise ValueError("All12 strong baseline packets are required")
    return [found[key] for key in sorted(found)]


def main():
    """Serialize immutable conditions, and resume only fully verified child outputs."""
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("views", "preflight-dir", "parent-dir", "prepared-root", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--baseline-runs", type=Path, nargs="+", required=True)
    p.add_argument("--max-seconds", type=int, default=28800)
    p.add_argument("--resume-completed", action="store_true")
    a = p.parse_args()
    if sys.platform not in ("darwin", "linux") or not 3660 <= a.max_seconds <= 28800:
        raise ValueError("Require a bounded Mac/Linux CPU batch")
    import fcntl

    out = a.output_dir.resolve()
    if not out.is_relative_to(ROOT / "results"):
        raise ValueError("Private research output required")
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / ".execution.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    common = json.loads((a.preflight_dir / "common_material_manifest.json").read_text())
    source = json.loads(a.views.read_text())
    if material_hash(source) != common["material_sha256"]:
        raise ValueError("Changed common source material")
    baselines = baseline_index(
        a.baseline_runs,
        common["material_sha256"],
        [b for b in source if b["source_group"] == "mbti_only"],
    )
    del source
    fields = {
        "protocol": "VM05_STRONG_SPARSE_SOURCE_CONTROLS_V1",
        "cases": cases(),
        "scoring_code": scoring_commitment(),
        "baselines": baselines,
        "supervisor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "resource_helper_sha256": hashlib.sha256(
            (
                ROOT / "scripts/run_grain_vector_mechanism_sparse_baselines.py"
            ).read_bytes()
        ).hexdigest(),
        "views_sha256": hashlib.sha256(a.views.read_bytes()).hexdigest(),
        "preflight_manifest_sha256": hashlib.sha256(
            (a.preflight_dir / "manifest.json").read_bytes()
        ).hexdigest(),
        "max_child_rss_bytes": 8 * 1024**3,
        "max_child_seconds": 3600,
        "bridge_outcomes_used": False,
    }
    fields["identity"] = material_hash(fields)
    path = out / "batch_manifest.json"
    if path.exists():
        if not a.resume_completed or json.loads(path.read_text()) != fields:
            raise ValueError("Existing batch requires identical explicit resume")
    elif has_substantive_outputs(out):
        raise ValueError("Fresh batch output required")
    else:
        write_json(path, fields)
        archive_source(ROOT, out / "source_snapshot.zip")
    (out / "job_logs").mkdir(exist_ok=True)
    start, completed, current = time.monotonic(), [], None
    try:
        for case in fields["cases"]:
            current = case["id"]
            if scoring_commitment() != fields["scoring_code"]:
                raise ValueError("Active source scorer changed")
            folder = out / current
            if folder.exists():
                result = load_oof(folder)
                resource = json.loads((folder / "supervisor_resource.json").read_text())
            else:
                remaining = a.max_seconds - (time.monotonic() - start)
                if remaining < 3660:
                    write_json(
                        out / "batch_status.json",
                        {
                            "status": "PAUSED_BUDGET",
                            "pid": os.getpid(),
                            "current": current,
                            "completed": completed,
                            "planned": len(fields["cases"]),
                        },
                    )
                    return
                command = [
                    sys.executable,
                    "-u",
                    str(ROOT / "scripts/run_grain_vector_mechanism_external_links.py"),
                ]
                for flag in ("views", "preflight_dir", "parent_dir", "prepared_root"):
                    command += ["--" + flag.replace("_", "-"), str(getattr(a, flag))]
                command += [
                    "--feature",
                    "word_char",
                    "--task",
                    case["task"],
                    "--arm",
                    case["arm"],
                    "--thread-control",
                    case["mode"],
                    "--drop-seed",
                    str(case["seed"]),
                    "--exclude-reference",
                    "--output-dir",
                    str(folder),
                    "--max-seconds",
                    "3600",
                ]

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

                with (out / "job_logs" / (current + ".log")).open("w") as log:
                    resource = bounded_child(
                        command, log, 3600, fields["max_child_rss_bytes"], update
                    )
                result = load_oof(folder)
                write_json(folder / "supervisor_resource.json", resource)
            validate_case(result["manifest"], case, common["material_sha256"])
            if result["summary"]["bridge_labels_used"]:
                raise ValueError("Unexpected bridge outcomes")
            completed.append(
                {
                    "case": current,
                    "authors": result["summary"]["authors"],
                    "auc": result["summary"]["score"],
                    "mean_fold_auc": result["mean_fold_score"],
                    **resource,
                }
            )
            write_json(
                out / "batch_status.json",
                {
                    "status": "RUNNING",
                    "completed": completed,
                    "current": current,
                    "planned": len(fields["cases"]),
                    "seconds": time.monotonic() - start,
                },
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
        old = out / "batch_status.json"
        last = json.loads(old.read_text()) if old.exists() else {}
        write_json(
            out / "batch_status.json",
            {
                "status": "FAILED_STOP",
                "current": current,
                "completed": completed,
                "exception_type": type(error).__name__,
                "last_observation": last,
                "planned": len(fields["cases"]),
            },
        )
        raise


if __name__ == "__main__":
    main()
