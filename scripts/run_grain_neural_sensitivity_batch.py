#!/usr/bin/env python3
"""Execute the originally declared no-PCA Ridge and fixed-layer sensitivities."""

import argparse
import fcntl
import json
import os
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
from grain_research.vector_mechanisms.inventory import archive_source
from grain_research.vector_mechanisms.ready_comparison import digest, verified_packet
from scripts.run_grain_vector_mechanism_controls import write_json
from scripts.run_grain_vector_mechanism_neural_source_queue import (
    cases as primary_cases,
)
from scripts.run_grain_vector_mechanism_neural_source_queue import execution_lock
from scripts.run_grain_vector_mechanism_source_controls import scoring_commitment
from scripts.run_grain_vector_mechanism_sparse_baselines import (
    bounded_child,
    has_substantive_outputs,
)
from scripts.summarize_grain_vector_external_links import load_oof


def cases():
    """Nine original-dimension Big5 heads followed by45 predeclared layer checks."""
    arms = ("raw", "cue_mask", "placebo_mask")
    rows = [
        {"task": "big5", "arm": arm, "feature": feature, "pca": False}
        for arm in arms
        for feature in ("bge_large", "e5_large", "qwen_block_10")
    ]
    rows += [
        {"task": task, "arm": arm, "feature": feature, "pca": True}
        for arm in arms
        for task in ("big5", "EI_cont", "SN_cont", "TF_cont", "JP_cont")
        for feature in ("qwen_block_04", "qwen_block_16", "qwen_final_norm")
    ]
    return [
        r
        | {
            "id": f"{r['task']}_{r['arm']}_{r['feature']}_{'pca16' if r['pca'] else 'no_pca'}"
        }
        for r in rows
    ]


def validate_result(result, case, source, common_hash):
    """Validate the actual PCA flag and exact official source membership, without recoding metadata."""
    group = "big5_only" if case["task"] == "big5" else "mbti_only"
    bundles = [b for b in source if b["source_group"] == group]
    m, s = result["manifest"], result["summary"]
    expected = {
        "protocol": "GRAIN_VM05_SOURCE_OOF_V1",
        "source_group": group,
        "exclude_reference": True,
        "thread_control": "retain",
        "drop_seed": 731,
        "permutation_seed": None,
        "seed": 731,
        "bridge_labels_used": False,
        "inner_cv": "3 author folds, seed731",
        "outer_cv": "5 source-specific official folds",
        "common_material_sha256": common_hash,
        "source_material_sha256": material_hash(bundles),
        "pca_applied": case["pca"],
        **{k: case[k] for k in ("task", "arm", "feature", "pca")},
    }
    if any(m.get(k) != v for k, v in expected.items()) or s["bridge_labels_used"]:
        raise ValueError("Sensitivity differs from frozen source/readout condition")
    folds = {b["author_id"]: b["official_folds"][case["task"]] for b in bundles}
    if set(result["ids"]) != set(folds) or any(
        int(f) != folds[u] for u, f in zip(result["ids"], result["folds"], strict=True)
    ):
        raise ValueError("Sensitivity source membership or official folds changed")


def validate_replay(folder, case):
    """Every saved official fold must reproduce with the declared projection setting."""
    s = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    f = pd.read_csv(folder / "replay_by_fold.csv")
    if (
        s["status"] != "COMPLETE"
        or s["fold_models"] != 5
        or s["new_label_values_read"]
        or s["bridge_labels_read"]
        or s["model_refits"]
        or not np.isfinite(s["max_absolute_error"])
        or s["max_absolute_error"] > 1e-10
        or len(f) != 5
        or set(f["run"]) != {case["id"]}
        or set(f["fold"]) != set(range(5))
        or not np.isfinite(f["max_absolute_error"]).all()
        or (f["max_absolute_error"] > 1e-10).any()
        or not f["pca_applied"].eq(case["pca"]).all()
    ):
        raise ValueError("Incomplete or wrong-projection saved-head replay")
    return s


def validate_primary(primary_dir, audit_dir, resource_dir):
    """Require all45 paired/replayed parents, not a live or merely paused primary queue."""
    m = json.loads((primary_dir / "batch_manifest.json").read_text(encoding="utf-8"))
    s = json.loads((primary_dir / "batch_status.json").read_text(encoding="utf-8"))
    am, audit = verified_packet(audit_dir)
    resource = json.loads(
        (resource_dir / "batch_manifest.json").read_text(encoding="utf-8")
    )
    if (
        m["protocol"] != "VM05_FULL_NEURAL_SOURCE_QUEUE_V1"
        or m["cases"] != primary_cases()
        or material_hash({k: v for k, v in m.items() if k != "identity"})
        != m["identity"]
        or s["status"] != "COMPLETE"
        or s.get("child_pid")
        or len(s["completed"]) != 45
        or len({r["case"] for r in s["completed"]}) != 45
        or {r["case"] for r in s["completed"]} != {r["id"] for r in primary_cases()}
        or audit["heads"] != 45
        or audit["pending_groups"]
        or len(audit["accepted_groups"]) != 15
        or am["batch_identity"] != m["identity"]
        or len(am["inputs"]) != 45
        or set(am["inputs"]) != {r["id"] for r in primary_cases()}
        or audit["bridge_labels_read"]
        or audit["model_refits"]
        or resource["protocol"] != "VM05_STRONG_SPARSE_SOURCE_CONTROLS_V1"
        or digest(resource_dir / "batch_manifest.json")
        != m["predecessor_manifest_sha256"]
    ):
        raise ValueError(
            "Complete primary comparison and original CPU resource required"
        )
    if any(digest(ROOT / name) != sha for name, sha in m["scoring_code"].items()):
        raise ValueError("Original primary scoring code changed")
    for hashes in am["inputs"].values():
        for name, sha in hashes.items():
            path = (primary_dir / name).resolve()
            if not path.is_relative_to(primary_dir.resolve()) or digest(path) != sha:
                raise ValueError("Audited primary source artifact changed")
    return m, am


def run(a):
    """Own all CPU leases, then fit/replay each fixed sensitivity within explicit budgets."""
    if not 4260 <= a.max_seconds <= 28800:
        raise ValueError(
            "Require bounded invocation with one complete fit/replay allowance"
        )
    out = a.output_dir.resolve()
    if not out.is_relative_to(ROOT / "results"):
        raise ValueError("Private result root required")
    primary, audit = validate_primary(a.primary_dir, a.audit_dir, a.resource_dir)
    common = json.loads((a.preflight_dir / "common_material_manifest.json").read_text())
    source = json.loads(a.views.read_text(encoding="utf-8"))
    for key, path in (
        ("views_sha256", a.views),
        ("preflight_manifest_sha256", a.preflight_dir / "manifest.json"),
        ("source_parent_sha256", a.parent_dir / "manifest.json"),
        ("vector_manifest_sha256", a.vector_dir / "manifest.json"),
    ):
        if digest(path) != primary[key]:
            raise ValueError("Sensitivity evidence differs from primary input")
    if material_hash(source) != common["material_sha256"]:
        raise ValueError("Common material mismatch")
    source = exclude_reference(
        source,
        pd.read_csv(a.preflight_dir / "budget_by_author.csv", dtype={"author_id": str}),
    )
    code = scoring_commitment() | {
        name: digest(ROOT / name)
        for name in (
            "scripts/run_grain_neural_sensitivity_batch.py",
            "scripts/run_grain_vector_mechanism_neural_source_queue.py",
            "scripts/run_grain_vector_mechanism_sparse_baselines.py",
            "scripts/summarize_grain_vector_external_replay.py",
            "scripts/summarize_grain_vector_external_links.py",
        )
    }
    out.mkdir(parents=True, exist_ok=True)
    with execution_lock(out) as lease, ExitStack() as locks:
        for folder in (a.resource_dir, a.primary_dir):
            handle = locks.enter_context((folder / ".execution.lock").open("a"))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        validate_primary(a.primary_dir, a.audit_dir, a.resource_dir)
        manifest = {
            "protocol": "VM05_FROZEN_NEURAL_SENSITIVITIES_V1",
            "cases": cases(),
            "primary_identity": primary["identity"],
            "primary_audit_identity": audit["identity"],
            "code_sha256": code,
            "fit_seconds": 3600,
            "replay_seconds": 600,
            "child_rss_limit": 8 * 1024**3,
            "bridge_labels_read": False,
        }
        manifest["identity"] = material_hash(manifest)
        path = out / "batch_manifest.json"
        if path.exists():
            if not a.resume_completed or json.loads(path.read_text()) != manifest:
                raise ValueError(
                    "Existing sensitivity batch needs identical explicit resume"
                )
        elif has_substantive_outputs(out):
            raise ValueError("Fresh sensitivity output required")
        else:
            write_json(path, manifest)
            archive_source(ROOT, out / "source_snapshot.zip")
            with zipfile.ZipFile(
                out / "source_snapshot.zip", "a", zipfile.ZIP_DEFLATED
            ) as z:
                for name in code:
                    if name not in z.namelist():
                        z.write(ROOT / name, name)
        lease["started"] = True
        started, completed = time.monotonic(), []
        (out / "job_logs").mkdir(exist_ok=True)

        def status(state, **fields):
            write_json(
                out / "batch_status.json",
                {
                    "status": state,
                    "pid": os.getpid(),
                    "completed": completed,
                    "planned": 54,
                    "bridge_labels_read": False,
                    **fields,
                },
            )

        for c in manifest["cases"]:
            if any(digest(ROOT / name) != sha for name, sha in code.items()):
                raise ValueError("Frozen sensitivity scoring code changed")
            folder, replay = out / c["id"], out / (c["id"] + "_replay")
            for phase, target, seconds in (
                ("FIT", folder, 3600),
                ("REPLAY", replay, 600),
            ):
                if target.exists():
                    continue
                if a.max_seconds - (time.monotonic() - started) < (
                    4260 if phase == "FIT" else 660
                ):
                    status(
                        "PAUSED_BUDGET",
                        child_pid=None,
                        next_case=c["id"],
                        next_phase=phase,
                    )
                    return
                if phase == "FIT":
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
                    for key in ("task", "arm", "feature"):
                        cmd += ["--" + key, c[key]]
                    cmd += ["--exclude-reference", "--max-seconds", "3600"]
                    if not c["pca"]:
                        cmd.append("--no-pca")
                else:
                    result = load_oof(folder)
                    validate_result(result, c, source, common["material_sha256"])
                    cmd = [
                        sys.executable,
                        "-u",
                        str(ROOT / "scripts/summarize_grain_vector_external_replay.py"),
                        "--runs",
                        str(folder),
                    ]
                    for key in ("views", "preflight_dir", "vector_dir"):
                        cmd += [
                            "--" + key.replace("_", "-"),
                            str(getattr(a, key).resolve()),
                        ]
                cmd += ["--output-dir", str(target)]

                def update(pid, elapsed, rss, peak, stage=phase, case_id=c["id"]):
                    status(
                        stage,
                        current=case_id,
                        child_pid=pid,
                        seconds=elapsed,
                        rss_bytes=rss,
                    )

                with (out / "job_logs" / (c["id"] + "_" + phase + ".log")).open(
                    "x", encoding="utf-8"
                ) as log:
                    resource = bounded_child(cmd, log, seconds, 8 * 1024**3, update)
                if phase == "FIT":
                    write_json(folder / "supervisor_resource.json", resource)
            result = load_oof(folder)
            validate_result(result, c, source, common["material_sha256"])
            rs = validate_replay(replay, c)
            if (
                digest(folder / "private_full_source_head.joblib")
                != result["summary"]["full_head_sha256"]
            ):
                raise ValueError("Full sensitivity source head checksum mismatch")
            completed.append(
                {
                    "case": c["id"],
                    "identity": result["manifest"]["identity"],
                    "authors": len(result["ids"]),
                    "score": result["summary"]["score"],
                    "replayed_folds": 5,
                    "replay_max": rs["max_absolute_error"],
                }
            )
            status("RUNNING", child_pid=None)
        status("COMPLETE", child_pid=None)


def main():
    """Accept a complete source parent, never reduce the sensitivity grid after outcomes."""
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "views",
        "preflight-dir",
        "parent-dir",
        "prepared-root",
        "vector-dir",
        "primary-dir",
        "audit-dir",
        "resource-dir",
        "output-dir",
    ):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--max-seconds", type=int, default=28800)
    p.add_argument("--resume-completed", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
