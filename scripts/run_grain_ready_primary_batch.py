#!/usr/bin/env python3
"""Run the raw reference-free same-ready comparison after the sensitivity CPU lease."""

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

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.external_heads import predict_head
from grain_research.vector_mechanisms.inventory import archive_source
from grain_research.vector_mechanisms.ready_comparison import (
    digest,
    load_registry_inputs,
    verified_packet,
)
from grain_research.vector_mechanisms.ready_registry import TASKS, source_indices
from scripts.run_grain_neural_sensitivity_batch import cases as sensitivity_cases
from scripts.run_grain_vector_mechanism_controls import write_json
from scripts.run_grain_vector_mechanism_neural_source_queue import execution_lock
from scripts.run_grain_vector_mechanism_source_controls import scoring_commitment
from scripts.run_grain_vector_mechanism_sparse_baselines import (
    bounded_child,
    has_substantive_outputs,
)
from scripts.summarize_grain_vector_external_links import load_oof

COHORT = "raw_ready_reference_free"
FEATURES = (
    "word_char",
    "bge_large",
    "e5_large",
    "qwen_block_10",
    "grain_qwen_block_10",
    "grain_bge_large",
)
INPUT_FLAGS = (
    "views",
    "registry_dir",
    "registry_audit_dir",
    "profile_dir",
    "preflight_dir",
    "parent_dir",
    "prepared_root",
    "vector_dir",
)
CODE = (
    "scripts/run_grain_ready_primary_batch.py",
    "scripts/run_grain_same_ready_comparison.py",
    "grain_research/vector_mechanisms/ready_comparison.py",
    "grain_research/vector_mechanisms/ready_registry.py",
    "grain_research/vector_mechanisms/external_readouts.py",
    "grain_research/vector_mechanisms/external_readout_cache.py",
    "scripts/run_grain_vector_mechanism_external_queue.py",
    "scripts/run_grain_vector_mechanism_neural_source_queue.py",
    "scripts/run_grain_vector_mechanism_sparse_baselines.py",
)


def cases():
    """Thirty original comparators/increments plus four genuinely changed MBTI profiles."""
    return [
        {"task": t, "feature": f, "id": f"{t}_raw_{f}"}
        for t in TASKS
        for f in (("sorted_distances",) if t != "big5" else ()) + FEATURES
    ]


def predecessor_done(folder):
    """Never replace a running owner or mistake a budget pause for full completion."""
    s = json.loads((folder / "batch_status.json").read_text(encoding="utf-8"))
    if s["status"] in ("FAILED_STOP", "PAUSED_BUDGET"):
        raise ValueError(
            "Sensitivity predecessor requires completion before this stage"
        )
    if s["status"] != "COMPLETE":
        return False
    expected = {c["id"] for c in sensitivity_cases()}
    if (
        s.get("child_pid")
        or len(s["completed"]) != 54
        or {c["case"] for c in s["completed"]} != expected
        or any(
            c["replayed_folds"] != 5
            or not np.isfinite(c["replay_max"])
            or c["replay_max"] > 1e-10
            for c in s["completed"]
        )
    ):
        raise ValueError("Incomplete sensitivity fit/replay family")
    proc = subprocess.run(
        ["ps", "-p", str(s["pid"]), "-o", "args="],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 1 and not proc.stdout.strip():
        return True
    if proc.returncode or "run_grain_neural_sensitivity_batch.py" not in proc.stdout:
        raise ValueError("Sensitivity owner PID cannot be verified")
    return False


def validate_ready(folder, case, source, common, reg, registry):
    """Accept a complete source-only ready refit, exact author folds and saved replay."""
    m, s = verified_packet(folder)
    result = load_oof(folder)
    ix = source_indices(reg, COHORT, case["task"])
    bundles = [source[i] for i in ix]
    expected = {
        "protocol": "VM05_SAME_READY_COMPARISON_V1",
        "cohort": COHORT,
        "task": case["task"],
        "feature": case["feature"],
        "arm": "raw",
        "registry_identity": registry["identity"],
        "source_material_sha256": material_hash(bundles),
        "common_material_sha256": common["material_sha256"],
        "scaler_fit": "inner_source_only",
        "pca": True,
        "pca_applied": case["feature"] != "word_char",
        "thread_control": "retain",
        "seed": 731,
        "permutation_seed": None,
        "bridge_labels_used": False,
        "preflight_only": False,
        "separate_calibration_fitted": False,
        "readout": "body_token" if "qwen_" in case["feature"] else "not_qwen",
    }
    folds = {b["author_id"]: b["official_folds"][case["task"]] for b in bundles}
    r = pd.read_csv(folder / "saved_head_replay.csv")
    if (
        any(m.get(k) != v for k, v in expected.items())
        or set(result["ids"]) != set(folds)
        or any(
            int(f) != folds[u]
            for u, f in zip(result["ids"], result["folds"], strict=True)
        )
        or s["bridge_labels_used"]
        or not np.isfinite(s["saved_fold_replay_max"])
        or s["saved_fold_replay_max"] > 1e-10
        or digest(folder / "private_full_source_head.joblib") != s["full_head_sha256"]
        or len(r) != 5
        or set(r["fold"]) != set(range(5))
        or not np.isfinite(r["max_absolute_error"]).all()
        or (r["max_absolute_error"] > 1e-10).any()
    ):
        raise ValueError(
            "Ready result differs from frozen source/refit/replay contract"
        )
    return {
        "case": case["id"],
        "identity": m["identity"],
        "authors": len(folds),
        "score": s["score"],
        "replayed_folds": 5,
        "replay_max": s["saved_fold_replay_max"],
    }


def reuse_big5_profile(folder, source, reg, profiles):
    """Reuse and independently replay the existing identical Big5 profile, never refit it."""
    ix = source_indices(reg, COHORT, "big5")
    if not np.array_equal(ix, source_indices(reg, "raw_ready", "big5")):
        raise ValueError("Big5 historical profile is not the identical cohort")
    r = load_oof(folder)
    m, s = r["manifest"], r["summary"]
    x = profiles["raw_sorted_distances"][ix]
    import hashlib

    if (
        any(
            m.get(k) != v
            for k, v in {
                "protocol": "VM05_READY_PROFILE_CONDITION_V1",
                "task": "big5",
                "feature": "sorted_distances",
                "arm": "raw",
                "pca": True,
                "cohort": "raw_ready",
                "permutation_seed": None,
                "bridge_labels_used": False,
                "source_material_sha256": material_hash([source[i] for i in ix]),
                "feature_input_sha256": hashlib.sha256(
                    np.ascontiguousarray(x).tobytes()
                ).hexdigest(),
            }.items()
        )
        or set(r["ids"]) != set(reg["author_ids"][ix])
        or s["bridge_labels_used"]
        or digest(folder / "private_full_source_head.joblib") != s["full_head_sha256"]
    ):
        raise ValueError("Existing Big5 profile differs from the identical parent")
    index = {u: j for j, u in enumerate(r["ids"])}
    order = np.asarray([index[u] for u in reg["author_ids"][ix]])
    folds = reg["big5_fold"][ix]
    if not np.array_equal(r["folds"][order], folds):
        raise ValueError("Existing Big5 official folds changed")
    errors = []
    for f in range(5):
        selected = folds == f
        pred = predict_head(
            joblib.load(folder / f"private_fold_{f}.joblib"), x[selected], "big5"
        )
        errors.append(float(np.max(abs(pred - r["p"][order][selected]))))
    if not np.isfinite(errors).all() or max(errors) > 1e-10:
        raise ValueError("Existing Big5 profile cannot be independently replayed")
    names = [
        "manifest.json",
        "summary.json",
        "private_oof_predictions.csv",
        "private_full_source_head.joblib",
    ] + [f"private_fold_{f}.joblib" for f in range(5)]
    return {
        "identity": m["identity"],
        "authors": len(ix),
        "score": s["score"],
        "max_replay_error": max(errors),
        "replayed_folds": 5,
        "new_fits": 0,
        "inputs": {n: digest(folder / n) for n in names},
    }


def run(a):
    """Bound waiting and fitting separately; hold every earlier CPU lease during execution."""
    if not 60 <= a.max_wait_seconds <= 10800 or not 3660 <= a.max_seconds <= 28800:
        raise ValueError("Require bounded CPU wait/fit budgets")
    out = a.output_dir.resolve()
    if not out.is_relative_to(ROOT / "results"):
        raise ValueError("Private results path required")
    registry, _ = verified_packet(a.registry_dir)
    predecessor = json.loads((a.predecessor_dir / "batch_manifest.json").read_text())
    if (
        predecessor["protocol"] != "VM05_FROZEN_NEURAL_SENSITIVITIES_V1"
        or predecessor["cases"] != sensitivity_cases()
        or material_hash({k: v for k, v in predecessor.items() if k != "identity"})
        != predecessor["identity"]
    ):
        raise ValueError("Require the declared54-case predecessor")
    code = scoring_commitment() | {n: digest(ROOT / n) for n in CODE}
    input_paths = {
        "views": a.views,
        "preflight": a.preflight_dir / "manifest.json",
        "source_parent": a.parent_dir / "manifest.json",
        "vector_parent": a.vector_dir / "manifest.json",
        "registry_audit": a.registry_audit_dir / "summary.json",
        "primary_parent": a.primary_dir / "batch_manifest.json",
        "strong_parent": a.resource_dir / "batch_manifest.json",
        "sensitivity_parent": a.predecessor_dir / "batch_manifest.json",
    }
    parent_names = [
        "manifest.json",
        "summary.json",
        "private_oof_predictions.csv",
        "private_full_source_head.joblib",
    ] + [f"private_fold_{f}.joblib" for f in range(5)]
    input_paths.update(
        {"big5_profile/" + n: a.big5_profile_parent / n for n in parent_names}
    )
    input_hashes = {n: digest(p) for n, p in input_paths.items()}

    def require_unchanged():
        if any(digest(ROOT / n) != sha for n, sha in code.items()) or any(
            digest(input_paths[n]) != sha for n, sha in input_hashes.items()
        ):
            raise ValueError("Frozen ready scoring code or source parent changed")

    out.mkdir(parents=True, exist_ok=True)
    with execution_lock(out) as lease:
        m = {
            "protocol": "VM05_RAW_READY_PRIMARY_BATCH_V1",
            "cases": cases(),
            "cohort": COHORT,
            "arm": "raw",
            "pca": True,
            "readout": "body_token",
            "registry_identity": registry["identity"],
            "scoring_code": code,
            "predecessor_identity": predecessor["identity"],
            "input_sha256": input_hashes,
            "fit_seconds": 3600,
            "child_rss_limit": 8 * 1024**3,
            "big5_profile_parent": str(
                a.big5_profile_parent.resolve().relative_to(ROOT)
            ),
            "bridge_labels_read": False,
            "program_complete": False,
        }
        m["identity"] = material_hash(m)
        path = out / "batch_manifest.json"
        if path.exists():
            if not a.resume_completed or json.loads(path.read_text()) != m:
                raise ValueError(
                    "Existing same-ready run needs identical explicit resume"
                )
            prior_status = out / "batch_status.json"
            if prior_status.exists():
                prior = json.loads(prior_status.read_text(encoding="utf-8"))
                if prior.get("child_pid"):
                    proc = subprocess.run(
                        ["ps", "-p", str(prior["child_pid"]), "-o", "pid="],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    if proc.returncode != 1 or proc.stdout.strip():
                        raise ValueError(
                            "Previous owned child remains alive or unverifiable"
                        )
        elif has_substantive_outputs(out):
            raise ValueError("Fresh same-ready output required")
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

        def status(state, **kwargs):
            write_json(
                out / "batch_status.json",
                {
                    "status": state,
                    "pid": os.getpid(),
                    "planned": 34,
                    "completed": completed,
                    "bridge_labels_read": False,
                    "program_complete": False,
                    **kwargs,
                },
            )

        started = time.monotonic()
        while not predecessor_done(a.predecessor_dir):
            require_unchanged()
            seconds = time.monotonic() - started
            status(
                "WAITING_FOR_SENSITIVITY_COMPLETION",
                wait_seconds=seconds,
                child_pid=None,
            )
            if seconds >= a.max_wait_seconds:
                status("PAUSED_WAIT_BUDGET", child_pid=None)
                return
            time.sleep(30)
        with ExitStack() as locks:
            for parent in (a.predecessor_dir, a.resource_dir, a.primary_dir):
                handle = locks.enter_context((parent / ".execution.lock").open("a"))
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if not predecessor_done(a.predecessor_dir):
                raise ValueError("Sensitivity restarted during CPU handoff")
            require_unchanged()
            source, common, reg, profiles, loaded_registry = load_registry_inputs(
                a.views,
                a.registry_dir,
                a.registry_audit_dir,
                a.profile_dir,
                a.preflight_dir,
            )
            if loaded_registry["identity"] != registry["identity"]:
                raise ValueError("Ready registry changed during CPU wait")
            reuse = reuse_big5_profile(a.big5_profile_parent, source, reg, profiles)
            write_json(out / "reused_big5_profile.json", reuse)
            started = time.monotonic()
            (out / "job_logs").mkdir(exist_ok=True)
            for c in m["cases"]:
                require_unchanged()
                folder = out / c["id"]
                if not folder.exists():
                    if a.max_seconds - (time.monotonic() - started) < 3660:
                        status("PAUSED_BUDGET", next_case=c["id"], child_pid=None)
                        return
                    folder.mkdir()
                    cmd = [
                        sys.executable,
                        "-u",
                        str(ROOT / "scripts/run_grain_same_ready_comparison.py"),
                    ]
                    for n in INPUT_FLAGS:
                        cmd += [
                            "--" + n.replace("_", "-"),
                            str(getattr(a, n).resolve()),
                        ]
                    cmd += [
                        "--cohort",
                        COHORT,
                        "--task",
                        c["task"],
                        "--feature",
                        c["feature"],
                        "--arm",
                        "raw",
                        "--output-dir",
                        str(folder),
                        "--worker",
                    ]

                    def update(pid, elapsed, rss, peak, case_id=c["id"]):
                        status(
                            "FIT",
                            current=case_id,
                            child_pid=pid,
                            seconds=elapsed,
                            rss_bytes=rss,
                        )

                    with (out / "job_logs" / (c["id"] + ".log")).open(
                        "x", encoding="utf-8"
                    ) as log:
                        resource = bounded_child(cmd, log, 3600, 8 * 1024**3, update)
                    write_json(folder / "supervisor_resource.json", resource)
                accepted = validate_ready(
                    folder / "fit", c, source, common, reg, registry
                )
                completed.append(accepted)
                status("CASE_COMPLETE", current=c["id"], child_pid=None)
                print(json.dumps(accepted), flush=True)
            status(
                "COMPLETE",
                child_pid=None,
                reused_profiles=1,
                seconds=time.monotonic() - started,
            )


def main():
    """Run only the fixed raw/reference-free stage; no mask, layer or outcome-based selection."""
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        *INPUT_FLAGS,
        "predecessor_dir",
        "primary_dir",
        "resource_dir",
        "big5_profile_parent",
        "output_dir",
    ):
        p.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    p.add_argument("--max-seconds", type=int, default=28800)
    p.add_argument("--max-wait-seconds", type=int, default=10800)
    p.add_argument("--resume-completed", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
