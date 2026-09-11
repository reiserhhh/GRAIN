#!/usr/bin/env python3
"""Portable NEXT90 audit; retain the frozen primary and exclude verified AppleDouble."""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_joint_moment_nullspace as run


def logical_files(folder):
    """Exclude only paired files with the AppleDouble magic and version header."""
    names, metadata = set(), 0
    for path in folder.iterdir():
        if path.name.startswith("._"):
            counterpart = folder / path.name[2:]
            with path.open("rb") as handle:
                header = handle.read(8)
            if not counterpart.is_file() or header != bytes.fromhex("0005160700020000"):
                raise ValueError("Unrecognized metadata or orphan sidecar")
            metadata += 1
        else:
            names.add(path.name)
    return names, metadata


def audit(out=run.OUT):
    """Audit unchanged experiment data; account separately for the portability fix."""
    m = run.verify(out)
    states = [run.old.read(out / f / "freeze.json") for f in ("analysis", "replay")]
    left, right = states
    if any(
        s["status"] != "COMPLETE" or s["counts"] != run.fn.COUNTS or s["extra_denials"]
        for s in states
    ):
        raise ValueError("Both complete stages required")
    if left["files"] != right["files"]:
        raise ValueError("Exact CSV and author-unit replay required")
    ignored = 0
    for folder, stage in zip(("analysis", "replay"), states, strict=True):
        run.old.verify_hashes(out / folder, stage["files"])
        names, count = logical_files(out / folder)
        if names != set(stage["files"]) | {"freeze.json"}:
            raise ValueError("Unexpected logical experiment files")
        ignored += count
    start = time.process_time()
    job = run.old.read(out / "private_jobs.json")[0]
    key = job["query_keys"][0]
    data = run.old.arrays(run.old.OLD / "private_inputs" / (key + ".npz"))
    grad = run.old.selected(
        run.old.NATIVE / "private_packets" / (key + ".npz"), ("body_gradient",)
    )["body_gradient"].astype(float)
    a, q, _ = run.fn.basis(data["grain"])
    expected = a @ np.linalg.lstsq(a, grad, rcond=None)[0]
    error = float(
        np.linalg.norm(expected - q @ (q.T @ grad)) / max(1.0, np.linalg.norm(grad))
    )
    raw = {
        k: pd.read_csv(
            out / "analysis" / ("private_" + k + ".csv"), keep_default_na=False
        )
        for k in run.fn.TABLES
    }
    for frame in raw.values():
        for k in frame:
            if k not in ("cohort", "author_key", "query", "map", "factor", "part"):
                frame[k] = pd.to_numeric(frame[k], errors="raise")
    recomputed = run.fn.aggregate(raw)
    for name in run.fn.TABLES:
        saved = pd.read_csv(out / "analysis" / (name + ".csv"), keep_default_na=False)
        pd.testing.assert_frame_equal(
            saved, recomputed[name], check_dtype=False, atol=1e-11, rtol=1e-11
        )
    if set(raw["query_geometry"].part) != {"native", "visible", "null"}:
        raise ValueError("Literal null category was lost in CSV parsing")
    if error > 1e-9 or max(left["checks"].values()) > 1e-8:
        raise ValueError("Numerical identities failed")
    total = run.used(out) + time.process_time() - start
    if total > run.CAP:
        raise TimeoutError("Canonical including replay exceeded60CPU")
    result = {
        "status": "PASS",
        "primary_commit": m["commit"],
        "corrected_auditor_sha256": run.old.sha(Path(__file__)),
        "primary_audit_failure": "Exact directory comparison counted macOS AppleDouble sidecars; numeric analysis and replay were already complete.",
        "verified_appledouble_sidecars_ignored": ignored,
        "numeric_outputs_changed": False,
        "contexts": 24,
        "authors": 12,
        "source_authors": 4,
        "evaluation_authors": 8,
        "numeric_cpu_seconds": total,
        "independent_projector_relative_error": error,
        "public_tables": 6,
        "public_counts": run.fn.COUNTS,
        "private_units": 12,
        "fits": 0,
        "native_calls": 0,
        "human_labels_read": False,
        "aggregate_recalculation": "PASS",
        "literal_null_category_preserved": True,
        "checks": left["checks"],
    }
    if (out / "audit.json").exists():
        raise FileExistsError("Preserve completed audit; do not overwrite")
    run.old.write(out / "audit.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        audit()
