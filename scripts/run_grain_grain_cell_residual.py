#!/usr/bin/env python3
"""Execute the frozen NEXT91 decomposition, without new neural calls or fits."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.vector_mechanisms import grain_cell_residual as fn
from grain_research.vector_mechanisms.stage_file_guard import install
from scripts import run_grain_joint_moment_nullspace as prev
from scripts.audit_grain_joint_moment_nullspace import logical_files

OUT = prev.old.BASE / "logic_grain_cell_residual_v1"
PLAN = ROOT / "docs/GRAIN_GRAIN_CELL_RESIDUAL_PLAN_V1.md"
CAP = 60


def write(path, value):
    """Serialize missing numeric values as null, never nonstandard JSON NaN."""

    def clean(v):
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [clean(x) for x in v]
        if isinstance(v, np.generic):
            v = v.item()
        return None if isinstance(v, float) and not np.isfinite(v) else v

    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def verify(out):
    """Verify frozen code, source input contracts and prepared job membership."""
    m = prev.old.read(out / "manifest.json")
    prev.old.verify_hashes(ROOT, m["inputs"])
    prev.old.verify_hashes(out, m["prepared"])
    return m


def used(out):
    """Count canonical numeric work, including replay, not file hashing or imports."""
    total = prev.old.read(out / "manifest.json")["prepare_cpu_seconds"]
    for stage in ("analysis", "replay"):
        path = out / stage / "freeze.json"
        if path.exists():
            total += prev.old.read(path)["cpu_seconds"]
    return total


def prepare(out):
    """Bind the audited NEXT90 panel and new implementation before effect analysis."""
    old = prev.verify(prev.OUT)
    audit = prev.old.read(prev.OUT / "audit.json")
    if audit["status"] != "PASS" or audit["authors"] != 12:
        raise ValueError("The corrected NEXT90 audit is required")
    state = prev.old.read(prev.OUT / "analysis/freeze.json")
    prev.old.verify_hashes(prev.OUT / "analysis", state["files"])
    paths = [
        PLAN,
        Path(__file__),
        Path(fn.__file__),
        ROOT / "tests/test_grain_cell_residual.py",
        prev.OUT / "manifest.json",
        prev.OUT / "audit.json",
        prev.OUT / "private_jobs.json",
        prev.OUT / "analysis/private_responses.csv",
        prev.OUT / "analysis/freeze.json",
    ]
    inputs = dict(old["inputs"])
    inputs.update({str(p.relative_to(ROOT)): prev.old.sha(p) for p in paths})
    start = time.process_time()
    jobs = prev.old.read(prev.OUT / "private_jobs.json")
    if len(jobs) != 12 or len({j["key"] for j in jobs}) != 12:
        raise ValueError("Exact original4/reused8 panel required")
    if sorted(j["cohort"] for j in jobs) != ["new8"] * 8 + ["source4"] * 4:
        raise ValueError("Unexpected source/evaluation cohort")
    for job in jobs:
        data = [
            prev.old.arrays(prev.old.OLD / "private_inputs" / (key + ".npz"))
            for key in job["query_keys"]
        ]
        for field in ("positions", "grain"):
            np.testing.assert_array_equal(data[0][field], data[1][field])
        np.testing.assert_array_equal(
            prev.old.body_directions(data[0])[:, data[0]["positions"]],
            prev.old.body_directions(data[1])[:, data[1]["positions"]],
        )
    cpu = time.process_time() - start
    out.mkdir(exist_ok=False)
    write(out / "private_jobs.json", jobs)
    write(
        out / "manifest.json",
        {
            "protocol": "NEXT91_GRAIN_CELL_RESIDUAL_V1",
            "inputs": inputs,
            "prepared": {"private_jobs.json": prev.old.sha(out / "private_jobs.json")},
            "prepare_cpu_seconds": cpu,
            "max_cpu_seconds": CAP,
            "authors": 12,
            "native_contexts": 24,
            "fits": 0,
            "native_calls": 0,
            "human_labels_read": False,
            "exact_cells_no_rounding": True,
            "source_plan_sha256": prev.old.sha(PLAN),
        },
    )
    print({"stage": "prepare", "status": "COMPLETE", "cpu_seconds": cpu}, flush=True)


def analyze(out, stage):
    """Run the complete finite panel with a two-author time-only gate."""
    verify(out)
    prior = used(out)
    jobs = prev.old.read(out / "private_jobs.json")
    reference = pd.read_csv(
        prev.OUT / "analysis/private_responses.csv", keep_default_na=False
    )
    reference = reference[reference["map"] == "grain_moments"].set_index(
        ["author_key", "query", "factor"]
    )
    dst = out / stage
    dst.mkdir(exist_ok=False)
    names = [job["key"] + ".json" for job in jobs] + [
        prefix + name + ".csv" for prefix in ("", "private_") for name in fn.TABLES
    ]
    reads = [
        folder / (key + ".npz")
        for job in jobs
        for key in job["query_keys"]
        for folder in (
            prev.old.OLD / "private_inputs",
            prev.old.NATIVE / "private_packets",
        )
    ]
    denials = install(
        reads,
        [dst / name for name in [*names, "freeze.json"]],
        [prev.current.OUT / "teachers/private_teachers.npz"],
        [ROOT / "scripts", ROOT / "grain_research"],
    )
    expected_denials = len(denials)
    start = time.process_time()
    raw, checks = {k: [] for k in fn.TABLES}, {}
    for number, job in enumerate(jobs):
        data = prev.old.arrays(
            prev.old.OLD / "private_inputs" / (job["query_keys"][0] + ".npz")
        )
        gradients = [
            prev.old.selected(
                prev.old.NATIVE / "private_packets" / (key + ".npz"), ("body_gradient",)
            )["body_gradient"]
            for key in job["query_keys"]
        ]
        ds = prev.old.body_directions(data)[:, data["positions"]]
        rows, errors = fn.analyze(data["grain"], ds, np.stack(gradients))
        for row in rows["responses"]:
            if row["part"] == "linear":
                previous = reference.loc[(job["key"], row["query"], row["factor"])]
                error = max(
                    abs(row["native"] - previous.native),
                    abs(row["action"] - previous.visible),
                )
                errors["old_map_action_parity"] = max(
                    errors.get("old_map_action_parity", 0), float(error)
                )
        if max(errors.values()) > 1e-8:
            raise ValueError("Frozen map parity or identities failed")
        for key, value in errors.items():
            checks[key] = max(checks.get(key, 0), value)
        write(
            dst / (job["key"] + ".json"),
            {"status": "COMPLETE", "checks": errors, "rows": rows},
        )
        for name in fn.TABLES:
            raw[name].extend(
                {"cohort": job["cohort"], "author_key": job["key"], **r}
                for r in rows[name]
            )
        elapsed = time.process_time() - start
        if prior + elapsed > CAP:
            raise TimeoutError("60CPU cap reached; completed author units preserved")
        if number == 1:
            projected = elapsed * len(jobs) / 2 * (2 if stage == "analysis" else 1)
            print(
                {
                    "stage": stage,
                    "authors_completed": 2,
                    "cpu_seconds": elapsed,
                    "projected_remaining_stage_and_replay": projected,
                },
                flush=True,
            )
            if prior + projected > CAP:
                raise TimeoutError(
                    "Two-author cost gate; no outcome-dependent budget extension"
                )
    raw = {name: pd.DataFrame(rows) for name, rows in raw.items()}
    tables = fn.aggregate(raw)
    if [len(tables[name]) for name in fn.TABLES] != fn.COUNTS:
        raise ValueError("Incomplete cohort or condition tables")
    for name in fn.TABLES:
        raw[name].to_csv(dst / ("private_" + name + ".csv"), index=False)
        tables[name].to_csv(dst / (name + ".csv"), index=False)
    elapsed = time.process_time() - start
    if prior + elapsed > CAP:
        raise TimeoutError("Full numeric stage exceeded cap")
    write(
        dst / "freeze.json",
        {
            "status": "COMPLETE",
            "cpu_seconds": elapsed,
            "checks": checks,
            "files": {name: prev.old.sha(dst / name) for name in names},
            "counts": fn.COUNTS,
            "extra_denials": len(denials) - expected_denials,
        },
    )
    print({"stage": stage, "status": "COMPLETE", "cpu_seconds": elapsed}, flush=True)


def audit(out):
    """Replay all summaries and independently form a projector on one whole author."""
    verify(out)
    if (out / "audit.json").exists():
        raise FileExistsError("Completed audit is immutable")
    states = [
        prev.old.read(out / stage / "freeze.json") for stage in ("analysis", "replay")
    ]
    if (
        any(
            s["status"] != "COMPLETE" or s["extra_denials"] or s["counts"] != fn.COUNTS
            for s in states
        )
        or states[0]["files"] != states[1]["files"]
    ):
        raise ValueError("Exact complete analysis/replay required")
    for stage, state in zip(("analysis", "replay"), states, strict=True):
        prev.old.verify_hashes(out / stage, state["files"])
        if logical_files(out / stage)[0] != set(state["files"]) | {"freeze.json"}:
            raise ValueError("Unexpected output files")
    start = time.process_time()
    raw = {
        name: pd.read_csv(out / "analysis" / ("private_" + name + ".csv"))
        for name in fn.TABLES
    }
    tables = fn.aggregate(raw)
    for name in fn.TABLES:
        saved = pd.read_csv(out / "analysis" / (name + ".csv"))
        pd.testing.assert_frame_equal(
            saved, tables[name], check_dtype=False, atol=1e-11, rtol=1e-11
        )
    job = prev.old.read(out / "private_jobs.json")[0]
    key = job["query_keys"][0]
    data = prev.old.arrays(prev.old.OLD / "private_inputs" / (key + ".npz"))
    j = prev.old.selected(
        prev.old.NATIVE / "private_packets" / (key + ".npz"), ("body_gradient",)
    )["body_gradient"].astype(float)
    maps = fn.CellMaps(data["grain"])
    norm = np.linalg.norm(maps.a2, axis=0)
    a = maps.a2[:, norm > 0] / norm[norm > 0]
    quadratic = a @ np.linalg.lstsq(a, j, rcond=max(a.shape) * np.finfo(float).eps)[0]
    error = float(
        np.linalg.norm(quadratic - maps.q2 @ (maps.q2.T @ j))
        / max(1, np.linalg.norm(j))
    )
    cell = np.empty_like(j)
    for index in range(len(maps.counts)):
        selected = maps.index == index
        cell[selected] = j[selected].mean(0)
    cell_error = float(np.max(abs(cell - maps.cell_project(j))))
    if max(error, cell_error, *states[0]["checks"].values()) > 1e-8:
        raise ValueError("Independent projection or decomposition check failed")
    total = used(out) + time.process_time() - start
    if total > CAP:
        raise TimeoutError("Canonical numeric budget including audit exceeded")
    result = {
        "status": "PASS",
        "authors": 12,
        "native_contexts": 24,
        "fits": 0,
        "native_calls": 0,
        "human_labels_read": False,
        "counts": fn.COUNTS,
        "numeric_cpu_seconds": total,
        "independent_quadratic_error": error,
        "independent_cell_error": cell_error,
        "exact_replay": True,
        "all_aggregates_rebuilt": True,
        "checks": states[0]["checks"],
    }
    write(out / "audit.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "analyze", "replay", "audit"])
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.stage in ("analyze", "replay"):
            analyze(args.output, "analysis" if args.stage == "analyze" else "replay")
        else:
            globals()[args.stage](args.output)
