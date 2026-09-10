#!/usr/bin/env python3
"""NEXT61: isolated source dictionary capacity and value-preserving field analysis."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.vector_mechanisms.stage_file_guard import install
from grain_research.vector_mechanisms.token_objective_compatibility import (
    RCOND,
    diagnose,
    reconstruct_design,
)
from grain_research.vector_mechanisms.token_potential import ARMS, validate_members
from scripts.run_grain_token_potential_extraction import (
    BASE,
    QUERIES,
    arrays,
    check_paths,
    forbidden,
    frozen,
    gradient_path,
    input_path,
    load_manifest,
    model_path,
    pure_input,
    read,
    ref,
    selected,
    sha,
    source_jobs,
    target_path,
    verify_hashes,
    write,
)
from scripts.run_grain_token_potential_extraction import (
    OUT as OLD,
)

OUT = BASE / "logic_token_objective_compatibility_v1"
PLAN = ROOT / "docs/GRAIN_TOKEN_OBJECTIVE_COMPATIBILITY_PLAN_V1.md"


def source_paths(src):
    """Enumerate only the source-reference H16, native gradient and scalar packets."""
    return [
        p
        for j in src
        for k in j["query_keys"]
        for p in (input_path(k), gradient_path(k), target_path(j, k))
    ]


def verify(out):
    """Verify protocol/code/metadata without opening any application data."""
    m = read(out / "manifest.json")
    verify_hashes(ROOT, m["code"])
    verify_hashes(out, {"private_jobs.json": m["jobs_sha256"]})
    return m


def prepare(out):
    """Bind the completed model family and source-only caches before new diagnostics."""
    started = time.process_time()
    old = load_manifest(OLD)
    fm = frozen(OLD, "models")
    if read(OLD / "replay_audit.json")["status"] != "PASS":
        raise ValueError("Prior functional audit required")
    jobs = read(OLD / "private_jobs.json")
    src = [j for j in jobs if j["cohort"] == "source4"]
    if len(src) != 4 or len(jobs) != 12:
        raise ValueError("Frozen source/application metadata differs")
    check_paths(old, source_paths(src))
    out.mkdir(exist_ok=False)
    write(out / "private_jobs.json", jobs)
    code = [
        PLAN,
        Path(__file__),
        ROOT / "grain_research/vector_mechanisms/token_objective_compatibility.py",
        ROOT / "tests/test_grain_token_objective_compatibility.py",
        OLD / "manifest.json",
        OLD / "models/freeze.json",
    ]
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_TOKEN_OBJECTIVE_COMPATIBILITY_V1",
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "code": {str(p.relative_to(ROOT)): sha(p) for p in code},
            "jobs_sha256": sha(out / "private_jobs.json"),
            "old_code": old["code"],
            "source_hashes": {
                str(p.relative_to(ROOT)): sha(p) for p in source_paths(src)
            },
            "model_hashes": fm["files"],
            "max_cpu_seconds": 60,
            "svd_rcond": RCOND,
            "prepare_cpu_seconds": time.process_time() - started,
            "new_neural_calls": 0,
            "human_labels_read": False,
            "application_target_reads": 0,
        },
    )
    print({"status": "PREPARED", "source_authors": 4, "new_models": 0})


def analyze_reference(out, reference, folder):
    """Analyze one source reference under an explicit target/application denylist."""
    start = time.process_time()
    m = verify(out)
    verify_hashes(ROOT, m["old_code"])
    jobs = read(out / "private_jobs.json")
    src = source_jobs(jobs, reference)
    reads = source_paths(src)
    verify_hashes(
        ROOT,
        {
            str(p.relative_to(ROOT)): m["source_hashes"][str(p.relative_to(ROOT))]
            for p in reads
        },
    )
    models = [model_path(OLD, reference, q, arm) for q in QUERIES for arm in ARMS]
    verify_hashes(OLD / "models", {p.name: m["model_hashes"][p.name] for p in models})
    dst = out / folder
    packets = [dst / f"{reference}_{q}_{arm}.npz" for q in QUERIES for arm in ARMS]
    status = dst / (reference + "_status.json")
    excluded = [j for j in jobs if j not in src]
    probes = (
        forbidden()
        + source_paths(excluded)
        + [OLD / "analysis/private_scores.csv", OLD / "analysis/private_actions.csv"]
    )
    denied = install(
        reads + models,
        packets + [status],
        probes,
        [ROOT / "scripts", ROOT / "grain_research"],
    )
    rows = []
    for qi, q in enumerate(QUERIES):
        bodies, lasts, gradients, scores = [], [], [], []
        for j in src:
            key = j["query_keys"][qi]
            _, h, last = pure_input(key)
            bodies.append(h)
            lasts.append(last)
            gradients.append(
                selected(gradient_path(key), ("body_gradient",))[
                    "body_gradient"
                ].astype(float)
            )
            scores.append(float(selected(target_path(j, key), ("score",))["score"]))
        for arm in ARMS:
            model = arrays(model_path(OLD, reference, q, arm))
            validate_members(model, [j["index"] for j in src])
            v, d, ell, k, anchor, errors = reconstruct_design(
                bodies, lasts, gradients, scores, model
            )
            result, packet = diagnose(
                v, d, ell, k, model["coef"], float(model["alpha"]), anchor
            )
            if (
                result["stationarity_error"] > 1e-8
                or result["preserved_value_change"] > 1e-5
                or result["objective_identity_relative_error"] > 1e-7
            ):
                raise ValueError("Source identity check exceeds fixed tolerance")
            np.savez(dst / f"{reference}_{q}_{arm}.npz", **packet)
            rows.append(
                {
                    "reference": reference,
                    "query_name": q,
                    "candidate": arm,
                    **result,
                    **{"rebuild_" + n: e for n, e in errors.items()},
                }
            )
    write(
        status,
        {
            "status": "COMPLETE",
            "cpu_seconds": time.process_time() - start,
            "rows": rows,
            "diagnostic_ls_solves": 16,
            "svd_decompositions": 16,
            "new_predictive_fits": 0,
            "denial_probes": len(probes),
            "extra_denials": len(denied) - len(probes),
            "application_target_reads": 0,
            "files": {p.name: sha(p) for p in packets},
        },
    )


def analyze(out, folder):
    """Run the frozen diagnostic once, or its sole replay, without application scores."""
    start, wall = time.process_time(), time.monotonic()
    m = verify(out)
    prior = (
        read(out / "primary/summary.json")["cpu_seconds"] if folder == "replay" else 0
    )
    dst = out / folder
    dst.mkdir(exist_ok=False)
    jobs = read(out / "private_jobs.json")
    statuses = []
    for reference in sorted({ref(j) for j in jobs}):
        remaining = 60 - prior - m["prepare_cpu_seconds"] - (time.monotonic() - wall)
        if remaining <= 0:
            raise TimeoutError("Diagnostic budget exhausted")
        subprocess.run(
            [
                sys.executable,
                __file__,
                "--stage",
                "reference",
                "--reference",
                reference,
                "--folder",
                folder,
                "--output",
                str(out),
            ],
            cwd=ROOT,
            check=True,
            timeout=remaining,
        )
        statuses.append(read(dst / (reference + "_status.json")))
        projected = (
            m["prepare_cpu_seconds"]
            + 10 * np.mean([s["cpu_seconds"] for s in statuses])
            + 5
        )
        if projected > 60:
            write(
                out / "bounded_stop.json",
                {
                    "reason": "Projected diagnostic/replay cost",
                    "projected_cpu": projected,
                    "completed_references": len(statuses),
                },
            )
            raise TimeoutError("Source output preserved, no reduced post-result run")
    if any(s["extra_denials"] for s in statuses):
        raise ValueError("Unexpected data access")
    frame = pd.DataFrame([r for s in statuses for r in s["rows"]])
    if len(frame) != 20:
        raise ValueError("All20 models required")
    frame.to_csv(dst / "source_objective_account.csv", index=False)
    fields = [
        "source_field_loss",
        "dictionary_external_loss",
        "unused_dictionary_gain",
        "preserved_score_field_gain",
        "field_norm_ratio",
        "preserving_delta_norm_ratio",
        "preserving_penalty_change",
        "anchor_condition",
        "field_anchor_coefficient_norm",
        "stationarity_error",
    ]
    summary = frame.assign(
        reference_group=np.where(
            frame.reference.eq("all4"), "all4", "loo_reference_descriptive"
        )
    )
    group = summary.groupby(["reference_group", "query_name", "candidate"], sort=True)[
        fields
    ].agg(["mean", "min", "max"])
    group.columns = ["_".join(c) for c in group.columns]
    group.reset_index().to_csv(dst / "group_summary.csv", index=False)
    cpu = time.process_time() - start + sum(s["cpu_seconds"] for s in statuses)
    write(
        dst / "summary.json",
        {
            "status": "COMPLETE",
            "cpu_seconds": cpu,
            "source_systems": 20,
            "diagnostic_ls_solves": sum(s["diagnostic_ls_solves"] for s in statuses),
            "svd_decompositions": sum(s["svd_decompositions"] for s in statuses),
            "new_predictive_fits": 0,
            "new_neural_calls": 0,
            "application_target_reads": 0,
            "files": {
                p.name: sha(p)
                for p in dst.iterdir()
                if not p.name.startswith(".") and p.suffix in (".csv", ".npz", ".json")
            },
        },
    )
    print(
        {
            "status": "COMPLETE",
            "folder": folder,
            "source_systems": 20,
            "cpu_seconds": cpu,
            "diagnostic_ls_solves": 80,
        }
    )


def audit(out):
    """Verify source-only replay, projection equations and additive loss identities."""
    start = time.process_time()
    m = verify(out)
    primary, replay = [
        read(out / name / "summary.json") for name in ("primary", "replay")
    ]
    for name, record in (("primary", primary), ("replay", replay)):
        verify_hashes(out / name, record["files"])
    for name in ("source_objective_account.csv", "group_summary.csv"):
        if sha(out / "primary" / name) != sha(out / "replay" / name):
            raise ValueError("Numerical table replay differs")
    for name in primary["files"]:
        if name.endswith(".npz"):
            a, b = [arrays(out / f / name) for f in ("primary", "replay")]
            for key in a:
                np.testing.assert_array_equal(a[key], b[key])
    f = pd.read_csv(out / "primary/source_objective_account.csv")
    closure = (
        (f.target_energy - f.represented_target_energy - f.dictionary_external_loss)
        .abs()
        .max()
    )
    if (
        closure > 1e-8
        or f.stationarity_error.max() > 1e-8
        or f.parameter_projection_relative_error.max() > 1e-5
        or f.feasible_projection_relative_error.max() > 1e-5
    ):
        raise ValueError("Source projection/force identity failed")
    if f.preserving_objective_change.min() < -1e-7:
        raise ValueError(
            "A source quadratic optimum was improved without changing the objective"
        )
    statuses = [
        read(out / folder / (r + "_status.json"))
        for folder in ("primary", "replay")
        for r in f.reference.unique()
    ]
    if any(s["extra_denials"] or s["application_target_reads"] for s in statuses):
        raise ValueError("Source-only diagnostic used excluded information")
    cpu = time.process_time() - start
    total = (
        m["prepare_cpu_seconds"] + primary["cpu_seconds"] + replay["cpu_seconds"] + cpu
    )
    if total > 60:
        raise TimeoutError("Cumulative frozen CPU cap exceeded")
    result = {
        "status": "PASS",
        "source_systems": 20,
        "logical_csvs_replayed": 2,
        "private_arrays_replayed": 20,
        "primary_diagnostic_ls_solves": 80,
        "replay_diagnostic_ls_solves": 80,
        "new_predictive_fits": 0,
        "neural_calls": 0,
        "human_labels_read": False,
        "application_targets_read": 0,
        "cpu_seconds": cpu,
        "cumulative_compute_cpu_seconds": total,
        "errors": {
            "orthogonal_energy": float(closure),
            "stationarity": float(f.stationarity_error.max()),
            "preserved_values": float(f.preserved_value_change.max()),
            "quadratic_identity_relative": float(
                f.objective_identity_relative_error.max()
            ),
            "projection_relative": float(
                max(
                    f.parameter_projection_relative_error.max(),
                    f.feasible_projection_relative_error.max(),
                )
            ),
            "source_gram_rebuild": float(f.rebuild_gram.max()),
        },
    }
    write(out / "replay_audit.json", result)
    print(result)


def main():
    """Keep preparation, each source reference and replay audit in fresh processes."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--stage", choices=("prepare", "analyze", "reference", "audit"), required=True
    )
    p.add_argument("--output", type=Path, default=OUT)
    p.add_argument("--reference", default="all4")
    p.add_argument("--folder", choices=("primary", "replay"), default="primary")
    a = p.parse_args()
    with threadpool_limits(limits=1):
        if a.stage == "reference":
            analyze_reference(a.output, a.reference, a.folder)
        elif a.stage == "analyze":
            analyze(a.output, a.folder)
        else:
            globals()[a.stage](a.output)


if __name__ == "__main__":
    main()
