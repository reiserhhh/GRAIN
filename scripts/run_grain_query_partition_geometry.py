#!/usr/bin/env python3
"""NEXT72: compile frozen source-query joint moments, then account for responses."""

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
from grain_research.vector_mechanisms.query_partition_geometry import (
    FIELDS,
    compile_geometry,
    summary,
)
from grain_research.vector_mechanisms.stage_file_guard import install
from scripts import run_grain_source_query_selection as old

SOURCE = old.BASE / "logic_source_query_selection_v1"
OUT = old.BASE / "logic_query_partition_geometry_v1/r2"
PLAN = ROOT / "docs/GRAIN_QUERY_PARTITION_GEOMETRY_PLAN_V1.md"
CAP = 90
TABLES = ("query_geometry.csv", "factor_geometry.csv", "contrast_geometry.csv")


def table_hashes(folder):
    """Hash only declared tables, never filesystem-generated dot-file sidecars."""
    return {name: old.sha(folder / name) for name in TABLES}


def prepare(out):
    """Freeze exact earlier pure input addresses; never copy native targets."""
    start = time.process_time()
    m = old.verify(SOURCE)
    if old.read(SOURCE / "replay_audit.json")["status"] != "PASS":
        raise ValueError("Audited NEXT50 required")
    jobs = old.read(SOURCE / "private_jobs.json")
    if len(jobs) != 12 or len({j["key"] for j in jobs}) != 12:
        raise ValueError("Exact twelve paired authors required")
    out.mkdir(exist_ok=False)
    old.write(out / "private_jobs.json", jobs)
    inputs = {
        str((SOURCE / p).relative_to(ROOT)): digest
        for p, digest in m["prepared"].items()
    }
    paths = [
        PLAN,
        Path(__file__),
        SOURCE / "manifest.json",
        SOURCE / "replay_audit.json",
        ROOT / "tests/test_grain_query_partition_geometry.py",
    ]
    paths += [
        ROOT / "grain_research/vector_mechanisms" / (n + ".py")
        for n in (
            "query_partition_geometry",
            "source_query_selection",
            "fixed_seed_attention",
            "stage_file_guard",
            "compiled_query_transfer",
        )
    ]
    paths += [
        ROOT / "scripts" / (n + ".py")
        for n in (
            "run_grain_source_query_selection",
            "run_grain_compiled_query_transfer",
        )
    ]
    inputs.update({str(p.relative_to(ROOT)): old.sha(p) for p in paths})
    cpu = time.process_time() - start
    if cpu > CAP:
        raise TimeoutError("Preparation exceeded cap")
    old.write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_QUERY_PARTITION_GEOMETRY_V1",
            "inputs": inputs,
            "prepared": {"private_jobs.json": old.sha(out / "private_jobs.json")},
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "prepare_cpu_seconds": cpu,
            "fits": 0,
            "new_neural_calls": 0,
            "human_labels_read": False,
            "current_Q_read": False,
        },
    )
    print({"status": "PREPARED", "cpu_seconds": cpu, "authors": 12}, flush=True)


def compile_stage(out, name):
    """Write all moments in a process denied access to target and old predictions."""
    old.verify(out)
    jobs = old.read(out / "private_jobs.json")
    dst = out / name
    dst.mkdir(exist_ok=False)
    reads = [SOURCE / "private_source_bank.npz"] + [
        SOURCE / "private_inputs" / (k + ".npz") for j in jobs for k in j["query_keys"]
    ]
    paths = [dst / (j["key"] + ".npz") for j in jobs]
    probes = [
        SOURCE / "predictions" / (jobs[0]["key"] + ".npz"),
        old.SOURCE / "predictions" / (jobs[0]["key"] + ".npz"),
    ]
    denied = install(
        reads,
        paths + [dst / "freeze.json"],
        probes,
        [ROOT / "grain_research", ROOT / "scripts"],
    )
    start = time.process_time()
    bank = old.arrays(reads[0])
    checks = []
    for j, path in zip(jobs, paths, strict=True):
        values = []
        for q, key in zip(old.QUERIES, j["query_keys"], strict=True):
            anchors, members = old.anchors(bank, j["cohort"], j["index"], q)
            values.append(
                compile_geometry(
                    anchors, old.arrays(SOURCE / "private_inputs" / (key + ".npz"))
                )
            )
            checks.append({"key": j["key"], "query": q, "sources": members})
        np.savez_compressed(
            path, **{k: np.stack([v[k] for v in values]) for k in FIELDS}
        )
        if time.process_time() - start > CAP:
            raise TimeoutError("Compilation cap exceeded")
    result = {
        "status": "COMPLETE",
        "cpu_seconds": time.process_time() - start,
        "checks": checks,
        "files": {p.name: old.sha(p) for p in paths},
        "target_reads": 0,
        "current_Q_reads": 0,
        "fits": 0,
        "neural_calls": 0,
        "denial_probes": len(probes),
        "extra_denials": len(denied) - len(probes),
    }
    old.write(dst / "freeze.json", result)
    print({k: v for k, v in result.items() if k not in ("checks", "files")}, flush=True)


def evaluate(out, name):
    """Join only frozen outputs to earlier predicted and native key-channel targets."""
    old.verify(out)
    fr = old.read(out / name / "freeze.json")
    if fr["status"] != "COMPLETE":
        raise ValueError("Frozen geometry required before native joins")
    old.verify_hashes(out / name, fr["files"])
    prev = old.read(SOURCE / "predictions/freeze.json")
    old.verify_hashes(SOURCE / "predictions", prev["files"])
    nf = old.read(old.SOURCE / "predictions/prediction_freeze.json")
    old.verify_hashes(old.SOURCE / "predictions", nf["files"])
    start = time.process_time()
    dst = out / ("analysis" if name == "moments" else "replay_analysis")
    dst.mkdir(exist_ok=False)
    jobs = old.read(out / "private_jobs.json")
    grouped = {c: [] for c in ("source4", "new8")}
    maximum = 0.0
    for j in jobs:
        geom = old.arrays(out / name / (j["key"] + ".npz"))
        baseline = old.arrays(SOURCE / "predictions" / (j["key"] + ".npz"))["actions"]
        compiled = geom["response"].sum(-1)
        maximum = max(maximum, float(np.max(np.abs(compiled - baseline))))
        np.testing.assert_allclose(compiled, baseline, atol=1e-11, rtol=1e-10)
        target = old.arrays(old.SOURCE / "predictions" / (j["key"] + ".npz"))[
            "actions"
        ][:, :, 1, :]
        grouped[j["cohort"]].append((geom, target))
    queries, factors, contrasts = [], [], []
    for cohort, pairs in grouped.items():
        g = {k: np.stack([p[0][k] for p in pairs]) for k in FIELDS}
        target = np.stack([p[1] for p in pairs])
        for a, arm in enumerate(old.ARMS):
            for m, mode in enumerate(old.MODES):
                base = {
                    "cohort": cohort,
                    "arm": arm,
                    "mode": mode,
                    "authors": len(pairs),
                }
                for q, query in enumerate(old.QUERIES):
                    selected = {k: v[:, q, a, m] for k, v in g.items()}
                    y, p = target[:, q, m], selected["response"].sum(-1)
                    queries.append(
                        base
                        | {"query": query}
                        | summary(selected)
                        | old.fidelity(y.ravel(), p.ravel())
                    )
                    for f in range(4):
                        factors.append(
                            base
                            | {"query": query, "factor": f"whole:F{f + 1:02d}"}
                            | summary({k: v[:, f] for k, v in selected.items()})
                            | old.fidelity(y[:, f], p[:, f])
                        )
                r = g["response"][:, 0, a, m] - g["response"][:, 1, a, m]
                u = g["uncentered"][:, 0, a, m] - g["uncentered"][:, 1, a, m]
                b = g["centering"][:, 0, a, m] - g["centering"][:, 1, a, m]
                actual = (target[:, 0, m] - target[:, 1, m]).ravel()
                record = base | old.fidelity(actual, r.sum(-1).ravel())
                for label, x in (("uncentered", u), ("centering", b), ("response", r)):
                    z = x.sum(-1).ravel()
                    record[label + "_rms"] = float(np.sqrt(np.mean(z * z)))
                    record[label + "_target_projection"] = float(
                        z @ actual / (actual @ actual)
                    )
                record["head_sign_retention"] = float(
                    np.abs(r.sum(-1)).sum() / np.abs(r).sum()
                )
                contrasts.append(record)
    for file, rows in (
        ("query_geometry.csv", queries),
        ("factor_geometry.csv", factors),
        ("contrast_geometry.csv", contrasts),
    ):
        pd.DataFrame(rows).to_csv(dst / file, index=False)
    cpu = time.process_time() - start
    old.write(
        dst / "summary.json",
        {
            "status": "COMPLETE",
            "cpu_seconds": cpu,
            "old_actions_max_error": maximum,
            "counts": [len(queries), len(factors), len(contrasts)],
            "files": table_hashes(dst),
        },
    )
    print(
        {"status": "EVALUATED", "cpu_seconds": cpu, "old_actions_max_error": maximum},
        flush=True,
    )


def audit(out):
    """Verify exact packet/table replay and the bounded common-potential account."""
    start = time.process_time()
    m = old.verify(out)
    fs = [old.read(out / n / "freeze.json") for n in ("moments", "replay")]
    for name in fs[0]["files"]:
        a, b = [old.arrays(out / n / name) for n in ("moments", "replay")]
        for k in FIELDS:
            np.testing.assert_array_equal(a[k], b[k])
    ss = [old.read(out / n / "summary.json") for n in ("analysis", "replay_analysis")]
    for n, s in zip(("analysis", "replay_analysis"), ss, strict=True):
        old.verify_hashes(out / n, s["files"])
    if (
        ss[0]["files"] != ss[1]["files"]
        or set(ss[0]["files"]) != set(TABLES)
        or ss[0]["counts"] != [24, 96, 12]
    ):
        raise ValueError("Aggregate replay or cardinality differs")
    cpu = (
        time.process_time()
        - start
        + m["prepare_cpu_seconds"]
        + sum(s["cpu_seconds"] for s in fs + ss)
    )
    if cpu > CAP or len(fs[0]["files"]) != 12:
        raise ValueError("Budget/cardinality failed")
    result = {
        "status": "PASS",
        "packets": 12,
        "csv_replay": 3,
        "old_actions_max_error": ss[0]["old_actions_max_error"],
        "total_cpu_seconds": cpu,
        "fits": 0,
        "new_neural_calls": 0,
        "human_labels_read": False,
    }
    old.write(out / "audit.json", result)
    print(result, flush=True)


def main():
    """Run isolated preparation, compilation, evaluation and audit stages."""
    p = argparse.ArgumentParser()
    p.add_argument(
        "--stage", required=True, choices=("prepare", "compile", "evaluate", "audit")
    )
    p.add_argument("--folder", default="moments", choices=("moments", "replay"))
    p.add_argument("--output", type=Path, default=OUT)
    a = p.parse_args()
    with threadpool_limits(limits=1):
        if a.stage == "compile":
            compile_stage(a.output, a.folder)
        elif a.stage == "evaluate":
            evaluate(a.output, a.folder)
        else:
            globals()[a.stage](a.output)


if __name__ == "__main__":
    main()
