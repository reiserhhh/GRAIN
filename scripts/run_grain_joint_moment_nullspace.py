#!/usr/bin/env python3
"""NEXT90 frozen current-moment analysis using existing paired full-L gradients."""

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
from grain_research.vector_mechanisms import joint_moment_nullspace as fn
from grain_research.vector_mechanisms.stage_file_guard import install
from scripts import run_grain_joint_query_functional_reader as current
from scripts import run_grain_pooled_feature_null_response as old

PREVIOUS = old.BASE / "logic_pooled_feature_null_response_v1"
OUT = old.BASE / "logic_joint_moment_nullspace_v1"
PLAN = ROOT / "docs/GRAIN_JOINT_MOMENT_NULLSPACE_PLAN_V1.md"
CAP = 60


def used(out):
    """Count canonical numerical work and replay, without hash/import overhead."""
    value = old.read(out / "manifest.json")["prepare_cpu_seconds"]
    for f in ("analysis", "replay"):
        p = out / f / "freeze.json"
        if p.exists():
            value += old.read(p)["cpu_seconds"]
    return value


def verify(out):
    """Verify the exact original data, target, current input and implementation."""
    m = old.read(out / "manifest.json")
    old.verify_hashes(ROOT, m["inputs"])
    old.verify_hashes(out, m["prepared"])
    return m


def prepare(out):
    """Bind all24 native contexts before computing the new projection account."""
    m = old.read(PREVIOUS / "manifest.json")
    old.verify_hashes(ROOT, m["inputs"])
    if old.read(PREVIOUS / "replay_audit.json")["status"] != "PASS":
        raise ValueError("Audited NEXT59 required")
    current.old.verify(current.OUT)
    if current.old.read(current.OUT / "audit.json")["status"] != "PASS":
        raise ValueError("Audited NEXT89 required")
    jobs = old.read(old.OLD / "private_jobs.json")
    recent = {j["key"]: j for j in old.read(current.OUT / "private_jobs.json")}
    if len(jobs) != 12 or len({j["key"] for j in jobs}) != 12:
        raise ValueError("Original4 and reused8 fixed authors required")
    files = dict(m["inputs"])
    paths = [
        PLAN,
        Path(__file__),
        Path(fn.__file__),
        current.p85.BASELINE,
        Path(current.fn.base.__file__),
        PREVIOUS / "manifest.json",
        PREVIOUS / "replay_audit.json",
        ROOT / "scripts/report_grain_joint_moment_nullspace.py",
        ROOT / "tests/test_grain_joint_moment_nullspace.py",
    ]
    start = time.process_time()
    bindings = []
    for job in jobs:
        if (
            job["cohort"] not in ("source4", "new8")
            or recent[job["key"]]["query_keys"] != job["query_keys"]
        ):
            raise ValueError("Current material/query membership differs")
        data = []
        for qi, key in enumerate(job["query_keys"]):
            ip = old.OLD / "private_inputs" / (key + ".npz")
            fresh = current.ep.STATE / "private_inputs" / (key + ".npz")
            a, b = old.arrays(ip), old.arrays(fresh)
            for field in ("state", "grain", "positions", "cos", "sin", "mask"):
                np.testing.assert_array_equal(a[field], b[field])
            packet = old.NATIVE / "private_packets" / (key + ".json")
            rec, content = old.packet_arrays(packet)
            j = content["body_gradient"].astype(float)
            if rec["key"] != key or j.shape != (
                len(a["positions"]),
                a["state"].shape[1],
            ):
                raise ValueError("Exact H16 gradient/address binding required")
            target = old.selected(old.target_path(job, key), ("score", "tau"))
            tau = np.einsum("th,fth->f", j, old.body_directions(a)[:, a["positions"]])
            np.testing.assert_allclose(tau, target["tau"], atol=1e-5, rtol=3e-4)
            np.testing.assert_allclose(
                rec["contrast"], target["score"], atol=1e-5, rtol=3e-4
            )
            bindings.append(
                {
                    "key": key,
                    "tau_error": float(np.max(abs(tau - target["tau"]))),
                    "score_error": abs(float(rec["contrast"] - target["score"])),
                }
            )
            data.append(a)
            paths.extend(
                (
                    ip,
                    fresh,
                    packet,
                    packet.with_suffix(".npz"),
                    old.target_path(job, key),
                )
            )
        current.fn.base.features(data)
        np.testing.assert_array_equal(
            old.body_directions(data[0])[:, data[0]["positions"]],
            old.body_directions(data[1])[:, data[1]["positions"]],
        )
        paths.append(old.basis_path(old.OLD, old.ref(job)))
    scale = old.arrays(current.p85.BASELINE)["scale"]
    if scale.shape != (6,) or np.any(scale <= 0):
        raise ValueError("Current map must retain all six feature blocks")
    cpu = time.process_time() - start
    files.update({str(p.relative_to(ROOT)): old.sha(p) for p in paths})
    out.mkdir(exist_ok=False)
    old.write(out / "private_jobs.json", jobs)
    old.write(out / "private_bindings.json", bindings)
    old.write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_JOINT_MOMENT_NULLSPACE_V1",
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "inputs": files,
            "prepared": {
                n: old.sha(out / n)
                for n in ("private_jobs.json", "private_bindings.json")
            },
            "prepare_cpu_seconds": cpu,
            "max_cpu_seconds": CAP,
            "authors": 12,
            "native_contexts": 24,
            "fits": 0,
            "native_calls": 0,
            "human_labels_read": False,
            "source_authors": 4,
            "reused_evaluation_authors": 8,
            "target": "cached full-L H16 BODY gradient, not local c or assembled fragment",
        },
    )
    print(
        {
            "status": "PREPARED",
            "contexts": 24,
            "cpu_seconds": cpu,
            "tau_parity_max": max(x["tau_error"] for x in bindings),
        },
        flush=True,
    )


def analyze(out, folder):
    """Compute the frozen finite panel, with cost-only checks and exact replay."""
    verify(out)
    prior = used(out)
    jobs = old.read(out / "private_jobs.json")
    dst = out / folder
    dst.mkdir(exist_ok=False)
    names = [j["key"] + ".json" for j in jobs]
    outputs = [
        *names,
        *[k + ".csv" for k in fn.TABLES],
        *["private_" + k + ".csv" for k in fn.TABLES],
        "freeze.json",
    ]
    reads = []
    for job in jobs:
        reads.append(old.basis_path(old.OLD, old.ref(job)))
        reads.extend(
            p / (key + ".npz")
            for key in job["query_keys"]
            for p in (old.OLD / "private_inputs", old.NATIVE / "private_packets")
        )
    denied = install(
        reads,
        [dst / n for n in outputs],
        [current.OUT / "teachers/private_teachers.npz"],
        [ROOT / "scripts", ROOT / "grain_research"],
    )
    start = time.process_time()
    raw = {k: [] for k in fn.TABLES}
    checks = {}
    for ji, job in enumerate(jobs):
        data = [
            old.arrays(old.OLD / "private_inputs" / (key + ".npz"))
            for key in job["query_keys"]
        ]
        gradients = [
            old.selected(
                old.NATIVE / "private_packets" / (key + ".npz"), ("body_gradient",)
            )["body_gradient"]
            for key in job["query_keys"]
        ]
        pca = old.selected(old.basis_path(old.OLD, old.ref(job)), ("basis",))["basis"]
        directions = old.body_directions(data[0])[:, data[0]["positions"]]
        rows, ch = fn.analyze(data[0]["grain"], directions, np.stack(gradients), pca)
        # Verify null tangents against the actual current feature implementation.
        _, q, _ = fn.basis(data[0]["grain"])
        field = np.asarray(gradients[0], float)
        null = field - fn.project(field, q, pca, "grain_moments")
        ds = []
        for d in data:
            v = np.zeros((1, *d["state"].shape))
            v[0, d["positions"]] = null
            ds.append(v)
        _, tangents = current.fn.base.features(data, ds)
        ch["actual_feature_tangent"] = max(
            float(np.linalg.norm(x)) for x in tangents
        ) / max(1.0, np.linalg.norm(field))
        if max(ch.values()) > 1e-8:
            raise ValueError("Actual input null-space contract failed")
        for k, v in ch.items():
            checks[k] = max(checks.get(k, 0.0), v)
        old.write(dst / names[ji], {"status": "COMPLETE", "checks": ch, "rows": rows})
        for k, rr in rows.items():
            raw[k].extend(
                {"cohort": job["cohort"], "author_key": job["key"]} | r for r in rr
            )
        elapsed = time.process_time() - start
        if prior + elapsed > CAP:
            raise TimeoutError("Finite60CPU budget; complete author units preserved")
        if ji == 1:
            estimate = elapsed * len(jobs) / 2 * (2 if folder == "analysis" else 1)
            print(
                {
                    "completed_authors": 2,
                    "cpu_seconds": elapsed,
                    "remaining_pass_estimate": estimate,
                },
                flush=True,
            )
            if prior + estimate > CAP:
                raise TimeoutError(
                    "Cost-only gate; do not extend the budget after outcomes"
                )
    raw = {k: pd.DataFrame(v) for k, v in raw.items()}
    tables = fn.aggregate(raw)
    if [len(tables[k]) for k in fn.TABLES] != fn.COUNTS:
        raise ValueError("All cohort/query/map/factor conditions required")
    for k in fn.TABLES:
        raw[k].to_csv(dst / ("private_" + k + ".csv"), index=False)
        tables[k].to_csv(dst / (k + ".csv"), index=False)
    cpu = time.process_time() - start
    if cpu + prior > CAP or len(denied) != 1:
        raise ValueError("Budget or stage file contract violated")
    old.write(
        dst / "freeze.json",
        {
            "status": "COMPLETE",
            "cpu_seconds": cpu,
            "checks": checks,
            "counts": fn.COUNTS,
            "fits": 0,
            "native_calls": 0,
            "extra_denials": len(denied) - 1,
            "files": {n: old.sha(dst / n) for n in outputs if n != "freeze.json"},
        },
    )
    print({"status": "COMPLETE", "cpu_seconds": cpu, "checks": checks}, flush=True)


def audit(out):
    """Verify exact replay and an independent first-author least-squares projector."""
    verify(out)
    left, right = [old.read(out / f / "freeze.json") for f in ("analysis", "replay")]
    if (
        left["status"] != "COMPLETE"
        or left["files"] != right["files"]
        or left["counts"] != fn.COUNTS
    ):
        raise ValueError("Complete exact replay required")
    for f, m in (("analysis", left), ("replay", right)):
        old.verify_hashes(out / f, m["files"])
        if m["extra_denials"] or {p.name for p in (out / f).iterdir()} != set(
            m["files"]
        ) | {"freeze.json"}:
            raise ValueError("Unexpected stage files or access")
    start = time.process_time()
    job = old.read(out / "private_jobs.json")[0]
    key = job["query_keys"][0]
    data = old.arrays(old.OLD / "private_inputs" / (key + ".npz"))
    g = old.selected(
        old.NATIVE / "private_packets" / (key + ".npz"), ("body_gradient",)
    )["body_gradient"].astype(float)
    a, q, _ = fn.basis(data["grain"])
    expected = a @ np.linalg.lstsq(a, g, rcond=None)[0]
    error = float(
        np.linalg.norm(expected - q @ (q.T @ g)) / max(1.0, np.linalg.norm(g))
    )
    if error > 1e-9:
        raise ValueError("Independent projection differs")
    total = used(out) + time.process_time() - start
    if total > CAP:
        raise TimeoutError("Canonical including replay exceeded60CPU")
    report = {
        "status": "PASS",
        "contexts": 24,
        "authors": 12,
        "source_authors": 4,
        "evaluation_authors": 8,
        "numeric_cpu_seconds": total,
        "independent_projector_relative_error": error,
        "public_tables": 6,
        "public_counts": fn.COUNTS,
        "private_units": 12,
        "fits": 0,
        "native_calls": 0,
        "human_labels_read": False,
        "checks": left["checks"],
    }
    old.write(out / "audit.json", report)
    print(report, flush=True)


def main():
    """Separate immutable preflight, numeric analysis/replay and audit processes."""
    p = argparse.ArgumentParser()
    p.add_argument(
        "--stage", required=True, choices=("prepare", "analyze", "replay", "audit")
    )
    p.add_argument("--output", type=Path, default=OUT)
    args = p.parse_args()
    with threadpool_limits(limits=1):
        if args.stage in ("analyze", "replay"):
            analyze(args.output, "analysis" if args.stage == "analyze" else "replay")
        else:
            {"prepare": prepare, "audit": audit}[args.stage](args.output)


if __name__ == "__main__":
    main()
