#!/usr/bin/env python3
"""NEXT71: no-fit component account of the frozen NEXT70 readers."""

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
from grain_research.vector_mechanisms import evidence_components as program
from grain_research.vector_mechanisms.stage_file_guard import install
from grain_research.vector_mechanisms.token_potential import hidden_gradient
from scripts import run_grain_normalized_evidence as previous

old, cache, ARMS = previous.old, previous.prior, previous.ARMS
OUT = old.BASE / "logic_evidence_components_v1"
CAP = 60


def prepare(out):
    """Freeze the account before joining any native response targets."""
    start = time.process_time()
    m = cache.manifest(previous.OUT)
    if old.read(previous.OUT / "audit.json")["status"] != "PASS":
        raise ValueError("Audited NEXT70 required")
    for name in ("models", "predictions"):
        old.frozen(previous.OUT, name)
    jobs = old.read(previous.OUT / "private_jobs.json")
    if len(jobs) != 12 or len(cache.items(jobs)) != 48:
        raise ValueError("Frozen source4/new8 required")
    code = [
        Path(__file__),
        ROOT / "grain_research/vector_mechanisms/evidence_components.py",
        ROOT / "tests/test_grain_evidence_components.py",
        ROOT / "docs/GRAIN_EVIDENCE_COMPONENT_PLAN_V1.md",
    ]
    lineage = [
        previous.OUT / n
        for n in (
            "models/freeze.json",
            "predictions/freeze.json",
            "analysis/summary.json",
            "audit.json",
        )
    ]
    out.mkdir(exist_ok=False)
    old.write(out / "private_jobs.json", jobs)
    old.write(
        out / "manifest.json",
        {
            **m,
            "protocol": "GRAIN_EVIDENCE_COMPONENTS_V1",
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "code": m["code"] | {str(p.relative_to(ROOT)): old.sha(p) for p in code},
            "baseline_freezes": m["baseline_freezes"]
            | {str(p.relative_to(ROOT)): old.sha(p) for p in lineage},
            "jobs_sha256": old.sha(out / "private_jobs.json"),
            "max_cpu_seconds": CAP,
            "fits": 0,
            "prepare_cpu_seconds": time.process_time() - start,
        },
    )
    print({"status": "PREPARED", "fits": 0}, flush=True)


def apply(out, folder):
    """Reconstruct all components using inputs/models only, before target access."""
    start = time.process_time()
    m = cache.manifest(out)
    old.frozen(previous.OUT, "models")
    old.frozen(previous.OUT, "predictions")
    jobs = old.read(out / "private_jobs.json")
    entries = cache.items(jobs)
    inputs = [cache.input_path(j, v, k) for j, v, q, k in entries]
    old.check_paths(m, inputs)
    models = [
        previous.model_path(previous.OUT, q, a) for q in old.QUERIES for a in ARMS
    ]
    controls = [cache.model_path(cache.OUT, q, cache.ARMS[0]) for q in old.QUERIES]
    old.check_paths(m, controls)
    predictions = [
        previous.OUT / "predictions" / (k + "_" + v + ".npz") for j, v, q, k in entries
    ]
    dst = out / folder
    dst.mkdir(exist_ok=False)
    outputs = [dst / (k + "_" + v + ".npz") for j, v, q, k in entries]
    probes = old.forbidden() + list(
        dict.fromkeys(p for j, v, q, k in entries for p in cache.teacher_paths(j, v, k))
    )
    probes += [
        previous.OUT / "analysis" / ("private_" + n + ".csv") for n in previous.NAMES
    ]
    earlier = old.frozen(out, "components")["cpu_seconds"] if folder == "replay" else 0
    denied = install(
        inputs + models + controls + predictions,
        outputs + [dst / "freeze.json"],
        probes,
        [ROOT / "scripts", ROOT / "grain_research"],
    )
    bank = {
        (q, a): old.arrays(previous.model_path(previous.OUT, q, a))
        for q in old.QUERIES
        for a in ARMS
    }
    local = {
        q: old.arrays(cache.model_path(cache.OUT, q, cache.ARMS[0]))
        for q in old.QUERIES
    }
    errors = {"density": 0.0, "tau": 0.0, "allocation": 0.0}
    for j, v, q, k in entries:
        d = cache.pure_input(j, v, k)
        pos = d["positions"]
        body, last, ds = (
            d["state"][pos].astype(float),
            d["state"][-1].astype(float),
            old.body_directions(d)[:, pos],
        )
        frozen = old.arrays(previous.OUT / "predictions" / (k + "_" + v + ".npz"))
        densities, taus, allocations = [], [], []
        for ai, a in enumerate(ARMS):
            parts = program.densities(body, last, bank[q, a])
            tau, allocation = program.actions(parts, ds, bank[q, a])
            for n, actual in (
                ("density", parts.sum(0)),
                ("tau", tau.sum(0)),
                ("allocation", allocation.sum(0)),
            ):
                errors[n] = max(errors[n], float(np.max(abs(actual - frozen[n][ai]))))
            densities.append(parts)
            taus.append(tau)
            allocations.append(allocation)
        baseline = program.reader.base.predict(body, last, local[q])[1]
        bt, ba = program.actions(baseline[None], ds, local[q])
        np.savez(
            dst / (k + "_" + v + ".npz"),
            density=np.stack(densities),
            tau=np.stack(taus),
            allocation=np.stack(allocations),
            baseline_density=baseline,
            baseline_tau=bt[0],
            baseline_allocation=ba[0],
        )
    cpu = time.process_time() - start
    if (
        max(errors.values()) > 1e-10
        or len(denied) != len(probes)
        or cpu + earlier + m["prepare_cpu_seconds"] > CAP
    ):
        raise ValueError(
            "Component reconstruction/access/budget failed: " + str(errors)
        )
    old.write(
        dst / "freeze.json",
        {
            "status": "COMPLETE",
            "cpu_seconds": cpu,
            "fits": 0,
            "current_target_reads": 0,
            "reconstruction_errors": errors,
            "denial_probes": len(probes),
            "extra_denials": len(denied) - len(probes),
            "files": {p.name: old.sha(p) for p in outputs},
        },
    )
    print(
        {
            "status": "COMPONENTS",
            "folder": folder,
            "packets": 48,
            "errors": errors,
            "cpu_seconds": cpu,
        },
        flush=True,
    )


def evaluate(out, folder):
    """Join native gradients only after complete component reconstruction is frozen."""
    start = time.process_time()
    m = cache.manifest(out)
    freeze = old.frozen(out, folder)
    jobs = old.read(out / "private_jobs.json")
    entries = cache.items(jobs)
    old.check_paths(
        m, [p for j, v, q, k in entries for p in cache.teacher_paths(j, v, k)]
    )
    summary = old.read(previous.OUT / "analysis/summary.json")
    old.verify_hashes(previous.OUT / "analysis", summary["files"])
    frozen_actions = pd.read_csv(previous.OUT / "analysis/private_actions.csv")
    bank = {
        (q, a): old.arrays(previous.model_path(previous.OUT, q, a))
        for q in old.QUERIES
        for a in ARMS
    }
    source, source_actions, new = {}, {}, {}
    checks = {
        "local_B": 0.0,
        "raw_content_selection_B": 0.0,
        "baseline_actions": 0.0,
        "frozen_actions": 0.0,
    }
    for j, v, q, k in entries:
        if v != "plus" and not (j["cohort"] == "new8" and v == "original"):
            continue
        d = cache.pure_input(j, v, k)
        ds = old.body_directions(d)[:, d["positions"]]
        g = old.selected(cache.teacher_paths(j, v, k)[0], ("body_gradient",))[
            "body_gradient"
        ].astype(float)
        packet = old.arrays(out / folder / (k + "_" + v + ".npz"))
        if v == "plus":
            minus = old.arrays(out / folder / (k + "_minus.npz"))
            gm = old.selected(
                cache.teacher_paths(j, "minus", k)[0], ("body_gradient",)
            )["body_gradient"].astype(float)
            cut = len(g) // 2
            delta = (g[cut:] - gm[cut:]) / 2
        for ai, a in enumerate(ARMS):
            model = bank[q, a]
            if v == "plus":
                full_parts = (packet["density"][ai] - minus["density"][ai]) / 2
                parts = full_parts[:, cut:]
                checks["local_B"] = max(checks["local_B"], float(np.max(abs(parts[0]))))
                if a == ARMS[0]:
                    checks["raw_content_selection_B"] = max(
                        checks["raw_content_selection_B"],
                        float(np.max(abs(parts[1:3]))),
                    )
                hidden = np.stack([hidden_gradient(x, model)[cut:] for x in full_parts])
                actual = len(g) * (delta @ model["basis"]) * model["scale"]
                for space, target, values in (
                    ("coordinate", actual, parts),
                    ("hidden", delta, hidden),
                ):
                    source.setdefault((q, a, space), []).append(
                        program.moments(target, values)
                    )
                target_action = np.einsum("th,fth->f", delta, ds[:, cut:])
                part_action = np.einsum("cth,fth->cf", hidden, ds[:, cut:])
                source_actions.setdefault((q, a), []).append(
                    (target_action, part_action)
                )
            else:
                for scope, key, bkey, direction in (
                    ("full", "tau", "baseline_tau", ds),
                    (
                        "allocation",
                        "allocation",
                        "baseline_allocation",
                        ds - ds.mean(1, keepdims=True),
                    ),
                ):
                    actual = np.einsum("th,fth->f", g, direction)
                    baseline, parts = packet[bkey], packet[key][ai]
                    oldrows = frozen_actions[
                        (frozen_actions.author_key == j["key"])
                        & (frozen_actions.query_name == q)
                        & (frozen_actions.view == v)
                    ].sort_values("factor")
                    basecol = "predicted" if scope == "full" else "predicted_allocation"
                    for candidate, prediction, check in (
                        ("matched_local", baseline, "baseline_actions"),
                        (a, parts.sum(0), "frozen_actions"),
                    ):
                        expected = oldrows[oldrows.candidate == candidate][
                            basecol
                        ].to_numpy()
                        if expected.shape != (4,):
                            raise ValueError("Four matched GRAIN actions required")
                        checks[check] = max(
                            checks[check], float(np.max(abs(prediction - expected)))
                        )
                    changes = parts.copy()
                    changes[0] -= baseline
                    new.setdefault((q, a, scope), []).append(
                        (actual, baseline, changes)
                    )
    sr, sp, sg = [], [], []
    for (q, a, space), values in source.items():
        energy = np.mean([x[0] for x in values])
        projection = np.mean([x[1] for x in values], axis=0)
        gram = np.mean([x[2] for x in values], axis=0)
        source[(q, a, space)] = program.projection_account(energy, projection, gram)
    for (q, a), values in source_actions.items():
        target = np.stack([x[0] for x in values])
        parts = np.stack([x[1] for x in values], axis=1)
        source[q, a, "grain_action"] = program.projection_account(
            *program.moments(target, parts)
        )
    for (q, a, space), x in source.items():
        meta = {"query_name": q, "candidate": a, "space": space, "authors": 4}
        sr.append(
            meta
            | {
                "target_energy": x["target_energy"],
                "sse_over_zero": x["sse_over_zero"],
                "total_projection": float(x["projection"].sum()),
                "total_energy_ratio": float(x["gram"].sum()),
            }
        )
        for i, c in enumerate(program.COMPONENTS):
            sp.append(
                meta
                | {
                    "component": c,
                    "signed_projection": x["projection"][i],
                    "energy_ratio": x["gram"][i, i],
                }
            )
            for j, other in enumerate(program.COMPONENTS):
                sg.append(
                    meta
                    | {
                        "component_i": c,
                        "component_j": other,
                        "gram_over_target_energy": x["gram"][i, j],
                    }
                )
    nr, nc, ng = [], [], []
    maximum_credit_error = 0.0
    for (q, a, scope), values in new.items():
        target, baseline = [np.stack([x[i] for x in values]) for i in (0, 1)]
        changes = np.stack([x[2] for x in values], axis=1)
        x = program.improvement_account(target, baseline, changes)
        maximum_credit_error = max(maximum_credit_error, x["closure_error"])
        meta = {
            "query_name": q,
            "candidate": a,
            "scope": scope,
            "authors": len(values),
            "actions": target.size,
        }
        nr.append(
            meta
            | {
                n: x[n]
                for n in (
                    "baseline_error",
                    "new_error",
                    "relative_improvement",
                    "closure_error",
                )
            }
        )
        for i, c in enumerate(program.CHANGES):
            nc.append(
                meta
                | {
                    "component": c,
                    "linear_error_term": x["linear_terms"][i],
                    "diagonal_error_term": x["gram"][i, i],
                    "symmetric_improvement_credit": x["credits"][i],
                }
            )
            for j, other in enumerate(program.CHANGES):
                ng.append(
                    meta
                    | {
                        "component_i": c,
                        "component_j": other,
                        "gram_over_baseline_error": x["gram"][i, j],
                    }
                )
    checks["credit_identity"] = maximum_credit_error
    if max(checks.values()) > 1e-10 or len(nr) != 8 or len(sr) != 12:
        raise ValueError("Account identity/cardinality failed: " + str(checks))
    tables = {
        "source_response": sr,
        "source_components": sp,
        "source_cross_terms": sg,
        "new8_improvement": nr,
        "new8_components": nc,
        "new8_cross_terms": ng,
    }
    dst = out / ("analysis" if folder == "components" else "replay_analysis")
    dst.mkdir(exist_ok=False)
    for n, rows in tables.items():
        pd.DataFrame(rows).to_csv(dst / (n + ".csv"), index=False)
    cpu = time.process_time() - start
    if cpu + freeze["cpu_seconds"] + m["prepare_cpu_seconds"] > CAP:
        raise TimeoutError("No account extension beyond budget")
    old.write(
        dst / "summary.json",
        {
            "status": "COMPLETE",
            "cpu_seconds": cpu,
            "checks": checks,
            "files": {
                p.name: old.sha(p)
                for p in dst.glob("*.csv")
                if not p.name.startswith(".")
            },
        },
    )
    print({"status": "EVALUATED", "checks": checks, "new8": nr}, flush=True)


def audit(out):
    """Exact replay and finite identities within the declared finite account."""
    start = time.process_time()
    m = cache.manifest(out)
    fs = [old.frozen(out, n) for n in ("components", "replay")]
    for name in fs[0]["files"]:
        a, b = [old.arrays(out / n / name) for n in ("components", "replay")]
        for k in a:
            np.testing.assert_array_equal(a[k], b[k])
    summaries = [
        old.read(out / n / "summary.json") for n in ("analysis", "replay_analysis")
    ]
    for n, s in zip(("analysis", "replay_analysis"), summaries, strict=True):
        old.verify_hashes(out / n, s["files"])
    if summaries[0]["files"] != summaries[1]["files"]:
        raise ValueError("Aggregate replay differs")
    total = (
        time.process_time()
        - start
        + m["prepare_cpu_seconds"]
        + sum(f["cpu_seconds"] for f in fs)
        + sum(s["cpu_seconds"] for s in summaries)
    )
    if total > CAP or len(fs[0]["files"]) != 48 or len(summaries[0]["files"]) != 6:
        raise ValueError("Finite account budget/cardinality failed")
    result = {
        "status": "PASS",
        "fits": 0,
        "new_neural_calls": 0,
        "packets": 48,
        "csv_replay": 6,
        "cumulative_cpu_seconds": total,
        "max_reconstruction_error": max(fs[0]["reconstruction_errors"].values()),
        "checks": summaries[0]["checks"],
    }
    old.write(out / "audit.json", result)
    print(result, flush=True)


def main():
    """Run stages separately; application access guard is process-local."""
    p = argparse.ArgumentParser()
    p.add_argument(
        "--stage", choices=("prepare", "apply", "evaluate", "audit"), required=True
    )
    p.add_argument("--folder", choices=("components", "replay"), default="components")
    p.add_argument("--output", type=Path, default=OUT)
    a = p.parse_args()
    with threadpool_limits(limits=1):
        if a.stage in ("apply", "evaluate"):
            globals()[a.stage](a.output, a.folder)
        else:
            globals()[a.stage](a.output)


if __name__ == "__main__":
    main()
