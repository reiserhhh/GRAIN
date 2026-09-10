#!/usr/bin/env python3
"""NEXT65: two full-context fits with all previous programs frozen as controls."""

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
from grain_research.vector_mechanisms import full_context_geometry as program
from grain_research.vector_mechanisms.stage_file_guard import install
from grain_research.vector_mechanisms.token_potential import (
    hidden_gradient,
    validate_members,
)
from scripts import run_grain_context_modulated_potential as previous

old = previous.old
OUT = old.BASE / "logic_full_context_geometry_v1"
PLAN = ROOT / "docs/GRAIN_FULL_CONTEXT_GEOMETRY_PLAN_V1.md"
CAP = 120
ARM = program.ARM


def model_path(out, q):
    """Resolve a single all4-source program per query."""
    return out / "models" / (q + ".npz")


def input_path(j, v, k):
    """Reuse exactly the earlier pure original/shifted packets without recapture."""
    return previous.input_path(previous.OUT, j, v, k)


def manifest(out):
    """Verify primary source lineage and frozen previous-control records."""
    m = old.load_manifest(out)
    old.verify_hashes(ROOT, m["baseline_freezes"])
    return m


def prepare(out):
    """Freeze plan, code and inherited data; do not open new8 targets for fitting."""
    start = time.process_time()
    base = previous.manifest(previous.OUT)
    old.frozen(previous.OUT, "models")
    old.frozen(previous.OUT, "predictions")
    if old.read(previous.OUT / "audit.json")["status"] != "PASS":
        raise ValueError("NEXT64 audit required")
    jobs = old.read(previous.OUT / "private_jobs.json")
    if len(jobs) != 12:
        raise ValueError("Frozen source4/new8 required")
    out.mkdir(exist_ok=False)
    old.write(out / "private_jobs.json", jobs)
    code = [
        PLAN,
        Path(__file__),
        ROOT / "grain_research/vector_mechanisms/full_context_geometry.py",
        ROOT / "tests/test_grain_full_context_geometry.py",
        ROOT / "scripts/run_grain_context_modulated_potential.py",
    ]
    inherited = [
        previous.OUT / f
        for f in (
            "models/freeze.json",
            "predictions/freeze.json",
            "analysis/summary.json",
            "audit.json",
        )
    ]
    old.write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_FULL_CONTEXT_GEOMETRY_V1",
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "code": base["code"] | {str(p.relative_to(ROOT)): old.sha(p) for p in code},
            "upstream_hashes": base["upstream_hashes"],
            "baseline_freezes": base["baseline_freezes"]
            | {str(p.relative_to(ROOT)): old.sha(p) for p in inherited},
            "jobs_sha256": old.sha(out / "private_jobs.json"),
            "prepare_cpu_seconds": time.process_time() - start,
            "planned_fits": 2,
            "max_cpu_seconds": CAP,
            "new_neural_calls": 0,
            "human_labels_read": False,
        },
    )
    print({"status": "PREPARED", "fits": 2}, flush=True)


def fit_query(out, query):
    """One solve on eight source views; deny every new8 input and target."""
    start = time.process_time()
    m = manifest(out)
    jobs = old.read(out / "private_jobs.json")
    src = sorted(
        [j for j in jobs if j["cohort"] == "source4"], key=lambda j: j["index"]
    )
    if [j["index"] for j in src] != list(range(4)):
        raise ValueError("Four source members required")
    qi = old.QUERIES.index(query)
    captures = previous.intervention.OUT / "capture"
    reads = [
        p
        for j in src
        for k in [j["query_keys"][qi]]
        for p in (
            old.input_path(k),
            input_path(j, "shifted", k),
            old.gradient_path(k),
            old.target_path(j, k),
            captures / (k + ".npz"),
        )
    ]
    old.check_paths(m, reads)
    fm = old.frozen(previous.OUT, "models")
    control = previous.path(previous.OUT, query, previous.ARMS[1])
    old.verify_hashes(
        previous.OUT / "models", {control.name: fm["files"][control.name]}
    )
    outputs = [model_path(out, query), out / "models" / (query + "_status.json")]
    probes = (
        old.forbidden()
        + [
            p
            for j in jobs
            if j["cohort"] == "new8"
            for k in j["query_keys"]
            for p in (old.input_path(k), old.gradient_path(k), old.target_path(j, k))
        ]
        + [
            previous.OUT / "analysis" / ("private_" + n + ".csv")
            for n in ("scores", "actions", "fields")
        ]
    )
    denied = install(
        reads + [control], outputs, probes, [ROOT / "scripts", ROOT / "grain_research"]
    )
    bodies, lasts, gradients, scores, ids, original = [], [], [], [], [], []
    for j in src:
        k = j["query_keys"][qi]
        _, body, last = old.pure_input(k)
        original.append(last)
        shifted = old.arrays(input_path(j, "shifted", k))["state"][-1].astype(float)
        cap = old.arrays(captures / (k + ".npz"))
        np.testing.assert_array_equal(shifted, cap["replacement_last"])
        records = (
            (
                last,
                old.selected(old.gradient_path(k), ("body_gradient",))["body_gradient"],
                old.selected(old.target_path(j, k), ("score",))["score"],
            ),
            (shifted, cap["body_gradient"], cap["score"]),
        )
        for l, g, y in records:
            bodies.append(body)
            lasts.append(l)
            gradients.append(g.astype(float))
            scores.append(float(y))
            ids.append(j["index"])
    model, diag = program.fit(
        bodies, lasts, gradients, scores, ids, original, old.arrays(control)
    )
    validate_members(model, [0, 1, 2, 3])
    np.savez(outputs[0], **model)
    old.write(
        outputs[1],
        {
            "status": "COMPLETE",
            "fits": 1,
            "cpu_seconds": time.process_time() - start,
            "denial_probes": len(probes),
            "extra_denials": len(denied) - len(probes),
            "diagnostics": {"query_name": query, **diag},
            "files": {outputs[0].name: old.sha(outputs[0])},
        },
    )


def fit(out):
    """Execute two solves with a first-query estimate and bounded wall/CPU cost."""
    start, wall = time.process_time(), time.monotonic()
    m = manifest(out)
    (out / "models").mkdir(exist_ok=False)
    statuses = []
    for q in old.QUERIES:
        subprocess.run(
            [
                sys.executable,
                __file__,
                "--stage",
                "fit-query",
                "--query",
                q,
                "--output",
                str(out),
            ],
            check=True,
            cwd=ROOT,
            timeout=max(0.1, CAP - (time.monotonic() - wall)),
        )
        statuses.append(old.read(out / "models" / (q + "_status.json")))
        estimate = (
            m["prepare_cpu_seconds"]
            + 2 * np.mean([s["cpu_seconds"] for s in statuses])
            + 30
        )
        print({"query": q, "estimated_cpu_with_reserve": estimate}, flush=True)
        if estimate > CAP:
            old.write(
                out / "bounded_stop.json",
                {"reason": "CPU projection", "queries_complete": len(statuses)},
            )
            raise TimeoutError("Preserve partial run; no rescue refit")
    cpu = (
        time.process_time()
        - start
        + m["prepare_cpu_seconds"]
        + sum(s["cpu_seconds"] for s in statuses)
    )
    if cpu > CAP or any(s["extra_denials"] for s in statuses):
        raise ValueError("Source cost/access violation")
    old.write(
        out / "models/freeze.json",
        {
            "status": "COMPLETE",
            "fits": 2,
            "cpu_seconds": cpu,
            "files": {
                p.name: old.sha(p)
                for p in (out / "models").iterdir()
                if p.suffix in (".json", ".npz") and not p.name.startswith(".")
            },
        },
    )


def predict(out, folder):
    """Teacher-free current-state application; controls are not refitted or reapplied."""
    start = time.process_time()
    m = manifest(out)
    fm = old.frozen(out, "models")
    jobs = old.read(out / "private_jobs.json")
    items = previous.items(jobs)
    inputs = [input_path(j, v, k) for j, v, q, k in items]
    old.check_paths(m, inputs)
    models = [model_path(out, q) for q in old.QUERIES]
    dst = out / folder
    dst.mkdir(exist_ok=False)
    outputs = [dst / (k + "_" + v + ".npz") for j, v, q, k in items]
    probes = (
        old.forbidden()
        + [
            p
            for j in jobs
            for k in j["query_keys"]
            for p in (old.gradient_path(k), old.target_path(j, k))
        ]
        + [
            previous.intervention.OUT / "capture" / (k + ".npz")
            for j in jobs
            if j["cohort"] == "source4"
            for k in j["query_keys"]
        ]
        + [
            previous.OUT / "analysis" / ("private_" + n + ".csv")
            for n in ("scores", "actions", "fields")
        ]
    )
    priorcpu = (
        old.frozen(out, "predictions")["cpu_seconds"] if folder == "replay" else 0
    )
    denied = install(
        inputs + models,
        outputs + [dst / "freeze.json"],
        probes,
        [ROOT / "scripts", ROOT / "grain_research"],
    )
    bank = {q: old.arrays(model_path(out, q)) for q in old.QUERIES}
    for j, v, q, k in items:
        data = old.arrays(input_path(j, v, k))
        if set(data) != {"state", "grain", "positions", "cos", "sin", "mask"}:
            raise ValueError("Pure input required")
        body, last = (
            data["state"][data["positions"]].astype(float),
            data["state"][-1].astype(float),
        )
        ds = old.body_directions(data)[:, data["positions"]]
        model = bank[q]
        validate_members(model, [0, 1, 2, 3])
        score, density = program.predict(body, last, model)
        gradient = hidden_gradient(density, model)
        np.savez(
            dst / (k + "_" + v + ".npz"),
            score=score,
            density=density,
            tau=np.einsum("th,fth->f", gradient, ds),
            allocation=np.einsum("th,fth->f", gradient, ds - ds.mean(1, keepdims=True)),
            **program.geometry_account(last, model),
        )
    cpu = time.process_time() - start
    if cpu + fm["cpu_seconds"] + priorcpu > CAP or len(denied) != len(probes):
        raise ValueError("Application cost/access violation")
    old.write(
        dst / "freeze.json",
        {
            "status": "COMPLETE",
            "cpu_seconds": cpu,
            "current_target_reads": 0,
            "denial_probes": len(probes),
            "extra_denials": len(denied) - len(probes),
            "files": {p.name: old.sha(p) for p in outputs},
        },
    )
    print({"status": "PREDICTED", "folder": folder, "cpu_seconds": cpu}, flush=True)


def evaluate(out, folder):
    """Add one candidate to the unchanged three-arm previously audited comparison."""
    start = time.process_time()
    m = manifest(out)
    old.frozen(out, folder)
    old.frozen(out, "models")
    source_summary = old.read(previous.OUT / "analysis/summary.json")
    old.verify_hashes(previous.OUT / "analysis", source_summary["files"])
    jobs = old.read(out / "private_jobs.json")
    paths = [old.gradient_path(k) for j in jobs for k in j["query_keys"]] + [
        previous.intervention.OUT / "capture" / (k + ".npz")
        for j in jobs
        if j["cohort"] == "source4"
        for k in j["query_keys"]
    ]
    old.check_paths(m, paths)
    originals = {
        n: pd.read_csv(previous.OUT / "analysis" / ("private_" + n + ".csv"))
        for n in ("scores", "actions", "fields")
    }
    scores, actions, fields, geometry = [], [], [], []
    for j, v, q, k in previous.items(jobs):
        p = old.arrays(out / folder / (k + "_" + v + ".npz"))
        model = old.arrays(model_path(out, q))
        meta = {
            "author_key": j["key"],
            "cohort": "source4_infit" if j["cohort"] == "source4" else "new8",
            "view": v,
            "query_name": q,
            "candidate": ARM,
        }
        same = lambda frame, key=j["key"], view=v, query=q: frame[
            (frame.author_key == key)
            & (frame.view == view)
            & (frame.query_name == query)
            & (frame.candidate == previous.ARMS[0])
        ]
        target = same(originals["scores"])
        if len(target) != 1:
            raise ValueError("Unique prior author/view/query required")
        scores.append(
            meta
            | {
                "actual": float(target.iloc[0].actual),
                "constant": float(model["origin"]),
                "predicted": float(p["score"]),
            }
        )
        a = same(originals["actions"]).sort_values("factor")
        if len(a) != 4:
            raise ValueError("All four factors required")
        for fi, row in enumerate(a.to_dict("records")):
            actions.append(
                meta
                | {
                    "factor": row["factor"],
                    "actual": row["actual"],
                    "predicted": float(p["tau"][fi]),
                    "actual_allocation": row["actual_allocation"],
                    "predicted_allocation": float(p["allocation"][fi]),
                }
            )
        gp = hidden_gradient(p["density"], model)
        g = old.selected(
            old.gradient_path(k)
            if v == "original"
            else previous.intervention.OUT / "capture" / (k + ".npz"),
            ("body_gradient",),
        )["body_gradient"].astype(float)
        target_density = len(g) * (g @ model["basis"]) * model["scale"]
        fields.append(
            meta
            | {
                "field_error_ratio": float(np.sum((gp - g) ** 2) / np.sum(g**2)),
                "normalized_density_loss": float(
                    np.mean((p["density"] - target_density) ** 2)
                    / model["density_scale"] ** 2
                ),
            }
        )
        geometry.append(
            meta
            | {
                n: float(p[n])
                for n in (
                    "inside_norm",
                    "outside_norm",
                    "radial_factor",
                    "in_span_kernel_mean",
                    "full_kernel_mean",
                    "factorization_error",
                )
            }
        )
    sd, ad, fd = [
        pd.concat((originals[n], pd.DataFrame(rows)), ignore_index=True)
        for n, rows in (("scores", scores), ("actions", actions), ("fields", fields))
    ]
    gd = pd.DataFrame(geometry)
    keys = ["author_key", "cohort", "view", "candidate"]
    sc = old.paired_queries(sd, keys, ["actual", "predicted", "constant"])
    ac = old.paired_queries(
        ad,
        keys + ["factor"],
        ["actual", "predicted", "actual_allocation", "predicted_allocation"],
    )
    src = ad[ad.cohort == "source4_infit"]
    idx = ["author_key", "cohort", "query_name", "candidate", "factor"]
    cols = ["actual", "predicted", "actual_allocation", "predicted_allocation"]
    delta = (
        src[src.view == "shifted"].set_index(idx)[cols]
        - src[src.view == "original"].set_index(idx)[cols]
    ).reset_index()
    dst = out / ("analysis" if folder == "predictions" else "replay_analysis")
    dst.mkdir(exist_ok=False)
    for f, n in (
        (sd, "scores"),
        (ad, "actions"),
        (fd, "fields"),
        (gd, "geometry"),
        (sc, "score_contrasts"),
        (ac, "action_contrasts"),
        (delta, "source_interaction"),
    ):
        f.to_csv(dst / ("private_" + n + ".csv"), index=False)
    tables = {
        n: previous.prior.aggregate(f, ks, score)
        for f, ks, n, score in (
            (sd, ["cohort", "view", "query_name"], "scores", True),
            (ad, ["cohort", "view", "query_name"], "actions", False),
            (ad, ["cohort", "view", "query_name", "factor"], "factor_actions", False),
            (sc, ["cohort", "view"], "score_contrasts", True),
            (ac, ["cohort", "view"], "action_contrasts", False),
            (ac, ["cohort", "view", "factor"], "factor_contrasts", False),
            (delta, ["cohort", "query_name"], "source_interaction", False),
        )
    }
    tables["fields"] = fd.groupby(
        ["cohort", "view", "query_name", "candidate"], as_index=False
    ).agg(
        field_error_ratio=("field_error_ratio", "mean"),
        normalized_density_loss=("normalized_density_loss", "mean"),
        authors=("author_key", "nunique"),
    )
    tables["geometry"] = gd.groupby(
        ["cohort", "view", "query_name"], as_index=False
    ).agg(
        radial_factor=("radial_factor", "mean"),
        in_span_kernel_mean=("in_span_kernel_mean", "mean"),
        full_kernel_mean=("full_kernel_mean", "mean"),
        outside_norm=("outside_norm", "mean"),
        factorization_error=("factorization_error", "max"),
        authors=("author_key", "nunique"),
    )
    tables["source_diagnostics"] = pd.DataFrame(
        [
            old.read(out / "models" / (q + "_status.json"))["diagnostics"]
            for q in old.QUERIES
        ]
    )
    criteria = {}
    for q in old.QUERIES:
        a = (
            tables["actions"]
            .query("cohort=='new8' and query_name==@q")
            .set_index("candidate")
        )
        s = (
            tables["scores"]
            .query("cohort=='new8' and query_name==@q")
            .set_index("candidate")
        )
        criteria[q] = {
            "better_action_than_augmented_additive": bool(
                a.loc[ARM, "action_rmse"] < a.loc[previous.ARMS[1], "action_rmse"]
            ),
            "action_sse_below_zero": bool(a.loc[ARM, "action_mse_over_zero"] < 1),
            "score_sse_below_source_constant": bool(
                s.loc[ARM, "score_mse_over_source_constant"] < 1
            ),
        }
    for n, f in tables.items():
        f.to_csv(dst / (n + ".csv"), index=False)
    old.write(
        dst / "summary.json",
        {
            "status": "COMPLETE",
            "promotion": all(all(c.values()) for c in criteria.values()),
            "criteria": criteria,
            "cpu_seconds": time.process_time() - start,
            "files": {
                p.name: old.sha(p)
                for p in dst.glob("*.csv")
                if not p.name.startswith(".")
            },
        },
    )
    print({"status": "EVALUATED", "criteria": criteria}, flush=True)


def audit(out):
    """Check frozen controls, exact replay, source normal blocks and actual derivatives."""
    start = time.process_time()
    manifest(out)
    freezes = [old.frozen(out, f) for f in ("models", "predictions", "replay")]
    jobs = old.read(out / "private_jobs.json")
    maximum = 0.0
    factor_error = 0.0
    for j, v, q, k in previous.items(jobs):
        p, r = [
            old.arrays(out / f / (k + "_" + v + ".npz"))
            for f in ("predictions", "replay")
        ]
        for n in p:
            np.testing.assert_array_equal(p[n], r[n])
        factor_error = max(factor_error, float(p["factorization_error"]))
        data = old.arrays(input_path(j, v, k))
        body, last = (
            data["state"][data["positions"]].astype(float),
            data["state"][-1].astype(float),
        )
        ds = old.body_directions(data)[:, data["positions"]]
        model = old.arrays(model_path(out, q))
        for fi in range(4):
            fd = (
                program.scalar(body + 1e-5 * ds[fi], last, model)
                - program.scalar(body - 1e-5 * ds[fi], last, model)
            ) / 2e-5
            maximum = max(maximum, abs(float(fd - p["tau"][fi])))
    csvs = [p for p in (out / "analysis").glob("*.csv") if not p.name.startswith(".")]
    for p in csvs:
        if old.sha(p) != old.sha(out / "replay_analysis" / p.name):
            raise ValueError("CSV replay differs")
    for name in ("scores", "actions"):
        raw = pd.read_csv(out / "analysis" / ("private_" + name + ".csv"))
        table = pd.read_csv(out / "analysis" / (name + ".csv"))
        for row in table.to_dict("records"):
            g = raw[
                (raw.cohort == row["cohort"])
                & (raw.view == row["view"])
                & (raw.query_name == row["query_name"])
                & (raw.candidate == row["candidate"])
            ]
            denom = (
                np.sum((g.actual - g.constant) ** 2)
                if name == "scores"
                else np.sum(g.actual**2)
            )
            key = (
                "score_mse_over_source_constant"
                if name == "scores"
                else "action_mse_over_zero"
            )
            np.testing.assert_allclose(
                np.sum((g.actual - g.predicted) ** 2) / denom,
                row[key],
                atol=1e-10,
                rtol=1e-10,
            )
        oldraw = pd.read_csv(previous.OUT / "analysis" / ("private_" + name + ".csv"))
        pd.testing.assert_frame_equal(
            raw[raw.candidate != ARM].reset_index(drop=True),
            oldraw,
            check_exact=False,
            atol=1e-14,
            rtol=1e-14,
        )
    statuses = [old.read(out / "models" / (q + "_status.json")) for q in old.QUERIES]
    cpu = time.process_time() - start
    total = (
        sum(f["cpu_seconds"] for f in freezes)
        + cpu
        + sum(
            old.read(out / f / "summary.json")["cpu_seconds"]
            for f in ("analysis", "replay_analysis")
        )
    )
    if (
        total > CAP
        or maximum > 1e-6
        or factor_error > 1e-10
        or len(csvs) != 17
        or sum(s["fits"] for s in statuses) != 2
        or any(s["extra_denials"] for s in statuses)
    ):
        raise ValueError(
            "Audit boundary differs: " + str((total, maximum, factor_error, len(csvs)))
        )
    old.write(
        out / "audit.json",
        {
            "status": "PASS",
            "new_fits": 2,
            "new_neural_calls": 0,
            "prediction_packets": 32,
            "csvs_replayed": len(csvs),
            "direction_checks": 128,
            "finite_difference_error": maximum,
            "kernel_factorization_error": factor_error,
            "additive_normal_block_error": max(
                s["diagnostics"]["additive_normal_block_error"] for s in statuses
            ),
            "cpu_seconds": cpu,
            "cumulative_compute_cpu_seconds": total,
        },
    )
    print(
        {"status": "PASS", "cpu_seconds": total, "finite_difference_error": maximum},
        flush=True,
    )


def main():
    """Run one stage in a fresh process with a single BLAS thread."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--stage",
        required=True,
        choices=("prepare", "fit", "fit-query", "predict", "evaluate", "audit"),
    )
    p.add_argument("--output", type=Path, default=OUT)
    p.add_argument("--query", choices=old.QUERIES, default=old.QUERIES[0])
    p.add_argument("--folder", choices=("predictions", "replay"), default="predictions")
    a = p.parse_args()
    with threadpool_limits(limits=1):
        if a.stage == "fit-query":
            fit_query(a.output, a.query)
        elif a.stage in ("predict", "evaluate"):
            globals()[a.stage](a.output, a.folder)
        else:
            globals()[a.stage](a.output)


if __name__ == "__main__":
    main()
