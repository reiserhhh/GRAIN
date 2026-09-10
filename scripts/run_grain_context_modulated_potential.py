#!/usr/bin/env python3
"""NEXT64: matched source-view conditional scalar, four fits, no native calls."""

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
from grain_research.vector_mechanisms import context_modulated_potential as program
from grain_research.vector_mechanisms.stage_file_guard import install
from grain_research.vector_mechanisms.token_potential import (
    hidden_gradient,
    validate_members,
)
from scripts import run_grain_context_reading_interaction as intervention
from scripts import run_grain_hermite_token_potential as prior

old = prior.old
OUT = old.BASE / "logic_context_modulated_potential_v1"
PLAN = ROOT / "docs/GRAIN_CONTEXT_MODULATED_POTENTIAL_PLAN_V1.md"
ARMS = ("frozen_original", *program.ARMS)
CAP = 120


def path(out, query, arm):
    """All4 source models only, never a pseudo-held-source augmented fit."""
    return (
        prior.model_path(prior.OUT, "all4", query)
        if arm == ARMS[0]
        else out / "models" / f"{query}_{arm}.npz"
    )


def input_path(out, job, view, key):
    """Use pure states; shifted packets contain no teacher targets."""
    return (
        old.input_path(key)
        if view == "original"
        else out / "private_shifted" / (key + ".npz")
    )


def items(jobs):
    """Enumerate originals for twelve authors and source-only shifted views."""
    return [
        (j, v, q, k)
        for j in jobs
        for v in (
            ("original", "shifted") if j["cohort"] == "source4" else ("original",)
        )
        for q, k in zip(old.QUERIES, j["query_keys"], strict=True)
    ]


def manifest(out):
    """Validate code and frozen source lineage without reading new8 targets."""
    m = old.load_manifest(out)
    old.verify_hashes(ROOT, m["baseline_freezes"])
    return m


def prepare(out):
    """Freeze four-source interventions, the plan and the existing pure-input cohort."""
    start = time.process_time()
    previous = prior.manifest(prior.OUT)
    intervention.verify(intervention.OUT)
    if old.read(intervention.OUT / "audit.json")["status"] != "PASS":
        raise ValueError("Audited native interaction required")
    fm = old.frozen(prior.OUT, "models")
    jobs = old.read(prior.OUT / "private_jobs.json")
    src = sorted(
        [j for j in jobs if j["cohort"] == "source4"], key=lambda j: j["index"]
    )
    if len(jobs) != 12 or [j["index"] for j in src] != list(range(4)):
        raise ValueError("Frozen source4/new8 required")
    inputs = [old.input_path(k) for j in jobs for k in j["query_keys"]]
    old.check_paths(previous, inputs)
    capture_status = old.read(intervention.OUT / "capture/status.json")
    if capture_status["status"] != "COMPLETE" or capture_status["calls"] != 9:
        raise ValueError("Complete native source interventions required")
    out.mkdir(exist_ok=False)
    (out / "private_shifted").mkdir()
    old.write(out / "private_jobs.json", jobs)
    captures = []
    for qi, q in enumerate(old.QUERIES):
        basepath = path(out, q, ARMS[0])
        old.verify_hashes(
            prior.OUT / "models", {basepath.name: fm["files"][basepath.name]}
        )
        banks = [old.pure_input(j["query_keys"][qi])[0] for j in src]
        for i, j in enumerate(src):
            k = j["query_keys"][qi]
            data = dict(banks[i])
            data["state"] = data["state"].copy()
            data["state"][-1] = np.mean(
                [b["state"][-1].astype(float) for t, b in enumerate(banks) if t != i],
                axis=0,
            )
            np.savez(out / "private_shifted" / (k + ".npz"), **data)
            cap = intervention.OUT / "capture" / (k + ".npz")
            np.testing.assert_array_equal(
                data["state"][-1],
                old.selected(cap, ("replacement_last",))["replacement_last"],
            )
            captures.append(cap)
    files = list((out / "private_shifted").glob("*.npz")) + captures
    code = [
        PLAN,
        Path(__file__),
        ROOT / "grain_research/vector_mechanisms/context_modulated_potential.py",
        ROOT / "tests/test_grain_context_modulated_potential.py",
    ]
    old.write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_CONTEXT_MODULATED_POTENTIAL_V1",
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "code": previous["code"]
            | {str(p.relative_to(ROOT)): old.sha(p) for p in code},
            "upstream_hashes": previous["upstream_hashes"]
            | {str(p.relative_to(ROOT)): old.sha(p) for p in files},
            "baseline_freezes": {
                str((prior.OUT / "models/freeze.json").relative_to(ROOT)): old.sha(
                    prior.OUT / "models/freeze.json"
                )
            },
            "jobs_sha256": old.sha(out / "private_jobs.json"),
            "prepare_cpu_seconds": time.process_time() - start,
            "planned_fits": 4,
            "max_cpu_seconds": CAP,
            "new_neural_calls": 0,
            "human_labels_read": False,
        },
    )
    print({"status": "PREPARED", "fits": 4}, flush=True)


def fit_query(out, query):
    """Fit matched source4 original+shifted views with every new8 data path denied."""
    start = time.process_time()
    m = manifest(out)
    jobs = old.read(out / "private_jobs.json")
    src = sorted(
        [j for j in jobs if j["cohort"] == "source4"], key=lambda j: j["index"]
    )
    qi = old.QUERIES.index(query)
    reads = [
        p
        for j in src
        for k in [j["query_keys"][qi]]
        for p in (
            old.input_path(k),
            input_path(out, j, "shifted", k),
            old.gradient_path(k),
            old.target_path(j, k),
            intervention.OUT / "capture" / (k + ".npz"),
        )
    ]
    old.check_paths(m, reads)
    baseline = path(out, query, ARMS[0])
    fm = old.frozen(prior.OUT, "models")
    old.verify_hashes(prior.OUT / "models", {baseline.name: fm["files"][baseline.name]})
    outputs = [path(out, query, a) for a in ARMS[1:]]
    status = out / "models" / (query + "_status.json")
    probes = old.forbidden() + [
        p
        for j in jobs
        if j["cohort"] == "new8"
        for k in j["query_keys"]
        for p in (old.input_path(k), old.gradient_path(k), old.target_path(j, k))
    ]
    denied = install(
        reads + [baseline],
        outputs + [status],
        probes,
        [ROOT / "scripts", ROOT / "grain_research"],
    )
    bodies, lasts, gs, ys, ids, original_lasts = [], [], [], [], [], []
    for j in src:
        k = j["query_keys"][qi]
        _, body, last = old.pure_input(k)
        original_lasts.append(last)
        shift = old.arrays(input_path(out, j, "shifted", k))["state"][-1].astype(float)
        cap = old.arrays(intervention.OUT / "capture" / (k + ".npz"))
        np.testing.assert_array_equal(shift, cap["replacement_last"])
        for l, g, y in (
            (
                last,
                old.selected(old.gradient_path(k), ("body_gradient",))["body_gradient"],
                old.selected(old.target_path(j, k), ("score",))["score"],
            ),
            (shift, cap["body_gradient"], cap["score"]),
        ):
            bodies.append(body)
            lasts.append(l)
            gs.append(g.astype(float))
            ys.append(float(y))
            ids.append(j["index"])
    base = old.arrays(baseline)
    validate_members(base, [0, 1, 2, 3])
    models, diag = program.fit_pair(bodies, lasts, gs, ys, ids, original_lasts, base)
    for p, model in zip(outputs, models, strict=True):
        np.savez(p, **model)
    old.write(
        status,
        {
            "status": "COMPLETE",
            "fits": 2,
            "cpu_seconds": time.process_time() - start,
            "denial_probes": len(probes),
            "extra_denials": len(denied) - len(probes),
            "diagnostics": [{"query_name": query, **d} for d in diag],
            "files": {p.name: old.sha(p) for p in outputs},
        },
    )


def fit(out):
    """Four total solves with a first-query estimate; no selected rescue rerun."""
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
        print({"query": q, "projected_cpu_with_reserve": estimate}, flush=True)
        if estimate > CAP:
            old.write(
                out / "bounded_stop.json",
                {"reason": "Projected CPU cap", "queries_complete": len(statuses)},
            )
            raise TimeoutError("Preserved partial artifacts")
    cpu = (
        time.process_time()
        - start
        + m["prepare_cpu_seconds"]
        + sum(s["cpu_seconds"] for s in statuses)
    )
    if cpu > CAP or any(s["extra_denials"] for s in statuses):
        raise ValueError("Fit access/cost boundary")
    old.write(
        out / "models/freeze.json",
        {
            "status": "COMPLETE",
            "fits": 4,
            "cpu_seconds": cpu,
            "files": {
                p.name: old.sha(p)
                for p in (out / "models").iterdir()
                if p.suffix in (".json", ".npz") and not p.name.startswith(".")
            },
        },
    )


def predict(out, folder):
    """Apply all three all4 programs to pure H16; no native teacher accessible."""
    start = time.process_time()
    m = manifest(out)
    fm = old.frozen(out, "models")
    old.frozen(prior.OUT, "models")
    jobs = old.read(out / "private_jobs.json")
    inventory = items(jobs)
    reads = [input_path(out, j, v, k) for j, v, q, k in inventory]
    old.check_paths(m, reads)
    models = [path(out, q, a) for q in old.QUERIES for a in ARMS]
    dst = out / folder
    dst.mkdir(exist_ok=False)
    outputs = [dst / (k + "_" + v + ".npz") for j, v, q, k in inventory]
    probes = (
        old.forbidden()
        + [
            p
            for j in jobs
            for k in j["query_keys"]
            for p in (old.gradient_path(k), old.target_path(j, k))
        ]
        + [
            intervention.OUT / "capture" / (k + ".npz")
            for j in jobs
            if j["cohort"] == "source4"
            for k in j["query_keys"]
        ]
    )
    prior_cpu = (
        old.frozen(out, "predictions")["cpu_seconds"] if folder == "replay" else 0
    )
    denied = install(
        reads + models,
        outputs + [dst / "freeze.json"],
        probes,
        [ROOT / "scripts", ROOT / "grain_research"],
    )
    bank = {(q, a): old.arrays(path(out, q, a)) for q in old.QUERIES for a in ARMS}
    for j, v, q, k in inventory:
        data = old.arrays(input_path(out, j, v, k))
        if set(data) != {"state", "grain", "positions", "cos", "sin", "mask"}:
            raise ValueError("Teacher-free six-field inputs required")
        body, last = (
            data["state"][data["positions"]].astype(float),
            data["state"][-1].astype(float),
        )
        ds = old.body_directions(data)[:, data["positions"]]
        rows = []
        for a in ARMS:
            model = bank[q, a]
            validate_members(model, [0, 1, 2, 3])
            score, density = program.predict(body, last, model)
            g = hidden_gradient(density, model)
            rows.append(
                (
                    score,
                    density,
                    np.einsum("th,fth->f", g, ds),
                    np.einsum("th,fth->f", g, ds - ds.mean(1, keepdims=True)),
                )
            )
        context = bank[q, ARMS[1]]
        delta = last - context["context_mean"]
        inside = (delta @ context["context_basis"]) @ context["context_basis"].T
        np.savez(
            dst / (k + "_" + v + ".npz"),
            **{
                n: np.asarray(x)
                for n, x in zip(
                    ("score", "density", "tau", "allocation"),
                    zip(*rows, strict=True),
                    strict=True,
                )
            },
            context_total_norm=np.linalg.norm(delta),
            context_inside_norm=np.linalg.norm(inside),
            context_outside_norm=np.linalg.norm(delta - inside),
        )
    cpu = time.process_time() - start
    if cpu + fm["cpu_seconds"] + prior_cpu > CAP or len(denied) != len(probes):
        raise ValueError("Prediction access/cost boundary")
    old.write(
        dst / "freeze.json",
        {
            "status": "COMPLETE",
            "cpu_seconds": cpu,
            "denial_probes": len(probes),
            "extra_denials": len(denied) - len(probes),
            "current_target_reads": 0,
            "files": {p.name: old.sha(p) for p in outputs},
        },
    )
    print(
        {
            "status": "PREDICTED",
            "folder": folder,
            "packets": len(outputs),
            "cpu_seconds": cpu,
        },
        flush=True,
    )


def evaluate(out, folder):
    """Join targets only after prediction freeze; preserve all views, queries, factors."""
    start = time.process_time()
    m = manifest(out)
    old.frozen(out, "models")
    old.frozen(out, folder)
    jobs = old.read(out / "private_jobs.json")
    paths = [
        p
        for j in jobs
        for k in j["query_keys"]
        for p in (old.gradient_path(k), old.target_path(j, k))
    ]
    paths += [
        intervention.OUT / "capture" / (k + ".npz")
        for j in jobs
        if j["cohort"] == "source4"
        for k in j["query_keys"]
    ]
    old.check_paths(m, paths)
    dst = out / ("analysis" if folder == "predictions" else "replay_analysis")
    dst.mkdir(exist_ok=False)
    scores, actions, fields, coverage = [], [], [], []
    bank = {(q, a): old.arrays(path(out, q, a)) for q in old.QUERIES for a in ARMS}
    for j, v, q, k in items(jobs):
        pred = old.arrays(out / folder / (k + "_" + v + ".npz"))
        data = old.arrays(input_path(out, j, v, k))
        ds = old.body_directions(data)[:, data["positions"]]
        if v == "original":
            target = old.selected(old.target_path(j, k), ("score", "tau"))
            g = old.selected(old.gradient_path(k), ("body_gradient",))[
                "body_gradient"
            ].astype(float)
        else:
            target = old.arrays(intervention.OUT / "capture" / (k + ".npz"))
            g = target["body_gradient"].astype(float)
        tau = np.einsum("th,fth->f", g, ds)
        if v == "original":
            np.testing.assert_allclose(tau, target["tau"], atol=1e-5, rtol=3e-4)
        alloc = np.einsum("th,fth->f", g, ds - ds.mean(1, keepdims=True))
        cohort = "source4_infit" if j["cohort"] == "source4" else "new8"
        coverage.append(
            {
                "author_key": j["key"],
                "cohort": cohort,
                "query_name": q,
                "view": v,
                **{
                    n: float(pred[n])
                    for n in (
                        "context_total_norm",
                        "context_inside_norm",
                        "context_outside_norm",
                    )
                },
            }
        )
        for ai, a in enumerate(ARMS):
            model = bank[q, a]
            meta = {
                "author_key": j["key"],
                "cohort": cohort,
                "query_name": q,
                "view": v,
                "candidate": a,
            }
            scores.append(
                meta
                | {
                    "actual": float(target["score"]),
                    "predicted": float(pred["score"][ai]),
                    "constant": float(model["origin"]),
                }
            )
            gp = hidden_gradient(pred["density"][ai], model)
            projected = len(g) * (g @ model["basis"]) * model["scale"]
            fields.append(
                meta
                | {
                    "field_error_ratio": float(np.sum((gp - g) ** 2) / np.sum(g**2)),
                    "normalized_density_loss": float(
                        np.mean((pred["density"][ai] - projected) ** 2)
                        / model["density_scale"] ** 2
                    ),
                }
            )
            for fi in range(4):
                actions.append(
                    meta
                    | {
                        "factor": f"whole:F{fi + 1:02d}",
                        "actual": tau[fi],
                        "predicted": pred["tau"][ai, fi],
                        "actual_allocation": alloc[fi],
                        "predicted_allocation": pred["allocation"][ai, fi],
                    }
                )
    sd, ad, fd, cd = map(pd.DataFrame, (scores, actions, fields, coverage))
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
    for frame, name in (
        (sd, "scores"),
        (ad, "actions"),
        (fd, "fields"),
        (cd, "context_coverage"),
        (sc, "score_contrasts"),
        (ac, "action_contrasts"),
        (delta, "source_interaction"),
    ):
        frame.to_csv(dst / ("private_" + name + ".csv"), index=False)
    tables = {
        n: prior.aggregate(f, ks, is_score)
        for f, ks, n, is_score in (
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
    cd["inside_energy_fraction"] = (cd.context_inside_norm / cd.context_total_norm) ** 2
    tables["context_coverage"] = cd.groupby(
        ["cohort", "view", "query_name"], as_index=False
    ).agg(
        inside_energy_fraction=("inside_energy_fraction", "mean"),
        outside_norm=("context_outside_norm", "mean"),
        authors=("author_key", "nunique"),
    )
    tables["source_diagnostics"] = pd.DataFrame(
        [
            d
            for q in old.QUERIES
            for d in old.read(out / "models" / (q + "_status.json"))["diagnostics"]
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
            "better_action_than_matched_control": bool(
                a.loc[ARMS[2], "action_rmse"] < a.loc[ARMS[1], "action_rmse"]
            ),
            "action_sse_below_zero": bool(a.loc[ARMS[2], "action_mse_over_zero"] < 1),
            "score_sse_below_source_constant": bool(
                s.loc[ARMS[2], "score_mse_over_source_constant"] < 1
            ),
        }
    for n, table in tables.items():
        table.to_csv(dst / (n + ".csv"), index=False)
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
    """Check exact replay, scalar derivatives, source objective and independent SSE."""
    start = time.process_time()
    manifest(out)
    freezes = [old.frozen(out, f) for f in ("models", "predictions", "replay")]
    jobs = old.read(out / "private_jobs.json")
    maximum = 0.0
    for j, v, q, k in items(jobs):
        p, r = [
            old.arrays(out / f / (k + "_" + v + ".npz"))
            for f in ("predictions", "replay")
        ]
        for name in p:
            np.testing.assert_array_equal(p[name], r[name])
        data = old.arrays(input_path(out, j, v, k))
        body, last = (
            data["state"][data["positions"]].astype(float),
            data["state"][-1].astype(float),
        )
        ds = old.body_directions(data)[:, data["positions"]]
        for ai, a in enumerate(ARMS):
            model = old.arrays(path(out, q, a))
            validate_members(model, [0, 1, 2, 3])
            for fi in range(4):
                fd = (
                    program.scalar(body + 1e-5 * ds[fi], last, model)
                    - program.scalar(body - 1e-5 * ds[fi], last, model)
                ) / 2e-5
                maximum = max(maximum, abs(float(fd - p["tau"][ai, fi])))
    csvs = [p for p in (out / "analysis").glob("*.csv") if not p.name.startswith(".")]
    for p in csvs:
        if old.sha(p) != old.sha(out / "replay_analysis" / p.name):
            raise ValueError("CSV replay differs")
    for name in ("scores", "actions"):
        raw = pd.read_csv(out / "analysis" / ("private_" + name + ".csv"))
        summary = pd.read_csv(out / "analysis" / (name + ".csv"))
        for row in summary.to_dict("records"):
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
            metric = (
                "score_mse_over_source_constant"
                if name == "scores"
                else "action_mse_over_zero"
            )
            np.testing.assert_allclose(
                np.sum((g.predicted - g.actual) ** 2) / denom,
                row[metric],
                rtol=1e-10,
                atol=1e-10,
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
        maximum > 1e-6
        or total > CAP
        or len(csvs) != 17
        or sum(s["fits"] for s in statuses) != 4
        or any(s["extra_denials"] for s in statuses)
    ):
        raise ValueError("Audit boundary differs: " + str((maximum, total, len(csvs))))
    old.write(
        out / "audit.json",
        {
            "status": "PASS",
            "direction_finite_difference_error": maximum,
            "direction_checks": 384,
            "csvs_replayed": len(csvs),
            "prediction_packets": 32,
            "fits": 4,
            "new_neural_calls": 0,
            "cpu_seconds": cpu,
            "cumulative_compute_cpu_seconds": total,
            "prediction_replay": "bitwise exact",
        },
    )
    print(
        {"status": "PASS", "cpu_seconds": total, "finite_difference_error": maximum},
        flush=True,
    )


def main():
    """Run one isolated bounded stage with single-threaded BLAS."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=("prepare", "fit", "fit-query", "predict", "evaluate", "audit"),
    )
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--query", choices=old.QUERIES, default=old.QUERIES[0])
    parser.add_argument(
        "--folder", choices=("predictions", "replay"), default="predictions"
    )
    a = parser.parse_args()
    with threadpool_limits(limits=1):
        if a.stage == "fit-query":
            fit_query(a.output, a.query)
        elif a.stage in ("predict", "evaluate"):
            globals()[a.stage](a.output, a.folder)
        else:
            globals()[a.stage](a.output)


if __name__ == "__main__":
    main()
