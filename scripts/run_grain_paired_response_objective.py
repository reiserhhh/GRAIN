#!/usr/bin/env python3
"""NEXT66: same-scalar paired response teaching, two fits and frozen controls."""

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
from grain_research.vector_mechanisms import full_context_geometry as scalar
from grain_research.vector_mechanisms import paired_response_objective as program
from grain_research.vector_mechanisms.stage_file_guard import install
from grain_research.vector_mechanisms.token_potential import (
    hidden_gradient,
    validate_members,
)
from scripts import run_grain_full_context_geometry as prior
from scripts.report_grain_full_context_geometry import paired_channel_account

old, previous = prior.old, prior.previous
OUT = old.BASE / "logic_paired_response_objective_v1"
PLAN = ROOT / "docs/GRAIN_PAIRED_RESPONSE_OBJECTIVE_PLAN_V1.md"
CAP, ARM = 120, program.ARM
ARMS = (*previous.ARMS, prior.ARM, ARM)


def manifest(out):
    """Check frozen code, source inputs and earlier result lineage."""
    return prior.manifest(out)


def prepare(out):
    """Freeze the pre-result plan/code and source lineage without target selection."""
    start = time.process_time()
    m = prior.manifest(prior.OUT)
    for name in ("models", "predictions"):
        old.frozen(prior.OUT, name)
    if old.read(prior.OUT / "audit.json")["status"] != "PASS":
        raise ValueError("NEXT65 audit required")
    jobs = old.read(prior.OUT / "private_jobs.json")
    if len(jobs) != 12:
        raise ValueError("Source4/new8 required")
    out.mkdir(exist_ok=False)
    old.write(out / "private_jobs.json", jobs)
    code = [
        PLAN,
        Path(__file__),
        ROOT / "grain_research/vector_mechanisms/paired_response_objective.py",
        ROOT / "tests/test_grain_paired_response_objective.py",
        ROOT / "scripts/report_grain_full_context_geometry.py",
        ROOT / "tests/test_grain_paired_objective_account.py",
    ]
    files = [
        prior.OUT / n
        for n in (
            "models/freeze.json",
            "predictions/freeze.json",
            "analysis/summary.json",
            "audit.json",
        )
    ]
    old.write(
        out / "manifest.json",
        {
            **m,
            "protocol": "GRAIN_PAIRED_RESPONSE_OBJECTIVE_V1",
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "code": m["code"] | {str(p.relative_to(ROOT)): old.sha(p) for p in code},
            "baseline_freezes": m["baseline_freezes"]
            | {str(p.relative_to(ROOT)): old.sha(p) for p in files},
            "jobs_sha256": old.sha(out / "private_jobs.json"),
            "prepare_cpu_seconds": time.process_time() - start,
            "planned_fits": 2,
            "max_cpu_seconds": CAP,
            "new_neural_calls": 0,
            "human_labels_read": False,
        },
    )
    print({"status": "PREPARED", "fits": 2}, flush=True)


def source_paths(jobs, qi):
    """Enumerate only source-view input and teacher files for one query."""
    return [
        p
        for j in jobs
        for k in [j["query_keys"][qi]]
        for p in (
            old.input_path(k),
            prior.input_path(j, "shifted", k),
            old.gradient_path(k),
            old.target_path(j, k),
            previous.intervention.OUT / "capture" / (k + ".npz"),
        )
    ]


def source_data(jobs, qi):
    """Load the previously frozen pairs; native data stay on the source side."""
    b, l, g, y, ids = [], [], [], [], []
    for j in jobs:
        k = j["query_keys"][qi]
        _, body, last = old.pure_input(k)
        shifted = old.arrays(prior.input_path(j, "shifted", k))["state"][-1].astype(
            float
        )
        cap = old.arrays(previous.intervention.OUT / "capture" / (k + ".npz"))
        np.testing.assert_array_equal(shifted, cap["replacement_last"])
        records = (
            (
                last,
                old.selected(old.gradient_path(k), ("body_gradient",))["body_gradient"],
                old.selected(old.target_path(j, k), ("score",))["score"],
            ),
            (shifted, cap["body_gradient"], cap["score"]),
        )
        for last0, g0, y0 in records:
            b.append(body)
            l.append(last0)
            g.append(g0.astype(float))
            y.append(float(y0))
            ids.append(j["index"])
    return b, l, g, y, ids


def fit_query(out, query):
    """One new solve with every current/new8 target denied."""
    start = time.process_time()
    m = manifest(out)
    jobs = old.read(out / "private_jobs.json")
    src = sorted(
        [j for j in jobs if j["cohort"] == "source4"], key=lambda j: j["index"]
    )
    if [j["index"] for j in src] != list(range(4)):
        raise ValueError("Frozen source membership differs")
    qi = old.QUERIES.index(query)
    reads = source_paths(src, qi)
    old.check_paths(m, reads)
    fm = old.frozen(prior.OUT, "models")
    control = prior.model_path(prior.OUT, query)
    old.verify_hashes(prior.OUT / "models", {control.name: fm["files"][control.name]})
    dst = prior.model_path(out, query)
    status = out / "models" / (query + "_status.json")
    probes = old.forbidden() + [
        p
        for j in jobs
        if j["cohort"] == "new8"
        for k in j["query_keys"]
        for p in (old.input_path(k), old.gradient_path(k), old.target_path(j, k))
    ]
    probes += [
        root / "analysis" / ("private_" + n + ".csv")
        for root in (prior.OUT, previous.OUT)
        for n in ("scores", "actions", "fields")
    ]
    denied = install(
        reads + [control],
        [dst, status],
        probes,
        [ROOT / "scripts", ROOT / "grain_research"],
    )
    model, diag = program.fit(*source_data(src, qi), old.arrays(control))
    validate_members(model, [0, 1, 2, 3])
    np.savez(dst, **model)
    old.write(
        status,
        {
            "status": "COMPLETE",
            "fits": 1,
            "cpu_seconds": time.process_time() - start,
            "denial_probes": len(probes),
            "extra_denials": len(denied) - len(probes),
            "diagnostics": {"query_name": query, **diag},
            "files": {dst.name: old.sha(dst)},
        },
    )


def fit(out):
    """Two sequential fits only, with source-first budget projection."""
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
            raise TimeoutError("Preserve partial fit; no rescue refit")
    cpu = (
        time.process_time()
        - start
        + m["prepare_cpu_seconds"]
        + sum(s["cpu_seconds"] for s in statuses)
    )
    if cpu > CAP or any(s["extra_denials"] for s in statuses):
        raise ValueError("Fit cost/access violation")
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
    """Reuse the exact NEXT65 pure-input predictor; the function family is unchanged."""
    prior.predict(out, folder)


def density(out, folder, k, view, arm):
    """Read a previously frozen prediction only; no control is refitted/reapplied."""
    if arm in previous.ARMS:
        return old.arrays(previous.OUT / "predictions" / (k + "_" + view + ".npz"))[
            "density"
        ][previous.ARMS.index(arm)]
    root, sub = (prior.OUT, "predictions") if arm == prior.ARM else (out, folder)
    return old.arrays(root / sub / (k + "_" + view + ".npz"))["density"]


def evaluate(out, folder):
    """Append one candidate and every paired channel to the frozen four-arm results."""
    start = time.process_time()
    m = manifest(out)
    old.frozen(out, folder)
    old.frozen(out, "models")
    summary = old.read(prior.OUT / "analysis/summary.json")
    old.verify_hashes(prior.OUT / "analysis", summary["files"])
    jobs = old.read(out / "private_jobs.json")
    paths = [old.gradient_path(k) for j in jobs for k in j["query_keys"]]
    paths += [
        previous.intervention.OUT / "capture" / (k + ".npz")
        for j in jobs
        if j["cohort"] == "source4"
        for k in j["query_keys"]
    ]
    old.check_paths(m, paths)
    originals = {
        n: pd.read_csv(prior.OUT / "analysis" / ("private_" + n + ".csv"))
        for n in ("scores", "actions", "fields")
    }
    new = {n: [] for n in originals}
    targets = {}
    bank = {q: old.arrays(prior.model_path(out, q)) for q in old.QUERIES}
    for j, v, q, k in previous.items(jobs):
        p = old.arrays(out / folder / (k + "_" + v + ".npz"))
        model = bank[q]
        meta = {
            "author_key": j["key"],
            "cohort": "source4_infit" if j["cohort"] == "source4" else "new8",
            "view": v,
            "query_name": q,
            "candidate": ARM,
        }

        def same(frame, key=j["key"], view=v, query=q):
            return frame[
                (frame.author_key == key)
                & (frame.view == view)
                & (frame.query_name == query)
                & (frame.candidate == previous.ARMS[0])
            ]

        s = same(originals["scores"])
        a = same(originals["actions"]).sort_values("factor")
        if len(s) != 1 or len(a) != 4:
            raise ValueError("Unique query/view/author and four factors required")
        new["scores"].append(
            meta
            | {
                "actual": float(s.iloc[0].actual),
                "constant": float(model["origin"]),
                "predicted": float(p["score"]),
            }
        )
        for fi, row in enumerate(a.to_dict("records")):
            new["actions"].append(
                meta
                | {
                    "factor": row["factor"],
                    "actual": row["actual"],
                    "predicted": float(p["tau"][fi]),
                    "actual_allocation": row["actual_allocation"],
                    "predicted_allocation": float(p["allocation"][fi]),
                }
            )
        g = old.selected(
            old.gradient_path(k)
            if v == "original"
            else previous.intervention.OUT / "capture" / (k + ".npz"),
            ("body_gradient",),
        )["body_gradient"].astype(float)
        target = len(g) * (g @ model["basis"]) * model["scale"]
        if j["cohort"] == "source4":
            targets[q, j["index"], v] = target
        gp = hidden_gradient(p["density"], model)
        new["fields"].append(
            meta
            | {
                "field_error_ratio": float(np.sum((gp - g) ** 2) / np.sum(g**2)),
                "normalized_density_loss": float(
                    np.mean((p["density"] - target) ** 2) / model["density_scale"] ** 2
                ),
            }
        )
    sd, ad, fd = [
        pd.concat((originals[n], pd.DataFrame(new[n])), ignore_index=True)
        for n in ("scores", "actions", "fields")
    ]
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
    for n, frame in (
        ("scores", sd),
        ("actions", ad),
        ("fields", fd),
        ("score_contrasts", sc),
        ("action_contrasts", ac),
        ("source_interaction", delta),
    ):
        frame.to_csv(dst / ("private_" + n + ".csv"), index=False)
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
    rows = []
    source = sorted(
        [j for j in jobs if j["cohort"] == "source4"], key=lambda j: j["index"]
    )
    for qi, q in enumerate(old.QUERIES):
        t0 = [targets[q, j["index"], "original"] for j in source]
        t1 = [targets[q, j["index"], "shifted"] for j in source]
        for arm in ARMS:
            p0 = [
                density(out, folder, j["query_keys"][qi], "original", arm)
                for j in source
            ]
            p1 = [
                density(out, folder, j["query_keys"][qi], "shifted", arm)
                for j in source
            ]
            rows.append(
                {
                    "query_name": q,
                    "candidate": arm,
                    "source_authors": 4,
                    **paired_channel_account(
                        t0, t1, p0, p1, float(bank[q]["density_scale"])
                    ),
                }
            )
    tables["paired_objective_account"] = pd.DataFrame(rows)
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
    for name, frame in tables.items():
        frame.to_csv(dst / (name + ".csv"), index=False)
    old.write(
        dst / "summary.json",
        {
            "status": "COMPLETE",
            "promotion": all(all(x.values()) for x in criteria.values()),
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
    """Replay actual predictions, objective identities and unchanged controls."""
    start = time.process_time()
    m = manifest(out)
    freezes = [old.frozen(out, f) for f in ("models", "predictions", "replay")]
    jobs = old.read(out / "private_jobs.json")
    maximum = 0.0
    for j, v, q, k in previous.items(jobs):
        pred, replay = [
            old.arrays(out / f / (k + "_" + v + ".npz"))
            for f in ("predictions", "replay")
        ]
        for name in pred:
            np.testing.assert_array_equal(pred[name], replay[name])
        data = old.arrays(prior.input_path(j, v, k))
        b, last = (
            data["state"][data["positions"]].astype(float),
            data["state"][-1].astype(float),
        )
        ds = old.body_directions(data)[:, data["positions"]]
        model = old.arrays(prior.model_path(out, q))
        for i in range(4):
            fd = (
                scalar.scalar(b + 1e-5 * ds[i], last, model)
                - scalar.scalar(b - 1e-5 * ds[i], last, model)
            ) / 2e-5
            maximum = max(maximum, abs(float(fd - pred["tau"][i])))
    csvs = [p for p in (out / "analysis").glob("*.csv") if not p.name.startswith(".")]
    for p in csvs:
        if old.sha(p) != old.sha(out / "replay_analysis" / p.name):
            raise ValueError("CSV replay mismatch")
    for name in ("scores", "actions", "fields"):
        current = pd.read_csv(out / "analysis" / ("private_" + name + ".csv"))
        original = pd.read_csv(prior.OUT / "analysis" / ("private_" + name + ".csv"))
        pd.testing.assert_frame_equal(
            current[current.candidate != ARM].reset_index(drop=True),
            original,
            check_exact=False,
            atol=1e-14,
            rtol=1e-14,
        )
        if name == "fields":
            continue
        for row in pd.read_csv(out / "analysis" / (name + ".csv")).to_dict("records"):
            group = current[
                (current.cohort == row["cohort"])
                & (current.view == row["view"])
                & (current.query_name == row["query_name"])
                & (current.candidate == row["candidate"])
            ]
            denom = (
                np.sum((group.actual - group.constant) ** 2)
                if name == "scores"
                else np.sum(group.actual**2)
            )
            metric = (
                "score_mse_over_source_constant"
                if name == "scores"
                else "action_mse_over_zero"
            )
            np.testing.assert_allclose(
                np.sum((group.actual - group.predicted) ** 2) / denom,
                row[metric],
                atol=1e-10,
                rtol=1e-10,
            )
    source = sorted(
        [j for j in jobs if j["cohort"] == "source4"], key=lambda j: j["index"]
    )
    error = 0.0
    for qi, q in enumerate(old.QUERIES):
        old.check_paths(m, source_paths(source, qi))
        data = source_data(source, qi)
        control = old.arrays(prior.model_path(prior.OUT, q))
        model = old.arrays(prior.model_path(out, q))
        for name in control:
            if name not in ("coef", "gram", "rhs", "arm"):
                np.testing.assert_array_equal(control[name], model[name])
        dg, dr, energy, diag = program.normals(*data, control)
        np.testing.assert_array_equal(dg, model["delta_gram"])
        np.testing.assert_array_equal(dr, model["delta_rhs"])
        np.testing.assert_array_equal(model["gram"], control["gram"] + dg)
        np.testing.assert_array_equal(model["rhs"], control["rhs"] + dr)
        loss = program.objective(*data, model, energy)
        frozen = old.read(out / "models" / (q + "_status.json"))["diagnostics"]
        expected = np.array(
            [
                frozen[n]
                for n in (
                    "source_score_loss",
                    "source_density_loss",
                    "source_delta_loss",
                    "source_penalty",
                )
            ]
        )
        error = max(
            error, float(np.max(abs(loss - expected))), diag["base_normal_error"]
        )
        account = pd.read_csv(out / "analysis/paired_objective_account.csv")
        row = account[(account.query_name == q) & (account.candidate == ARM)].iloc[0]
        np.testing.assert_allclose(
            row.half_difference_sse_over_zero, loss[2], atol=1e-10, rtol=1e-10
        )
    statuses = [old.read(out / "models" / (q + "_status.json")) for q in old.QUERIES]
    cpu = time.process_time() - start
    total = (
        sum(x["cpu_seconds"] for x in freezes)
        + cpu
        + sum(
            old.read(out / f / "summary.json")["cpu_seconds"]
            for f in ("analysis", "replay_analysis")
        )
    )
    if (
        total > CAP
        or maximum > 1e-6
        or error > 1e-8
        or len(csvs) != 16
        or sum(s["fits"] for s in statuses) != 2
    ):
        raise ValueError(
            "Primary audit boundary differs: " + str((total, maximum, error, len(csvs)))
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
            "objective_reconstruction_error": error,
            "cpu_seconds": cpu,
            "cumulative_compute_cpu_seconds": total,
        },
    )
    print(
        {"status": "PASS", "cpu_seconds": total, "finite_difference_error": maximum},
        flush=True,
    )


def main():
    """Separate bounded fit/application processes, each using one BLAS thread."""
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
