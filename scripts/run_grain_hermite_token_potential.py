#!/usr/bin/env python3
"""NEXT62: frozen-descriptor Hermite scalar program, bounded source-only fitting."""

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
from grain_research.vector_mechanisms.hermite_token_potential import (
    fit as fit_program,
)
from grain_research.vector_mechanisms.hermite_token_potential import (
    predict as apply_program,
)
from grain_research.vector_mechanisms.stage_file_guard import install
from grain_research.vector_mechanisms.token_potential import (
    hidden_gradient,
    validate_members,
)
from scripts import run_grain_token_potential_extraction as old

OUT = old.BASE / "logic_hermite_token_potential_v1"
PLAN = ROOT / "docs/GRAIN_HERMITE_TOKEN_POTENTIAL_PLAN_V1.md"
ARM = "hermite_kernel_then_mean"
CONTROL = "kernel_then_mean"
CAP = 120


def model_path(out, reference, query):
    """Resolve the only new arm at its frozen reference/query."""
    return old.model_path(out, reference, query, ARM)


def prepare(out):
    """Freeze code, inherited descriptors and old outputs before new fitting."""
    start = time.process_time()
    previous = old.load_manifest(old.OUT)
    old.frozen(old.OUT, "models")
    old.frozen(old.OUT, "predictions")
    if old.read(old.OUT / "replay_audit.json")["status"] != "PASS":
        raise ValueError("Audited NEXT60 baseline required")
    jobs = old.read(old.OUT / "private_jobs.json")
    if len(jobs) != 12 or sum(j["cohort"] == "source4" for j in jobs) != 4:
        raise ValueError("Frozen 4+8 membership required")
    out.mkdir(exist_ok=False)
    old.write(out / "private_jobs.json", jobs)
    code = [
        PLAN,
        Path(__file__),
        ROOT / "grain_research/vector_mechanisms/hermite_token_potential.py",
        ROOT / "tests/test_grain_hermite_token_potential.py",
        ROOT / "scripts/run_grain_token_potential_extraction.py",
    ]
    inherited = [old.OUT / "models/freeze.json", old.OUT / "predictions/freeze.json"]
    old.write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_HERMITE_TOKEN_POTENTIAL_V1",
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "code": previous["code"]
            | {str(p.relative_to(ROOT)): old.sha(p) for p in code},
            "upstream_hashes": previous["upstream_hashes"],
            "baseline_freezes": {
                str(p.relative_to(ROOT)): old.sha(p) for p in inherited
            },
            "jobs_sha256": old.sha(out / "private_jobs.json"),
            "prepare_cpu_seconds": time.process_time() - start,
            "max_cpu_seconds": CAP,
            "planned_fits": 10,
            "new_neural_calls": 0,
            "human_labels_read": False,
        },
    )
    print({"status": "PREPARED", "fits": 10}, flush=True)


def manifest(out):
    """Check primary code and original baseline freeze records, not target values."""
    m = old.load_manifest(out)
    old.verify_hashes(ROOT, m["baseline_freezes"])
    return m


def fit_reference(out, reference):
    """Use only allowed sources and this reference's unchanged old descriptor."""
    start = time.process_time()
    m = manifest(out)
    jobs = old.read(out / "private_jobs.json")
    src = old.source_jobs(jobs, reference)
    inputs = [
        p
        for j in src
        for k in j["query_keys"]
        for p in (old.input_path(k), old.gradient_path(k), old.target_path(j, k))
    ]
    old.check_paths(m, inputs)
    baseline = [old.model_path(old.OUT, reference, q, CONTROL) for q in old.QUERIES]
    fm = old.frozen(old.OUT, "models")
    old.verify_hashes(
        old.OUT / "models", {p.name: fm["files"][p.name] for p in baseline}
    )
    outputs = [model_path(out, reference, q) for q in old.QUERIES]
    status = out / "models" / (reference + "_status.json")
    probes = old.forbidden() + [
        p
        for j in jobs
        if j not in src
        for k in j["query_keys"]
        for p in (old.input_path(k), old.gradient_path(k), old.target_path(j, k))
    ]
    denied = install(
        inputs + baseline,
        outputs + [status],
        probes,
        [ROOT / "scripts", ROOT / "grain_research"],
    )
    diagnostics = []
    for qi, q in enumerate(old.QUERIES):
        bodies, lasts, gradients, scores = [], [], [], []
        for j in src:
            k = j["query_keys"][qi]
            _, h, last = old.pure_input(k)
            bodies.append(h)
            lasts.append(last)
            gradients.append(
                old.selected(old.gradient_path(k), ("body_gradient",))[
                    "body_gradient"
                ].astype(float)
            )
            scores.append(
                float(old.selected(old.target_path(j, k), ("score",))["score"])
            )
        base = old.arrays(baseline[qi])
        members = [j["index"] for j in src]
        validate_members(base, members)
        model, diag = fit_program(bodies, lasts, gradients, scores, base)
        validate_members(model, members)
        np.savez(outputs[qi], **model)
        diagnostics.append(
            {"reference": reference, "query_name": q, "candidate": ARM, **diag}
        )
    old.write(
        status,
        {
            "status": "COMPLETE",
            "fits": 2,
            "cpu_seconds": time.process_time() - start,
            "denial_probes": len(probes),
            "extra_denials": len(denied) - len(probes),
            "current_target_reads": 0,
            "diagnostics": diagnostics,
            "files": {p.name: old.sha(p) for p in outputs},
        },
    )


def fit(out):
    """Fit once with a first-reference estimate and a measured CPU allowance."""
    start, wall = time.process_time(), time.monotonic()
    m = manifest(out)
    jobs = old.read(out / "private_jobs.json")
    (out / "models").mkdir(exist_ok=False)
    statuses = []
    for reference in sorted({old.ref(j) for j in jobs}):
        remaining = CAP - (time.monotonic() - wall)
        if remaining <= 0:
            raise TimeoutError("Bounded source wall cap")
        subprocess.run(
            [
                sys.executable,
                __file__,
                "--stage",
                "fit-reference",
                "--reference",
                reference,
                "--output",
                str(out),
            ],
            check=True,
            cwd=ROOT,
            timeout=remaining,
        )
        statuses.append(old.read(out / "models" / (reference + "_status.json")))
        projected = (
            m["prepare_cpu_seconds"]
            + 5 * np.mean([s["cpu_seconds"] for s in statuses])
            + 30
        )
        print(
            {
                "reference": reference,
                "cpu_seconds": statuses[-1]["cpu_seconds"],
                "projected_with_30s_reserve": projected,
            },
            flush=True,
        )
        if projected > CAP:
            old.write(
                out / "bounded_stop.json",
                {
                    "reason": "Projected CPU exceeds cap",
                    "references_complete": len(statuses),
                    "projected": projected,
                },
            )
            raise TimeoutError("Source artifacts preserved; no outcome-selected refit")
    cpu = (
        time.process_time()
        - start
        + m["prepare_cpu_seconds"]
        + sum(s["cpu_seconds"] for s in statuses)
    )
    if cpu > CAP or any(s["extra_denials"] for s in statuses):
        raise ValueError("Source cost or access contract")
    old.write(
        out / "models/freeze.json",
        {
            "status": "COMPLETE",
            "fits": 10,
            "cpu_seconds": cpu,
            "current_target_reads": 0,
            "new_neural_calls": 0,
            "files": {
                p.name: old.sha(p)
                for p in sorted((out / "models").iterdir())
                if p.suffix in (".npz", ".json") and not p.name.startswith(".")
            },
        },
    )


def predict(out, folder):
    """Apply to pure current inputs; teacher files are inaccessible until evaluation."""
    start = time.process_time()
    m, fm = manifest(out), old.frozen(out, "models")
    prior = old.frozen(out, "predictions")["cpu_seconds"] if folder == "replay" else 0
    jobs = old.read(out / "private_jobs.json")
    inputs = [old.input_path(k) for j in jobs for k in j["query_keys"]]
    old.check_paths(m, inputs)
    models = [out / "models" / n for n in fm["files"] if n.endswith(".npz")]
    dst = out / folder
    dst.mkdir(exist_ok=False)
    outputs = [dst / (k + ".npz") for j in jobs for k in j["query_keys"]]
    probes = old.forbidden() + [
        p
        for j in jobs
        for k in j["query_keys"]
        for p in (old.gradient_path(k), old.target_path(j, k))
    ]
    denied = install(
        inputs + models,
        outputs + [dst / "freeze.json"],
        probes,
        [ROOT / "scripts", ROOT / "grain_research"],
    )
    for j in jobs:
        for q, k in zip(old.QUERIES, j["query_keys"], strict=True):
            data, h, last = old.pure_input(k)
            ds = old.body_directions(data)[:, data["positions"]]
            model = old.arrays(model_path(out, old.ref(j), q))
            validate_members(
                model, [s["index"] for s in old.source_jobs(jobs, old.ref(j))]
            )
            score, density = apply_program(h, last, model)
            g = hidden_gradient(density, model)
            np.savez(
                dst / (k + ".npz"),
                score=np.array(score),
                density=density,
                tau=np.einsum("th,fth->f", g, ds),
                allocation=np.einsum("th,fth->f", g, ds - ds.mean(1, keepdims=True)),
            )
    cpu = time.process_time() - start
    if cpu + fm["cpu_seconds"] + prior > CAP or len(denied) != len(probes):
        raise ValueError("Application cost/access contract")
    old.write(
        dst / "freeze.json",
        {
            "status": "COMPLETE",
            "fits": 0,
            "cpu_seconds": cpu,
            "new_neural_calls": 0,
            "current_target_reads": 0,
            "denial_probes": len(probes),
            "extra_denials": len(denied) - len(probes),
            "files": {p.name: old.sha(p) for p in outputs},
        },
    )
    print({"status": "PREDICTED", "folder": folder, "cpu_seconds": cpu}, flush=True)


def aggregate(frame, names, is_score):
    """Retain score, amplitude and allocation evidence without outcome selection."""
    rows = []
    for key, g in frame.groupby(names + ["candidate"], sort=True):
        metrics = (
            old.score_metrics(g.actual, g.predicted, g.constant)
            if is_score
            else old.fidelity(g.actual.to_numpy(), g.predicted.to_numpy())
        )
        if not is_score:
            metrics.update(
                {
                    "allocation_" + k: v
                    for k, v in old.fidelity(
                        g.actual_allocation.to_numpy(),
                        g.predicted_allocation.to_numpy(),
                    ).items()
                }
            )
        rows.append(
            dict(zip(names + ["candidate"], key, strict=True))
            | metrics
            | {"authors": g.author_key.nunique()}
        )
    return pd.DataFrame(rows)


def evaluate(out, folder):
    """After freezing, join targets and the unchanged cached NEXT60 predictions."""
    start = time.process_time()
    m = manifest(out)
    old.frozen(out, "models")
    old.frozen(out, folder)
    old.frozen(old.OUT, "predictions")
    old.frozen(old.OUT, "models")
    jobs = old.read(out / "private_jobs.json")
    old.check_paths(
        m,
        [
            p
            for j in jobs
            for k in j["query_keys"]
            for p in (old.gradient_path(k), old.target_path(j, k), old.input_path(k))
        ],
    )
    dst = out / ("analysis" if folder == "predictions" else "replay_analysis")
    dst.mkdir(exist_ok=False)
    scores, actions, fields = [], [], []
    for j in jobs:
        for q, k in zip(old.QUERIES, j["query_keys"], strict=True):
            candidate = old.arrays(out / folder / (k + ".npz"))
            prior = old.arrays(old.OUT / "predictions" / (k + ".npz"))
            base_pred = {n: a[1] for n, a in prior.items()}
            target = old.selected(old.target_path(j, k), ("score", "tau"))
            g = old.selected(old.gradient_path(k), ("body_gradient",))[
                "body_gradient"
            ].astype(float)
            data, _, _ = old.pure_input(k)
            ds = old.body_directions(data)[:, data["positions"]]
            actual_alloc = np.einsum("th,fth->f", g, ds - ds.mean(1, keepdims=True))
            np.testing.assert_allclose(
                np.einsum("th,fth->f", g, ds), target["tau"], atol=1e-5, rtol=3e-4
            )
            for arm, pred, path in (
                (CONTROL, base_pred, old.model_path(old.OUT, old.ref(j), q, CONTROL)),
                (ARM, candidate, model_path(out, old.ref(j), q)),
            ):
                model = old.arrays(path)
                meta = {
                    "author_key": j["key"],
                    "cohort": j["cohort"],
                    "query_name": q,
                    "candidate": arm,
                }
                scores.append(
                    meta
                    | {
                        "actual": float(target["score"]),
                        "predicted": float(pred["score"]),
                        "constant": float(model["origin"]),
                    }
                )
                gp = hidden_gradient(pred["density"], model)
                fields.append(
                    meta
                    | {
                        "field_error_ratio": float(
                            np.sum((gp - g) ** 2) / np.sum(g * g)
                        )
                    }
                )
                for fi in range(4):
                    actions.append(
                        meta
                        | {
                            "factor": f"whole:F{fi + 1:02d}",
                            "actual": target["tau"][fi],
                            "predicted": pred["tau"][fi],
                            "actual_allocation": actual_alloc[fi],
                            "predicted_allocation": pred["allocation"][fi],
                        }
                    )
    sd, ad, fd = map(pd.DataFrame, (scores, actions, fields))
    sc = old.paired_queries(
        sd, ["author_key", "cohort", "candidate"], ["actual", "predicted", "constant"]
    )
    ac = old.paired_queries(
        ad,
        ["author_key", "cohort", "candidate", "factor"],
        ["actual", "predicted", "actual_allocation", "predicted_allocation"],
    )
    if (len(sd), len(ad), len(fd), len(sc), len(ac)) != (48, 192, 48, 24, 96):
        raise ValueError("Complete two-arm 24-context comparison required")
    for f, n in (
        (sd, "scores"),
        (ad, "actions"),
        (fd, "fields"),
        (sc, "score_contrasts"),
        (ac, "action_contrasts"),
    ):
        f.to_csv(dst / ("private_" + n + ".csv"), index=False)
    tables = {
        n: aggregate(f, keys, score)
        for f, keys, n, score in (
            (sd, ["cohort", "query_name"], "scores", True),
            (sc, ["cohort"], "score_contrasts", True),
            (ad, ["cohort", "query_name"], "actions", False),
            (ad, ["cohort", "query_name", "factor"], "factor_actions", False),
            (ac, ["cohort"], "action_contrasts", False),
            (ac, ["cohort", "factor"], "factor_contrasts", False),
        )
    }
    tables["fields"] = fd.groupby(
        ["cohort", "query_name", "candidate"], as_index=False
    ).agg(
        field_error_ratio=("field_error_ratio", "mean"),
        authors=("author_key", "nunique"),
    )
    tables["source_diagnostics"] = pd.DataFrame(
        [
            d
            for r in sorted({old.ref(j) for j in jobs})
            for d in old.read(out / "models" / (r + "_status.json"))["diagnostics"]
        ]
    )
    for name, table in tables.items():
        table.to_csv(dst / (name + ".csv"), index=False)
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
            "better_action_than_control": bool(
                a.loc[ARM, "action_rmse"] < a.loc[CONTROL, "action_rmse"]
            ),
            "action_sse_below_zero": bool(a.loc[ARM, "action_mse_over_zero"] < 1),
            "score_sse_below_source_constant": bool(
                s.loc[ARM, "score_mse_over_source_constant"] < 1
            ),
        }
    promotion = all(all(v.values()) for v in criteria.values())
    old.write(
        dst / "summary.json",
        {
            "status": "COMPLETE",
            "promotion": promotion,
            "criteria": criteria,
            "cpu_seconds": time.process_time() - start,
            "human_labels_read": False,
            "new_neural_calls": 0,
            "files": {
                p.name: old.sha(p)
                for p in sorted(dst.glob("*.csv"))
                if not p.name.startswith(".")
            },
        },
    )
    print(
        {"status": "EVALUATED", "promotion": promotion, "criteria": criteria},
        flush=True,
    )


def audit(out):
    """Check exact replay, actual directional derivatives and independent SSEs."""
    start = time.process_time()
    manifest(out)
    freezes = [old.frozen(out, f) for f in ("models", "predictions", "replay")]
    jobs = old.read(out / "private_jobs.json")
    errors = {"normal_equation": 0.0, "direction_finite_difference": 0.0, "metric": 0.0}
    for j in jobs:
        for q, k in zip(old.QUERIES, j["query_keys"], strict=True):
            p, replay = [
                old.arrays(out / f / (k + ".npz")) for f in ("predictions", "replay")
            ]
            for n in p:
                np.testing.assert_array_equal(p[n], replay[n])
            model = old.arrays(model_path(out, old.ref(j), q))
            validate_members(
                model, [s["index"] for s in old.source_jobs(jobs, old.ref(j))]
            )
            penalty = np.eye(len(model["coef"])) * model["alpha"]
            penalty[0, 0] = 0
            errors["normal_equation"] = max(
                errors["normal_equation"],
                float(
                    np.max(
                        abs((model["gram"] + penalty) @ model["coef"] - model["rhs"])
                    )
                ),
            )
            data, h, last = old.pure_input(k)
            ds = old.body_directions(data)[:, data["positions"]]
            for fi in range(4):
                plus, minus = [
                    apply_program(h + s * 1e-5 * ds[fi], last, model)[0]
                    for s in (1, -1)
                ]
                errors["direction_finite_difference"] = max(
                    errors["direction_finite_difference"],
                    abs(float((plus - minus) / 2e-5 - p["tau"][fi])),
                )
    csvs = [p for p in (out / "analysis").glob("*.csv") if not p.name.startswith(".")]
    for p in csvs:
        if old.sha(p) != old.sha(out / "replay_analysis" / p.name):
            raise ValueError("Numerical table replay differs")
    for name in ("scores", "actions"):
        raw = pd.read_csv(out / "analysis" / ("private_" + name + ".csv"))
        summary = pd.read_csv(out / "analysis" / (name + ".csv"))
        for row in summary.to_dict("records"):
            g = raw[
                (raw.cohort == row["cohort"])
                & (raw.query_name == row["query_name"])
                & (raw.candidate == row["candidate"])
            ]
            y, pred = g.actual.to_numpy(), g.predicted.to_numpy()
            denom = (
                np.sum((y - g.constant.to_numpy()) ** 2)
                if name == "scores"
                else np.sum(y * y)
            )
            field = (
                "score_mse_over_source_constant"
                if name == "scores"
                else "action_mse_over_zero"
            )
            errors["metric"] = max(
                errors["metric"],
                abs(float(np.sum((pred - y) ** 2) / denom - row[field])),
            )
    statuses = [
        old.read(out / "models" / (r + "_status.json"))
        for r in sorted({old.ref(j) for j in jobs})
    ]
    for s in statuses:
        if s["extra_denials"]:
            raise ValueError("Unexpected source access")
        for d in s["diagnostics"]:
            if (
                d["source_objective"] > d["baseline_source_objective"] + 1e-8
                or d["capacity_energy_error"] > 1e-8
            ):
                raise ValueError("Source nesting/capacity identity")
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
        or errors["normal_equation"] > 1e-8
        or errors["metric"] > 1e-8
        or errors["direction_finite_difference"] > 1e-6
        or len(csvs) != 13
        or any(f.get("extra_denials", 0) for f in freezes)
        or sum(s["fits"] for s in statuses) != 10
    ):
        raise ValueError("Frozen audit boundary differs: " + str((total, errors)))
    old.write(
        out / "replay_audit.json",
        {
            "status": "PASS",
            "errors": errors,
            "contexts": 24,
            "direction_checks": 96,
            "new_fits": 10,
            "audit_fits": 0,
            "new_neural_calls": 0,
            "cpu_seconds": cpu,
            "cumulative_compute_cpu_seconds": total,
            "prediction_replay": "bitwise exact",
            "csvs_replayed": len(csvs),
        },
    )
    print({"status": "PASS", "cpu_seconds_total": total, "errors": errors}, flush=True)


def main():
    """Execute one bounded stage in a fresh single-BLAS-thread process."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--stage",
        required=True,
        choices=("prepare", "fit", "fit-reference", "predict", "evaluate", "audit"),
    )
    p.add_argument("--output", type=Path, default=OUT)
    p.add_argument("--reference", default="all4")
    p.add_argument("--folder", choices=("predictions", "replay"), default="predictions")
    a = p.parse_args()
    with threadpool_limits(limits=1):
        if a.stage == "fit-reference":
            fit_reference(a.output, a.reference)
        elif a.stage in ("predict", "evaluate"):
            globals()[a.stage](a.output, a.folder)
        else:
            globals()[a.stage](a.output)


if __name__ == "__main__":
    main()
