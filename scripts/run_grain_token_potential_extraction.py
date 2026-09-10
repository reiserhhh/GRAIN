#!/usr/bin/env python3
"""NEXT60: cache-only token potential, isolated source fitting and target-free use."""

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
from grain_research.vector_mechanisms.token_potential import (
    ARMS,
    fit_potential,
    hidden_gradient,
    predict_potential,
    source_descriptor,
    validate_members,
)
from scripts.run_grain_complete_function_extraction import (
    BASE,
    QUERIES,
    arrays,
    body_directions,
    fidelity,
    forbidden,
    frozen,
    paired_queries,
    read,
    ref,
    score_metrics,
    selected,
    sha,
    source_jobs,
    verify_hashes,
    write,
)
from scripts.run_grain_pooled_feature_null_response import NATIVE, target_path

OLD = BASE / "logic_complete_function_extraction_v1"
PREVIOUS = BASE / "logic_pooled_feature_null_response_v1"
PLAN = ROOT / "docs/GRAIN_TOKEN_POTENTIAL_EXTRACTION_PLAN_V1.md"
OUT = BASE / "logic_token_potential_extraction_v1"


def model_path(out, reference, query, arm):
    """Address one frozen source/query/arm, never a current-author fitted model."""
    return out / "models" / f"{reference}_{query}_{arm}.npz"


def load_manifest(out):
    """Verify code and job metadata without opening teacher caches."""
    m = read(out / "manifest.json")
    verify_hashes(ROOT, m["code"])
    verify_hashes(out, {"private_jobs.json": m["jobs_sha256"]})
    return m


def check_paths(m, paths):
    """Verify only the data allowlist of the particular stage/reference."""
    for p in paths:
        key = str(p.relative_to(ROOT))
        if sha(p) != m["upstream_hashes"][key]:
            raise ValueError("Frozen source/input hash differs: " + key)


def input_path(key):
    """Resolve the previously frozen pure six-field H16/GRAIN/layout packet."""
    return OLD / "private_inputs" / (key + ".npz")


def gradient_path(key):
    """Use the original H16 BODY gradient, not a later endpoint gradient."""
    return NATIVE / "private_packets" / (key + ".npz")


def pure_input(key):
    """Enforce no teacher fields and a BODY-only perturbation with separate last."""
    d = arrays(input_path(key))
    if set(d) != {"state", "grain", "positions", "cos", "sin", "mask"}:
        raise ValueError("Pure input packet required")
    pos = d["positions"]
    if np.any(np.diff(pos) <= 0) or pos[-1] >= len(d["state"]) - 1:
        raise ValueError("Ordered non-final BODY positions required")
    return d, d["state"][pos].astype(float), d["state"][-1].astype(float)


def prepare(out):
    """Reuse audited H16 bindings and freeze the new protocol before fitting."""
    start = time.process_time()
    old = read(PREVIOUS / "manifest.json")
    verify_hashes(ROOT, old["inputs"])
    if read(PREVIOUS / "replay_audit.json")["status"] != "PASS":
        raise ValueError("Prior native binding audit is required")
    jobs = read(OLD / "private_jobs.json")
    if len(jobs) != 12 or sum(j["cohort"] == "source4" for j in jobs) != 4:
        raise ValueError("Frozen 4+8 source/application inventory required")
    out.mkdir(exist_ok=False)
    write(out / "private_jobs.json", jobs)
    code = [
        PLAN,
        Path(__file__),
        ROOT / "grain_research/vector_mechanisms/token_potential.py",
        ROOT / "grain_research/vector_mechanisms/position_reader.py",
        ROOT / "grain_research/vector_mechanisms/stage_file_guard.py",
        ROOT / "tests/test_grain_token_potential.py",
        PREVIOUS / "manifest.json",
        PREVIOUS / "replay_audit.json",
    ]
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_TOKEN_POTENTIAL_EXTRACTION_V1",
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "code": {str(p.relative_to(ROOT)): sha(p) for p in code},
            "jobs_sha256": sha(out / "private_jobs.json"),
            "upstream_hashes": old["inputs"],
            "max_cpu_seconds": 180,
            "prepare_cpu_seconds": time.process_time() - start,
            "alpha": 0.01,
            "rank": 8,
            "landmarks_per_author": 16,
            "new_neural_calls": 0,
            "human_labels_read": False,
        },
    )
    print({"status": "PREPARED", "contexts": 24}, flush=True)


def fit_reference(out, reference):
    """Fit four models with all excluded-source/current target paths denied."""
    start = time.process_time()
    m = load_manifest(out)
    jobs = read(out / "private_jobs.json")
    src = source_jobs(jobs, reference)
    reads = [
        p
        for j in src
        for k in j["query_keys"]
        for p in (input_path(k), gradient_path(k), target_path(j, k))
    ]
    check_paths(m, reads)
    outputs = [model_path(out, reference, q, a) for q in QUERIES for a in ARMS]
    status_path = out / "models" / (reference + "_status.json")
    excluded = [j for j in jobs if j not in src]
    probes = forbidden() + [
        p
        for j in excluded
        for k in j["query_keys"]
        for p in (input_path(k), gradient_path(k), target_path(j, k))
    ]
    denied = install(
        reads,
        outputs + [status_path],
        probes,
        [ROOT / "scripts", ROOT / "grain_research"],
    )
    diagnostics = []
    for qi, q in enumerate(QUERIES):
        bodies, lasts, gradients, scores = [], [], [], []
        for j in src:
            k = j["query_keys"][qi]
            _, h, last = pure_input(k)
            g = selected(gradient_path(k), ("body_gradient",))["body_gradient"].astype(
                float
            )
            bodies.append(h)
            lasts.append(last)
            gradients.append(g)
            scores.append(float(selected(target_path(j, k), ("score",))["score"]))
        ids = [j["index"] for j in src]
        descriptor = source_descriptor(
            bodies, gradients, ids, rank=m["rank"], landmarks=m["landmarks_per_author"]
        )
        for arm in ARMS:
            model, diag = fit_potential(
                bodies, lasts, gradients, scores, descriptor, arm, m["alpha"]
            )
            validate_members(model, ids)
            np.savez(model_path(out, reference, q, arm), **model)
            diagnostics.append(
                {"reference": reference, "query_name": q, "candidate": arm, **diag}
            )
    write(
        status_path,
        {
            "status": "COMPLETE",
            "fits": 4,
            "cpu_seconds": time.process_time() - start,
            "denial_probes": len(probes),
            "extra_denials": len(denied) - len(probes),
            "current_target_reads": 0,
            "diagnostics": diagnostics,
            "files": {p.name: sha(p) for p in outputs},
        },
    )


def fit(out):
    """Fit once and stop after the first complete reference if the cost is excessive."""
    start, wall = time.process_time(), time.monotonic()
    m = load_manifest(out)
    jobs = read(out / "private_jobs.json")
    (out / "models").mkdir(exist_ok=False)
    statuses = []
    for reference in sorted({ref(j) for j in jobs}):
        remaining = 180 - (time.monotonic() - wall)
        if remaining <= 0:
            raise TimeoutError("Fit wall budget exhausted")
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
        statuses.append(read(out / "models" / (reference + "_status.json")))
        projected = (
            m["prepare_cpu_seconds"]
            + 5 * np.mean([s["cpu_seconds"] for s in statuses])
            + 30
        )
        print(
            {
                "reference": reference,
                "cpu_seconds": statuses[-1]["cpu_seconds"],
                "projected_with_30s_application_reserve": projected,
            },
            flush=True,
        )
        if projected > 180:
            write(
                out / "bounded_stop.json",
                {
                    "reason": "Projected CPU exceeds frozen cap",
                    "references_complete": len(statuses),
                    "projected": projected,
                },
            )
            raise TimeoutError(
                "Preserved source reference; no smaller outcome-selected refit"
            )
    cpu = (
        time.process_time()
        - start
        + m["prepare_cpu_seconds"]
        + sum(s["cpu_seconds"] for s in statuses)
    )
    if cpu > 180 or any(s["extra_denials"] for s in statuses):
        raise RuntimeError("Source budget/access boundary differs")
    write(
        out / "models/freeze.json",
        {
            "status": "COMPLETE",
            "fits": 20,
            "cpu_seconds": cpu,
            "current_target_reads": 0,
            "new_neural_calls": 0,
            "files": {
                p.name: sha(p)
                for p in sorted((out / "models").iterdir())
                if p.suffix in (".npz", ".json") and not p.name.startswith(".")
            },
        },
    )


def predict(out, folder):
    """Apply frozen scalar models using current H16/GRAIN only, before target joins."""
    start = time.process_time()
    m, fm = load_manifest(out), frozen(out, "models")
    prior = frozen(out, "predictions")["cpu_seconds"] if folder == "replay" else 0
    jobs = read(out / "private_jobs.json")
    inputs = [input_path(k) for j in jobs for k in j["query_keys"]]
    check_paths(m, inputs)
    models = [out / "models" / n for n in fm["files"] if n.endswith(".npz")]
    dst = out / folder
    dst.mkdir(exist_ok=False)
    outputs = [dst / (k + ".npz") for j in jobs for k in j["query_keys"]]
    probes = forbidden() + [
        p
        for j in jobs
        for k in j["query_keys"]
        for p in (gradient_path(k), target_path(j, k))
    ]
    denied = install(
        inputs + models,
        outputs + [dst / "freeze.json"],
        probes,
        [ROOT / "scripts", ROOT / "grain_research"],
    )
    for j in jobs:
        members = [s["index"] for s in source_jobs(jobs, ref(j))]
        for q, k in zip(QUERIES, j["query_keys"], strict=True):
            data, h, last = pure_input(k)
            ds = body_directions(data)[:, data["positions"]]
            allocation = ds - ds.mean(axis=1, keepdims=True)
            rows = []
            for arm in ARMS:
                model = arrays(model_path(out, ref(j), q, arm))
                validate_members(model, members)
                score, density = predict_potential(h, last, model)
                g = hidden_gradient(density, model)
                rows.append(
                    (
                        score,
                        density,
                        np.einsum("th,fth->f", g, ds),
                        np.einsum("th,fth->f", g, allocation),
                    )
                )
            np.savez(
                dst / (k + ".npz"),
                score=np.array([r[0] for r in rows]),
                density=np.array([r[1] for r in rows]),
                tau=np.array([r[2] for r in rows]),
                allocation=np.array([r[3] for r in rows]),
            )
    cpu = time.process_time() - start
    # Replay cumulative budget is also checked by the separate final audit.
    if cpu + fm["cpu_seconds"] + prior > 180 or len(denied) != len(probes):
        raise RuntimeError("Prediction budget/access boundary differs")
    write(
        dst / "freeze.json",
        {
            "status": "COMPLETE",
            "fits": 0,
            "cpu_seconds": cpu,
            "new_neural_calls": 0,
            "current_target_reads": 0,
            "denial_probes": len(probes),
            "extra_denials": len(denied) - len(probes),
            "files": {p.name: sha(p) for p in outputs},
        },
    )
    print(
        {"status": "PREDICTED", "folder": folder, "contexts": 24, "cpu_seconds": cpu},
        flush=True,
    )


def evaluate(out, folder):
    """Join frozen predictions to original targets; oracle projection is evaluation only."""
    start = time.process_time()
    m = load_manifest(out)
    frozen(out, "models")
    frozen(out, folder)
    jobs = read(out / "private_jobs.json")
    check_paths(
        m,
        [
            p
            for j in jobs
            for k in j["query_keys"]
            for p in (gradient_path(k), target_path(j, k), input_path(k))
        ],
    )
    dst = out / ("analysis" if folder == "predictions" else "replay_analysis")
    dst.mkdir(exist_ok=False)
    scores, actions, fields = [], [], []
    for j in jobs:
        for q, key in zip(QUERIES, j["query_keys"], strict=True):
            p = arrays(out / folder / (key + ".npz"))
            target = selected(target_path(j, key), ("score", "tau"))
            g = selected(gradient_path(key), ("body_gradient",))[
                "body_gradient"
            ].astype(float)
            data, _, _ = pure_input(key)
            ds = body_directions(data)[:, data["positions"]]
            allocation = ds - ds.mean(axis=1, keepdims=True)
            native_alloc = np.einsum("th,fth->f", g, allocation)
            np.testing.assert_allclose(
                np.einsum("th,fth->f", g, ds), target["tau"], atol=1e-5, rtol=3e-4
            )
            for ai, arm in enumerate((*ARMS, "oracle_source_U_evaluation_only")):
                model = arrays(model_path(out, ref(j), q, ARMS[min(ai, 1)]))
                meta = {
                    "author_key": j["key"],
                    "cohort": j["cohort"],
                    "query_name": q,
                    "candidate": arm,
                }
                if ai < 2:
                    gp = hidden_gradient(p["density"][ai], model)
                    tau, alloc = p["tau"][ai], p["allocation"][ai]
                    scores.append(
                        meta
                        | {
                            "actual": float(target["score"]),
                            "predicted": p["score"][ai],
                            "constant": float(model["origin"]),
                        }
                    )
                else:
                    gp = (g @ model["basis"]) @ model["basis"].T
                    tau = np.einsum("th,fth->f", gp, ds)
                    alloc = np.einsum("th,fth->f", gp, allocation)
                fields.append(
                    meta | {"field_error_ratio": np.sum((gp - g) ** 2) / np.sum(g * g)}
                )
                for fi in range(4):
                    actions.append(
                        meta
                        | {
                            "factor": f"whole:F{fi + 1:02d}",
                            "actual": target["tau"][fi],
                            "predicted": tau[fi],
                            "actual_allocation": native_alloc[fi],
                            "predicted_allocation": alloc[fi],
                        }
                    )
    sd, ad, fd = map(pd.DataFrame, (scores, actions, fields))
    sc = paired_queries(
        sd, ["author_key", "cohort", "candidate"], ["actual", "predicted", "constant"]
    )
    ac = paired_queries(
        ad,
        ["author_key", "cohort", "candidate", "factor"],
        ["actual", "predicted", "actual_allocation", "predicted_allocation"],
    )
    if (len(sd), len(ad), len(sc), len(ac)) != (48, 288, 24, 144):
        raise ValueError("Expected complete 24-context comparison")
    tables = {}
    for frame, n in (
        (sd, "scores"),
        (ad, "actions"),
        (fd, "fields"),
        (sc, "score_contrasts"),
        (ac, "action_contrasts"),
    ):
        frame.to_csv(dst / ("private_" + n + ".csv"), index=False)
    for frame, names, label, is_score in (
        (sd, ["cohort", "query_name"], "scores", True),
        (sc, ["cohort"], "score_contrasts", True),
        (ad, ["cohort", "query_name"], "actions", False),
        (ad, ["cohort", "query_name", "factor"], "factor_actions", False),
        (ac, ["cohort"], "action_contrasts", False),
        (ac, ["cohort", "factor"], "factor_contrasts", False),
    ):
        rows = []
        for key, group in frame.groupby(names + ["candidate"], sort=True):
            metrics = (
                score_metrics(group.actual, group.predicted, group.constant)
                if is_score
                else fidelity(group.actual.to_numpy(), group.predicted.to_numpy())
            )
            if not is_score:
                metrics.update(
                    {
                        "allocation_" + k: v
                        for k, v in fidelity(
                            group.actual_allocation.to_numpy(),
                            group.predicted_allocation.to_numpy(),
                        ).items()
                    }
                )
            rows.append(
                dict(zip(names + ["candidate"], key, strict=True))
                | metrics
                | {"authors": group.author_key.nunique()}
            )
        tables[label] = pd.DataFrame(rows)
    tables["fields"] = fd.groupby(
        ["cohort", "query_name", "candidate"], as_index=False
    ).agg(
        field_error_ratio=("field_error_ratio", "mean"),
        authors=("author_key", "nunique"),
    )
    tables["source_diagnostics"] = pd.DataFrame(
        [
            r
            for reference in sorted({ref(j) for j in jobs})
            for r in read(out / "models" / (reference + "_status.json"))["diagnostics"]
        ]
    )
    for n, table in tables.items():
        table.to_csv(dst / (n + ".csv"), index=False)
    criteria = {}
    for q in QUERIES:
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
                a.loc[ARMS[1], "action_rmse"] < a.loc[ARMS[0], "action_rmse"]
            ),
            "action_sse_below_zero": bool(a.loc[ARMS[1], "action_mse_over_zero"] < 1),
            "score_sse_below_source_constant": bool(
                s.loc[ARMS[1], "score_mse_over_source_constant"] < 1
            ),
        }
    promoted = all(all(v.values()) for v in criteria.values())
    write(
        dst / "summary.json",
        {
            "status": "COMPLETE",
            "promotion": promoted,
            "criteria": criteria,
            "cpu_seconds": time.process_time() - start,
            "human_labels_read": False,
            "new_neural_calls": 0,
            "files": {
                p.name: sha(p)
                for p in sorted(dst.glob("*.csv"))
                if not p.name.startswith(".")
            },
        },
    )
    print(
        {"status": "EVALUATED", "promotion": promoted, "criteria": criteria}, flush=True
    )


def audit(out):
    """Check numerical replay, equations, source exclusion and independent metrics."""
    start = time.process_time()
    load_manifest(out)
    fm, fp, fr = [frozen(out, name) for name in ("models", "predictions", "replay")]
    jobs = read(out / "private_jobs.json")
    errors = {"normal_equation": 0.0, "direction_finite_difference": 0.0, "metric": 0.0}
    for j in jobs:
        for q, key in zip(QUERIES, j["query_keys"], strict=True):
            p, rp = [
                arrays(out / folder / (key + ".npz"))
                for folder in ("predictions", "replay")
            ]
            for name in p:
                np.testing.assert_array_equal(p[name], rp[name])
            data, h, last = pure_input(key)
            ds = body_directions(data)[:, data["positions"]]
            for ai, arm in enumerate(ARMS):
                model = arrays(model_path(out, ref(j), q, arm))
                validate_members(model, [s["index"] for s in source_jobs(jobs, ref(j))])
                penalty = np.eye(len(model["coef"])) * model["alpha"]
                penalty[0, 0] = 0
                errors["normal_equation"] = max(
                    errors["normal_equation"],
                    float(
                        np.max(
                            abs(
                                (model["gram"] + penalty) @ model["coef"] - model["rhs"]
                            )
                        )
                    ),
                )
                for fi in range(4):
                    eps = 1e-5
                    plus, minus = [
                        predict_potential(h + sign * eps * ds[fi], last, model)[0]
                        for sign in (1, -1)
                    ]
                    errors["direction_finite_difference"] = max(
                        errors["direction_finite_difference"],
                        abs(float((plus - minus) / (2 * eps) - p["tau"][ai, fi])),
                    )
    for p in (out / "analysis").glob("*.csv"):
        if not p.name.startswith(".") and sha(p) != sha(
            out / "replay_analysis" / p.name
        ):
            raise ValueError("CSV replay mismatch: " + p.name)
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
            denominator = (
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
                abs(float(np.sum((pred - y) ** 2) / denominator - row[field])),
            )
    cpu = time.process_time() - start
    total = (
        sum(f["cpu_seconds"] for f in (fm, fp, fr))
        + cpu
        + sum(
            read(out / n / "summary.json")["cpu_seconds"]
            for n in ("analysis", "replay_analysis")
        )
    )
    if (
        total > 180
        or errors["normal_equation"] > 1e-8
        or errors["metric"] > 1e-8
        or errors["direction_finite_difference"] > 1e-6
    ):
        raise ValueError(
            "Frozen audit/budget threshold exceeded: " + str((total, errors))
        )
    statuses = [
        read(out / "models" / (r + "_status.json"))
        for r in sorted({ref(j) for j in jobs})
    ]
    if (
        sum(s["fits"] for s in statuses) != 20
        or any(s["extra_denials"] for s in statuses)
        or fp["extra_denials"]
        or fr["extra_denials"]
    ):
        raise ValueError("Stage access/fitting boundary differs")
    write(
        out / "replay_audit.json",
        {
            "status": "PASS",
            "errors": errors,
            "contexts": 24,
            "fits": 0,
            "new_neural_calls": 0,
            "cpu_seconds": cpu,
            "cumulative_compute_cpu_seconds": total,
            "prediction_replay": "bitwise exact",
            "csvs_replayed": len(list((out / "analysis").glob("*.csv"))),
        },
    )
    print({"status": "PASS", "cpu_seconds_total": total, "errors": errors})


def main():
    """Run one explicitly selected stage in a fresh process with one BLAS thread."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--stage",
        choices=("prepare", "fit", "fit-reference", "predict", "evaluate", "audit"),
        required=True,
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
