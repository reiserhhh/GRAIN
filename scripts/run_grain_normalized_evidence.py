#!/usr/bin/env python3
"""NEXT70: matched mass/relative evidence readers, four bounded numeric fits."""

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
from grain_research.vector_mechanisms import normalized_evidence as program
from grain_research.vector_mechanisms.stage_file_guard import install
from grain_research.vector_mechanisms.token_potential import (
    hidden_gradient,
    validate_members,
)
from scripts import run_grain_coupled_pooling as prior
from scripts import run_grain_nonlinear_pooling as previous

old, ARMS = prior.old, program.ARMS
OUT = old.BASE / "logic_normalized_evidence_v1"
CAP = 180
NAMES = ("scores", "actions", "fields", "cross_fields", "cross_actions")


def model_path(out, q, arm):
    """One model per query/arm; earlier controls are never refitted."""
    return out / "models" / f"{q}_{arm}.npz"


def prepare(out):
    """Freeze code and audited NEXT69 lineage before fitting."""
    start = time.process_time()
    m = prior.manifest(previous.OUT)
    if old.read(previous.OUT / "audit.json")["status"] != "PASS":
        raise ValueError("Audited NEXT69 required")
    for name in ("models", "predictions"):
        old.frozen(previous.OUT, name)
    jobs = old.read(previous.OUT / "private_jobs.json")
    if len(jobs) != 12 or len(prior.items(jobs)) != 48:
        raise ValueError("Frozen source4/new8 required")
    for q in old.QUERIES:
        local = old.arrays(prior.model_path(prior.OUT, q, prior.ARMS[0]))
        if local["landmarks"].shape != (64, 11) or len(local["coef"]) != 1169:
            raise ValueError("Frozen local dictionary differs")
    code = [
        Path(__file__),
        ROOT / "grain_research/vector_mechanisms/normalized_evidence.py",
        ROOT / "tests/test_grain_normalized_evidence.py",
        ROOT / "docs/GRAIN_NORMALIZED_EVIDENCE_READER_PLAN_V1.md",
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
            "protocol": "GRAIN_NORMALIZED_EVIDENCE_V1",
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "code": m["code"] | {str(p.relative_to(ROOT)): old.sha(p) for p in code},
            "baseline_freezes": m["baseline_freezes"]
            | {str(p.relative_to(ROOT)): old.sha(p) for p in lineage},
            "jobs_sha256": old.sha(out / "private_jobs.json"),
            "planned_fits": 4,
            "max_cpu_seconds": CAP,
            "prepare_cpu_seconds": time.process_time() - start,
        },
    )
    print({"status": "PREPARED", "planned_fits": 4}, flush=True)


def fit_query(out, query):
    """Fit two source-only arms in an isolated, file-guarded process."""
    start = time.process_time()
    m = prior.manifest(out)
    jobs = old.read(out / "private_jobs.json")
    src = [j for j in jobs if j["cohort"] == "source4"]
    qi = old.QUERIES.index(query)
    reads = [
        p
        for j in src
        for k in [j["query_keys"][qi]]
        for v in prior.program.VIEWS
        for p in (prior.input_path(j, v, k), *prior.teacher_paths(j, v, k))
    ]
    baseline = prior.model_path(prior.OUT, query, prior.ARMS[0])
    old.check_paths(m, reads + [baseline])
    probes = old.forbidden() + [
        p
        for j in jobs
        if j["cohort"] == "new8"
        for k in j["query_keys"]
        for p in (old.input_path(k), old.gradient_path(k), old.target_path(j, k))
    ]
    probes += [previous.OUT / "analysis" / ("private_" + n + ".csv") for n in NAMES]
    status = out / "models" / (query + "_status.json")
    denied = install(
        reads + [baseline],
        [model_path(out, query, a) for a in ARMS] + [status],
        probes,
        [ROOT / "scripts", ROOT / "grain_research"],
    )
    models, diags = program.fit(prior.source_records(src, qi), old.arrays(baseline))
    for a, model in zip(ARMS, models, strict=True):
        validate_members(model, [0, 1, 2, 3])
        np.savez(model_path(out, query, a), **model)
    old.write(
        status,
        {
            "status": "COMPLETE",
            "fits": 2,
            "diagnostics": diags,
            "cpu_seconds": time.process_time() - start,
            "denial_probes": len(probes),
            "extra_denials": len(denied) - len(probes),
        },
    )
    print({"status": "FITTED", "query": query, "diagnostics": diags}, flush=True)


def fit(out):
    """Two query subprocesses, preserving partial artifacts on the hard budget stop."""
    start = time.process_time()
    m = prior.manifest(out)
    dst = out / "models"
    dst.mkdir(exist_ok=False)
    statuses = []
    for q in old.QUERIES:
        remaining = (
            CAP - m["prepare_cpu_seconds"] - sum(s["cpu_seconds"] for s in statuses)
        )
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__)),
                "--stage",
                "fit-query",
                "--query",
                q,
                "--output",
                str(out),
            ],
            check=True,
            timeout=max(1, remaining),
        )
        statuses.append(old.read(dst / (q + "_status.json")))
        if (
            m["prepare_cpu_seconds"]
            + 2 * np.mean([s["cpu_seconds"] for s in statuses])
            + 30
            > CAP
        ):
            old.write(
                out / "bounded_stop.json",
                {"reason": "CPU projection", "queries_complete": len(statuses)},
            )
            raise TimeoutError("Preserve partial fits; no rescue")
    cpu = (
        time.process_time()
        - start
        + m["prepare_cpu_seconds"]
        + sum(s["cpu_seconds"] for s in statuses)
    )
    if cpu > CAP or any(s["extra_denials"] for s in statuses):
        raise ValueError("Fit budget/access boundary differs")
    old.write(
        dst / "freeze.json",
        {
            "status": "COMPLETE",
            "fits": 4,
            "cpu_seconds": cpu,
            "files": {
                p.name: old.sha(p)
                for p in dst.iterdir()
                if p.suffix in (".npz", ".json") and not p.name.startswith(".")
            },
        },
    )


def predict(out, folder):
    """Freeze all two-arm predictions before any native current-target join."""
    start = time.process_time()
    m = prior.manifest(out)
    fm = old.frozen(out, "models")
    jobs = old.read(out / "private_jobs.json")
    entries = prior.items(jobs)
    inputs = [prior.input_path(j, v, k) for j, v, q, k in entries]
    old.check_paths(m, inputs)
    models = [model_path(out, q, a) for q in old.QUERIES for a in ARMS]
    dst = out / folder
    dst.mkdir(exist_ok=False)
    outputs = [dst / (k + "_" + v + ".npz") for j, v, q, k in entries]
    probes = old.forbidden() + list(
        dict.fromkeys(p for j, v, q, k in entries for p in prior.teacher_paths(j, v, k))
    )
    probes += [previous.OUT / "analysis" / ("private_" + n + ".csv") for n in NAMES]
    replay_cpu = (
        old.frozen(out, "predictions")["cpu_seconds"] if folder == "replay" else 0
    )
    denied = install(
        inputs + models,
        outputs + [dst / "freeze.json"],
        probes,
        [ROOT / "scripts", ROOT / "grain_research"],
    )
    bank = {
        (q, a): old.arrays(model_path(out, q, a)) for q in old.QUERIES for a in ARMS
    }
    for j, v, q, k in entries:
        d = prior.pure_input(j, v, k)
        pos = d["positions"]
        b, last = d["state"][pos].astype(float), d["state"][-1].astype(float)
        ds = old.body_directions(d)[:, pos]
        values = {n: [] for n in ("score", "density", "tau", "allocation")}
        for a in ARMS:
            score, density = program.predict(b, last, bank[q, a])
            g = hidden_gradient(density, bank[q, a])
            for n, value in zip(
                values,
                (
                    score,
                    density,
                    np.einsum("th,fth->f", g, ds),
                    np.einsum("th,fth->f", g, ds - ds.mean(1, keepdims=True)),
                ),
                strict=True,
            ):
                values[n].append(value)
        np.savez(
            dst / (k + "_" + v + ".npz"),
            **{n: np.asarray(value) for n, value in values.items()},
        )
    cpu = time.process_time() - start
    if cpu + fm["cpu_seconds"] + replay_cpu > CAP or len(denied) != len(probes):
        raise ValueError("Application budget/access contract differs")
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
    """Evaluate frozen predictions and preserve the eight existing control rows."""
    start = time.process_time()
    m = prior.manifest(out)
    old.frozen(out, folder)
    old.frozen(out, "models")
    jobs = old.read(out / "private_jobs.json")
    entries = prior.items(jobs)
    old.check_paths(
        m, [p for j, v, q, k in entries for p in prior.teacher_paths(j, v, k)]
    )
    summary = old.read(previous.OUT / "analysis/summary.json")
    old.verify_hashes(previous.OUT / "analysis", summary["files"])
    before = {
        n: pd.read_csv(previous.OUT / "analysis" / ("private_" + n + ".csv"))
        for n in NAMES
    }
    rows = {n: [] for n in NAMES}
    bank = {
        (q, a): old.arrays(model_path(out, q, a)) for q in old.QUERIES for a in ARMS
    }
    for j, v, q, k in entries:
        d = prior.pure_input(j, v, k)
        ds = old.body_directions(d)[:, d["positions"]]
        gp, yp = prior.teacher_paths(j, v, k)
        g = old.selected(gp, ("body_gradient",))["body_gradient"].astype(float)
        score = float(old.selected(yp, ("score",))["score"])
        pred = old.arrays(out / folder / (k + "_" + v + ".npz"))
        tau = np.einsum("th,fth->f", g, ds)
        allocation = np.einsum("th,fth->f", g, ds - ds.mean(1, keepdims=True))
        if v == "plus":
            minus = old.arrays(out / folder / (k + "_minus.npz"))
            gm = old.selected(
                prior.teacher_paths(j, "minus", k)[0], ("body_gradient",)
            )["body_gradient"].astype(float)
        for ai, a in enumerate(ARMS):
            model = bank[q, a]
            meta = {
                "author_key": j["key"],
                "cohort": "source4_infit" if j["cohort"] == "source4" else "new8",
                "view": v,
                "query_name": q,
                "candidate": a,
            }
            rows["scores"].append(
                meta
                | {
                    "actual": score,
                    "constant": float(model["origin"]),
                    "predicted": float(pred["score"][ai]),
                }
            )
            for fi in range(4):
                rows["actions"].append(
                    meta
                    | {
                        "factor": f"whole:F{fi + 1:02d}",
                        "actual": float(tau[fi]),
                        "predicted": float(pred["tau"][ai, fi]),
                        "actual_allocation": float(allocation[fi]),
                        "predicted_allocation": float(pred["allocation"][ai, fi]),
                    }
                )
            estimate = hidden_gradient(pred["density"][ai], model)
            target = len(g) * (g @ model["basis"]) * model["scale"]
            rows["fields"].append(
                meta
                | {
                    "field_error_ratio": float(
                        np.sum((estimate - g) ** 2) / np.sum(g * g)
                    ),
                    "normalized_density_loss": float(
                        np.mean((pred["density"][ai] - target) ** 2)
                        / model["density_scale"] ** 2
                    ),
                }
            )
            if v != "plus":
                continue
            cut = len(g) // 2
            delta = (g[cut:] - gm[cut:]) / 2
            density = (pred["density"][ai] - minus["density"][ai]) / 2
            estimate = hidden_gradient(density, model)[cut:]
            target = len(g) * (delta @ model["basis"]) * model["scale"]
            cm = {"author_key": j["key"], "query_name": q, "candidate": a}
            rows["cross_fields"].append(
                cm
                | {
                    "field_sse_over_zero": float(
                        np.sum((estimate - delta) ** 2) / np.sum(delta**2)
                    ),
                    "coordinate_error": float(np.mean((density[cut:] - target) ** 2)),
                    "coordinate_energy": float(np.mean(target**2)),
                    "predicted_field_norm": float(np.linalg.norm(estimate)),
                    "actual_field_norm": float(np.linalg.norm(delta)),
                }
            )
            actual, action = [
                np.einsum("th,fth->f", gg, ds[:, cut:]) for gg in (delta, estimate)
            ]
            for fi in range(4):
                rows["cross_actions"].append(
                    cm
                    | {
                        "factor": f"whole:F{fi + 1:02d}",
                        "actual": float(actual[fi]),
                        "predicted": float(action[fi]),
                    }
                )
    frames = {
        n: pd.concat((before[n], pd.DataFrame(rows[n])), ignore_index=True)
        for n in NAMES
    }
    sd, ad, fd, cd, ca = [frames[n] for n in NAMES]
    aggregate = prior.prior.previous.prior.aggregate
    tables = {
        "scores": aggregate(sd, ["cohort", "view", "query_name"], True),
        "actions": aggregate(ad, ["cohort", "view", "query_name"], False),
        "factor_actions": aggregate(
            ad, ["cohort", "view", "query_name", "factor"], False
        ),
        "fields": fd.groupby(
            ["cohort", "view", "query_name", "candidate"], as_index=False
        ).agg(
            field_error_ratio=("field_error_ratio", "mean"),
            normalized_density_loss=("normalized_density_loss", "mean"),
            authors=("author_key", "nunique"),
        ),
    }
    cross_rows = []
    for (q, a), g in cd.groupby(["query_name", "candidate"], sort=True):
        ac = ca[(ca.query_name == q) & (ca.candidate == a)]
        cross_rows.append(
            {
                "query_name": q,
                "candidate": a,
                "authors": len(g),
                "coordinate_sse_over_zero": float(
                    g.coordinate_error.mean() / g.coordinate_energy.mean()
                ),
                "full_field_relative_sse_mean": float(g.field_sse_over_zero.mean()),
                "action_sse_over_zero": float(
                    np.sum((ac.predicted - ac.actual) ** 2) / np.sum(ac.actual**2)
                ),
                "action_r": old.fidelity(ac.actual.to_numpy(), ac.predicted.to_numpy())[
                    "action_r"
                ],
                "actual_action_rms": float(np.sqrt(np.mean(ac.actual**2))),
                "predicted_action_rms": float(np.sqrt(np.mean(ac.predicted**2))),
            }
        )
    tables["cross_response"] = pd.DataFrame(cross_rows)
    tables["source_diagnostics"] = pd.concat(
        (
            pd.read_csv(previous.OUT / "analysis/source_diagnostics.csv"),
            pd.DataFrame(
                [
                    d | {"query_name": q}
                    for q in old.QUERIES
                    for d in old.read(out / "models" / (q + "_status.json"))[
                        "diagnostics"
                    ]
                ]
            ),
        ),
        ignore_index=True,
    )
    dst = out / ("analysis" if folder == "predictions" else "replay_analysis")
    dst.mkdir(exist_ok=False)
    for n, f in frames.items():
        f.to_csv(dst / ("private_" + n + ".csv"), index=False)
    for n, f in tables.items():
        f.to_csv(dst / (n + ".csv"), index=False)
    old.write(
        dst / "summary.json",
        {
            "status": "COMPLETE",
            "cpu_seconds": time.process_time() - start,
            "files": {
                p.name: old.sha(p)
                for p in dst.glob("*.csv")
                if not p.name.startswith(".")
            },
        },
    )
    print(
        {
            "status": "EVALUATED",
            "cross_response": [r for r in cross_rows if r["candidate"] in ARMS],
        },
        flush=True,
    )


def audit(out):
    """Exact packet/table replay, same-scalar gradients and independent loss reconstruction."""
    start = time.process_time()
    prior.manifest(out)
    freezes = [old.frozen(out, n) for n in ("models", "predictions", "replay")]
    jobs = old.read(out / "private_jobs.json")
    bank = {
        (q, a): old.arrays(model_path(out, q, a)) for q in old.QUERIES for a in ARMS
    }
    maximum, checks = 0.0, 0
    for j, v, q, k in prior.items(jobs):
        pred, replay = [
            old.arrays(out / n / (k + "_" + v + ".npz"))
            for n in ("predictions", "replay")
        ]
        for name in pred:
            np.testing.assert_array_equal(pred[name], replay[name])
        d = prior.pure_input(j, v, k)
        b, last = d["state"][d["positions"]].astype(float), d["state"][-1].astype(float)
        ds = old.body_directions(d)[:, d["positions"]]
        for ai, a in enumerate(ARMS):
            for fi in range(4):
                fd = (
                    program.scalar(b + 1e-5 * ds[fi], last, bank[q, a])
                    - program.scalar(b - 1e-5 * ds[fi], last, bank[q, a])
                ) / 2e-5
                maximum = max(maximum, abs(fd - pred["tau"][ai, fi]))
                checks += 1
    summaries = [
        old.read(out / n / "summary.json") for n in ("analysis", "replay_analysis")
    ]
    for n, s in zip(("analysis", "replay_analysis"), summaries, strict=True):
        old.verify_hashes(out / n, s["files"])
    if summaries[0]["files"] != summaries[1]["files"]:
        raise ValueError("CSV replay differs")
    for n in NAMES:
        current = pd.read_csv(out / "analysis" / ("private_" + n + ".csv"))
        before = pd.read_csv(previous.OUT / "analysis" / ("private_" + n + ".csv"))
        pd.testing.assert_frame_equal(
            current[~current.candidate.isin(ARMS)].reset_index(drop=True),
            before,
            check_exact=False,
            atol=1e-14,
            rtol=1e-14,
        )
    error = 0.0
    for qi, q in enumerate(old.QUERIES):
        records = prior.source_records(
            [j for j in jobs if j["cohort"] == "source4"], qi
        )
        diags = old.read(out / "models" / (q + "_status.json"))["diagnostics"]
        for a, diag in zip(ARMS, diags, strict=True):
            model = bank[q, a]
            loss = program.objective(
                records, model, program.objective_reference.energies(records, model)
            )
            error = max(
                error, float(np.max(abs(loss - [diag[n] for n in program.LOSS_NAMES])))
            )
    cpu = time.process_time() - start
    total = (
        cpu
        + sum(f["cpu_seconds"] for f in freezes)
        + sum(s["cpu_seconds"] for s in summaries)
    )
    if (
        maximum > 1e-6
        or error > 1e-8
        or total > CAP
        or checks != 384
        or len(freezes[1]["files"]) != 48
    ):
        raise ValueError(
            "Audit boundary differs: " + str((maximum, error, total, checks))
        )
    result = {
        "status": "PASS",
        "fits": 4,
        "new_neural_calls": 0,
        "prediction_packets": 48,
        "csv_replay": len(summaries[0]["files"]),
        "direction_checks": checks,
        "max_finite_difference_error": float(maximum),
        "objective_reconstruction_error": error,
        "cpu_seconds": cpu,
        "cumulative_cpu_seconds": total,
    }
    old.write(out / "audit.json", result)
    print(result, flush=True)


def main():
    """Each guarded stage runs in its own bounded process."""
    p = argparse.ArgumentParser()
    p.add_argument(
        "--stage",
        choices=("prepare", "fit", "fit-query", "predict", "evaluate", "audit"),
        required=True,
    )
    p.add_argument("--query", choices=old.QUERIES)
    p.add_argument("--folder", choices=("predictions", "replay"), default="predictions")
    p.add_argument("--output", type=Path, default=OUT)
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
