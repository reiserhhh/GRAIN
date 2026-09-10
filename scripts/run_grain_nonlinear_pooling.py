#!/usr/bin/env python3
"""NEXT69: two bounded numeric fits with frozen NEXT68 controls and input-only application."""

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
from grain_research.vector_mechanisms import nonlinear_pooling as program
from grain_research.vector_mechanisms.stage_file_guard import install
from grain_research.vector_mechanisms.token_potential import (
    hidden_gradient,
    validate_members,
)
from scripts import run_grain_coupled_pooling as prior

old = prior.old
OUT = old.BASE / "logic_nonlinear_pooling_v1"
CAP = 180


def model_path(out, q):
    """One model per query, with no extra fitted control."""
    return out / "models" / (q + ".npz")


def prepare(out):
    """Freeze the previous audited experiment and new implementation before fitting."""
    start = time.process_time()
    m = prior.manifest(prior.OUT)
    if old.read(prior.OUT / "audit.json")["status"] != "PASS":
        raise ValueError("Audited NEXT68 required")
    fm = old.frozen(prior.OUT, "models")
    old.frozen(prior.OUT, "predictions")
    jobs = old.read(prior.OUT / "private_jobs.json")
    if len(jobs) != 12 or len(prior.items(jobs)) != 48:
        raise ValueError("Frozen source4/new8 layout required")
    models = [prior.model_path(prior.OUT, q, prior.ARMS[0]) for q in old.QUERIES]
    for p in models:
        local = old.arrays(p)
        if local["landmarks"].shape != (64, 11) or len(local["coef"]) != 1169:
            raise ValueError("NEXT69 source dictionary contract differs")
    code = [
        Path(__file__),
        ROOT / "grain_research/vector_mechanisms/nonlinear_pooling.py",
        ROOT / "tests/test_grain_nonlinear_pooling.py",
        ROOT / "docs/GRAIN_NONLINEAR_POOLING_PLAN_V1.md",
    ]
    lineage = [
        prior.OUT / n
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
            "protocol": "GRAIN_NONLINEAR_POOLING_V1",
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "code": m["code"] | {str(p.relative_to(ROOT)): old.sha(p) for p in code},
            "upstream_hashes": m["upstream_hashes"]
            | {str(p.relative_to(ROOT)): fm["files"][p.name] for p in models},
            "baseline_freezes": m["baseline_freezes"]
            | {str(p.relative_to(ROOT)): old.sha(p) for p in lineage},
            "jobs_sha256": old.sha(out / "private_jobs.json"),
            "planned_fits": 2,
            "max_cpu_seconds": CAP,
            "prepare_cpu_seconds": time.process_time() - start,
        },
    )
    print({"status": "PREPARED", "planned_fits": 2}, flush=True)


def fit_query(out, query):
    """Isolate source-only fitting; deny all new8 packets and outcome tables."""
    start = time.process_time()
    m = prior.manifest(out)
    jobs = old.read(out / "private_jobs.json")
    src = [j for j in jobs if j["cohort"] == "source4"]
    qi = old.QUERIES.index(query)
    reads = [
        p
        for j in src
        for k in [j["query_keys"][qi]]
        for v in program.prior.VIEWS
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
    probes += [
        prior.OUT / "analysis" / ("private_" + n + ".csv")
        for n in ("scores", "actions", "fields")
    ]
    status = out / "models" / (query + "_status.json")
    denied = install(
        reads + [baseline],
        [model_path(out, query), status],
        probes,
        [ROOT / "scripts", ROOT / "grain_research"],
    )
    model, diag = program.fit(prior.source_records(src, qi), old.arrays(baseline))
    validate_members(model, [0, 1, 2, 3])
    np.savez(model_path(out, query), **model)
    old.write(
        status,
        {
            "status": "COMPLETE",
            "fits": 1,
            "diagnostic": diag,
            "cpu_seconds": time.process_time() - start,
            "denial_probes": len(probes),
            "extra_denials": len(denied) - len(probes),
        },
    )
    print({"status": "FITTED", "query": query, "diagnostic": diag}, flush=True)


def fit(out):
    """Two sequential query solves, keeping partial results on a hard budget stop."""
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
            raise TimeoutError("Preserve partial fits, no rescue")
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
            "fits": 2,
            "cpu_seconds": cpu,
            "files": {
                p.name: old.sha(p)
                for p in dst.iterdir()
                if p.suffix in (".npz", ".json") and not p.name.startswith(".")
            },
        },
    )


def predict(out, folder):
    """Apply source-frozen numeric models without reading current teacher outputs."""
    start = time.process_time()
    m = prior.manifest(out)
    fm = old.frozen(out, "models")
    jobs = old.read(out / "private_jobs.json")
    entries = prior.items(jobs)
    inputs = [prior.input_path(j, v, k) for j, v, q, k in entries]
    old.check_paths(m, inputs)
    models = [model_path(out, q) for q in old.QUERIES]
    dst = out / folder
    dst.mkdir(exist_ok=False)
    outputs = [dst / (k + "_" + v + ".npz") for j, v, q, k in entries]
    probes = old.forbidden() + list(
        dict.fromkeys(p for j, v, q, k in entries for p in prior.teacher_paths(j, v, k))
    )
    probes += [
        prior.OUT / "analysis" / ("private_" + n + ".csv")
        for n in ("scores", "actions", "fields")
    ]
    prevcpu = old.frozen(out, "predictions")["cpu_seconds"] if folder == "replay" else 0
    denied = install(
        inputs + models,
        outputs + [dst / "freeze.json"],
        probes,
        [ROOT / "scripts", ROOT / "grain_research"],
    )
    bank = {q: old.arrays(model_path(out, q)) for q in old.QUERIES}
    for j, v, q, k in entries:
        d = prior.pure_input(j, v, k)
        pos = d["positions"]
        b, last = d["state"][pos].astype(float), d["state"][-1].astype(float)
        ds = old.body_directions(d)[:, pos]
        score, density = program.predict(b, last, bank[q])
        g = hidden_gradient(density, bank[q])
        np.savez(
            dst / (k + "_" + v + ".npz"),
            score=np.array(score),
            density=density,
            tau=np.einsum("th,fth->f", g, ds),
            allocation=np.einsum("th,fth->f", g, ds - ds.mean(1, keepdims=True)),
        )
    cpu = time.process_time() - start
    if cpu + fm["cpu_seconds"] + prevcpu > CAP or len(denied) != len(probes):
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


def shape_account(response, positions):
    """Measure how much predicted B density lies outside the old affine-position range."""
    x = np.column_stack((np.ones(len(positions)), positions))
    projection = x @ np.linalg.lstsq(x, response, rcond=None)[0]
    return float(np.mean(response**2)), float(np.mean((response - projection) ** 2))


def evaluate(out, folder):
    """Join targets only after prediction freeze; retain all seven NEXT68 controls."""
    start = time.process_time()
    m = prior.manifest(out)
    old.frozen(out, folder)
    old.frozen(out, "models")
    jobs = old.read(out / "private_jobs.json")
    entries = prior.items(jobs)
    old.check_paths(
        m, [p for j, v, q, k in entries for p in prior.teacher_paths(j, v, k)]
    )
    summary = old.read(prior.OUT / "analysis/summary.json")
    old.verify_hashes(prior.OUT / "analysis", summary["files"])
    names = ("scores", "actions", "fields", "cross_fields", "cross_actions")
    previous = {
        n: pd.read_csv(prior.OUT / "analysis" / ("private_" + n + ".csv"))
        for n in names
    }
    rows = {n: [] for n in names}
    shapes = []
    bank = {q: old.arrays(model_path(out, q)) for q in old.QUERIES}
    for j, v, q, k in entries:
        model = bank[q]
        d = prior.pure_input(j, v, k)
        ds = old.body_directions(d)[:, d["positions"]]
        gp, yp = prior.teacher_paths(j, v, k)
        g = old.selected(gp, ("body_gradient",))["body_gradient"].astype(float)
        score = float(old.selected(yp, ("score",))["score"])
        pred = old.arrays(out / folder / (k + "_" + v + ".npz"))
        tau = np.einsum("th,fth->f", g, ds)
        allocation = np.einsum("th,fth->f", g, ds - ds.mean(1, keepdims=True))
        meta = {
            "author_key": j["key"],
            "cohort": "source4_infit" if j["cohort"] == "source4" else "new8",
            "view": v,
            "query_name": q,
            "candidate": program.ARM,
        }
        rows["scores"].append(
            meta
            | {
                "actual": score,
                "constant": float(model["origin"]),
                "predicted": float(pred["score"]),
            }
        )
        for fi in range(4):
            rows["actions"].append(
                meta
                | {
                    "factor": f"whole:F{fi + 1:02d}",
                    "actual": float(tau[fi]),
                    "predicted": float(pred["tau"][fi]),
                    "actual_allocation": float(allocation[fi]),
                    "predicted_allocation": float(pred["allocation"][fi]),
                }
            )
        estimate = hidden_gradient(pred["density"], model)
        t = len(g) * (g @ model["basis"]) * model["scale"]
        rows["fields"].append(
            meta
            | {
                "field_error_ratio": float(np.sum((estimate - g) ** 2) / np.sum(g * g)),
                "normalized_density_loss": float(
                    np.mean((pred["density"] - t) ** 2) / model["density_scale"] ** 2
                ),
            }
        )
        if v != "plus":
            continue
        minus = old.arrays(out / folder / (k + "_minus.npz"))
        gm = old.selected(prior.teacher_paths(j, "minus", k)[0], ("body_gradient",))[
            "body_gradient"
        ].astype(float)
        cut = len(g) // 2
        delta = (g[cut:] - gm[cut:]) / 2
        density = (pred["density"] - minus["density"]) / 2
        estimate = hidden_gradient(density, model)[cut:]
        t = len(g) * (delta @ model["basis"]) * model["scale"]
        cm = {"author_key": j["key"], "query_name": q, "candidate": program.ARM}
        rows["cross_fields"].append(
            cm
            | {
                "field_sse_over_zero": float(
                    np.sum((estimate - delta) ** 2) / np.sum(delta**2)
                ),
                "coordinate_error": float(np.mean((density[cut:] - t) ** 2)),
                "coordinate_energy": float(np.mean(t * t)),
                "predicted_field_norm": float(np.linalg.norm(estimate)),
                "actual_field_norm": float(np.linalg.norm(delta)),
            }
        )
        actual = np.einsum("th,fth->f", delta, ds[:, cut:])
        action = np.einsum("th,fth->f", estimate, ds[:, cut:])
        for fi in range(4):
            rows["cross_actions"].append(
                cm
                | {
                    "factor": f"whole:F{fi + 1:02d}",
                    "actual": float(actual[fi]),
                    "predicted": float(action[fi]),
                }
            )
        energy, outside = shape_account(density[cut:], np.linspace(-1, 1, len(g))[cut:])
        shapes.append(
            {
                "query_name": q,
                "prediction_energy": energy,
                "off_position_energy": outside,
            }
        )
    frames = {
        n: pd.concat((previous[n], pd.DataFrame(rows[n])), ignore_index=True)
        for n in names
    }
    sd, ad, fd, cd, ca = [frames[n] for n in names]
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
    for (q, arm), g in cd.groupby(["query_name", "candidate"], sort=True):
        a = ca[(ca.query_name == q) & (ca.candidate == arm)]
        cross_rows.append(
            {
                "query_name": q,
                "candidate": arm,
                "authors": len(g),
                "coordinate_sse_over_zero": float(
                    g.coordinate_error.mean() / g.coordinate_energy.mean()
                ),
                "full_field_relative_sse_mean": float(g.field_sse_over_zero.mean()),
                "action_sse_over_zero": float(
                    np.sum((a.predicted - a.actual) ** 2) / np.sum(a.actual**2)
                ),
                "action_r": old.fidelity(a.actual.to_numpy(), a.predicted.to_numpy())[
                    "action_r"
                ],
                "actual_action_rms": float(np.sqrt(np.mean(a.actual**2))),
                "predicted_action_rms": float(np.sqrt(np.mean(a.predicted**2))),
            }
        )
    tables["cross_response"] = pd.DataFrame(cross_rows)
    tables["source_diagnostics"] = pd.concat(
        (
            pd.read_csv(prior.OUT / "analysis/source_diagnostics.csv"),
            pd.DataFrame(
                [
                    old.read(out / "models" / (q + "_status.json"))["diagnostic"]
                    | {"query_name": q}
                    for q in old.QUERIES
                ]
            ),
        ),
        ignore_index=True,
    )
    shape = (
        pd.DataFrame(shapes)
        .groupby("query_name", as_index=False)
        .agg(
            authors=("prediction_energy", "size"),
            prediction_energy=("prediction_energy", "mean"),
            off_position_energy=("off_position_energy", "mean"),
        )
    )
    shape["off_position_fraction"] = shape.off_position_energy / shape.prediction_energy
    tables["response_shape"] = shape
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
            "cross_response": [r for r in cross_rows if r["candidate"] == program.ARM],
            "response_shape": shape.to_dict("records"),
        },
        flush=True,
    )


def audit(out):
    """Check exact replay, complete matched controls and actual same-scalar derivatives."""
    start = time.process_time()
    prior.manifest(out)
    freezes = [old.frozen(out, n) for n in ("models", "predictions", "replay")]
    jobs = old.read(out / "private_jobs.json")
    bank = {q: old.arrays(model_path(out, q)) for q in old.QUERIES}
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
        for fi in range(4):
            fd = (
                program.scalar(b + 1e-5 * ds[fi], last, bank[q])
                - program.scalar(b - 1e-5 * ds[fi], last, bank[q])
            ) / 2e-5
            maximum = max(maximum, abs(fd - pred["tau"][fi]))
            checks += 1
    summaries = [
        old.read(out / n / "summary.json") for n in ("analysis", "replay_analysis")
    ]
    for n, s in zip(("analysis", "replay_analysis"), summaries, strict=True):
        old.verify_hashes(out / n, s["files"])
    if summaries[0]["files"] != summaries[1]["files"]:
        raise ValueError("CSV replay differs")
    for n in ("scores", "actions", "fields", "cross_fields", "cross_actions"):
        current = pd.read_csv(out / "analysis" / ("private_" + n + ".csv"))
        before = pd.read_csv(prior.OUT / "analysis" / ("private_" + n + ".csv"))
        pd.testing.assert_frame_equal(
            current[current.candidate != program.ARM].reset_index(drop=True),
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
        diag = old.read(out / "models" / (q + "_status.json"))["diagnostic"]
        loss = program.objective(
            records, bank[q], program.prior.energies(records, bank[q])
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
        or checks != 192
        or len(freezes[1]["files"]) != 48
    ):
        raise ValueError(
            "Audit boundary differs: " + str((maximum, error, total, checks))
        )
    result = {
        "status": "PASS",
        "fits": 2,
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
    """Run one explicit stage in a dedicated bounded process."""
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
