#!/usr/bin/env python3
"""NEXT68: matched source-only local/joint scalars and target-free current application."""

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
from grain_research.vector_mechanisms import coupled_pooling as program
from grain_research.vector_mechanisms.cross_token_reading import intervention
from grain_research.vector_mechanisms.stage_file_guard import install
from grain_research.vector_mechanisms.token_potential import (
    hidden_gradient,
    validate_members,
)
from scripts import run_grain_cross_token_reading as cross
from scripts import run_grain_paired_response_objective as prior

old = prior.old
OUT = old.BASE / "logic_coupled_pooling_v1"
PLAN = ROOT / "docs/GRAIN_COUPLED_POOLING_PLAN_V1.md"
CAP = 180
ARMS = program.ARMS


def model_path(out, q, arm):
    """One source-trained model per query and architecture."""
    return out / "models" / f"{q}_{arm}.npz"


def manifest(out):
    """Verify code/lineage and metadata; source fitting checks source data separately."""
    m = old.read(out / "manifest.json")
    old.verify_hashes(ROOT, m["code"])
    old.verify_hashes(ROOT, m["baseline_freezes"])
    old.verify_hashes(out, {"private_jobs.json": m["jobs_sha256"]})
    return m


def items(jobs):
    """Four source views and original-only application, with paired query membership."""
    return [
        (j, v, q, k)
        for j in jobs
        for v in (program.VIEWS if j["cohort"] == "source4" else ("original",))
        for q, k in zip(old.QUERIES, j["query_keys"], strict=True)
    ]


def input_path(j, view, key):
    """Pure upstream packet; front interventions are deterministic input-only operations."""
    return (
        prior.prior.input_path(j, view, key)
        if view == "shifted"
        else old.input_path(key)
    )


def pure_input(j, view, key):
    """Never read a native gradient/score while constructing predictor input."""
    data = old.arrays(input_path(j, view, key))
    if set(data) != {"state", "grain", "positions", "cos", "sin", "mask"}:
        raise ValueError("Pure six-field packet required")
    if view in ("plus", "minus"):
        pos = data["positions"]
        plus, minus, _ = intervention(
            data["state"], pos, old.body_directions(data)[:, pos]
        )
        data["state"] = plus if view == "plus" else minus
    return data


def teacher_paths(j, view, key):
    """Explicit original or intervened native teacher paths; never used in application."""
    if view == "original":
        return old.gradient_path(key), old.target_path(j, key)
    p = (
        prior.previous.intervention.OUT / "capture" / (key + ".npz")
        if view == "shifted"
        else cross.OUT / "capture" / (key + "_" + view + ".npz")
    )
    return p, p


def source_records(jobs, qi):
    """Prepare four matched views per source author, retaining original GRAIN coordinates."""
    rows = []
    for j in sorted(jobs, key=lambda j: j["index"]):
        k = j["query_keys"][qi]
        for view in program.VIEWS:
            d = pure_input(j, view, k)
            gp, yp = teacher_paths(j, view, k)
            rows.append(
                {
                    "author": j["index"],
                    "view": view,
                    "body": d["state"][d["positions"]].astype(float),
                    "last": d["state"][-1].astype(float),
                    "gradient": old.selected(gp, ("body_gradient",))[
                        "body_gradient"
                    ].astype(float),
                    "score": float(old.selected(yp, ("score",))["score"]),
                }
            )
    return rows


def prepare(out):
    """Freeze primary code and exact source intervention lineage before any fitting."""
    start = time.process_time()
    m = prior.manifest(prior.OUT)
    cm = cross.verify(cross.OUT)
    for path in (prior.OUT / "audit.json", cross.OUT / "audit.json"):
        if old.read(path)["status"] != "PASS":
            raise ValueError("Audited NEXT66/67 required")
    fm = old.frozen(prior.OUT, "models")
    old.frozen(prior.OUT, "predictions")
    capture = old.read(cross.OUT / "capture/status.json")
    old.verify_hashes(cross.OUT / "capture", capture["files"])
    jobs = old.read(prior.OUT / "private_jobs.json")
    if len(jobs) != 12 or sum(j["cohort"] == "source4" for j in jobs) != 4:
        raise ValueError("Frozen source4/new8 required")
    for j in [j for j in jobs if j["cohort"] == "source4"]:
        for k in j["query_keys"]:
            d, _, _ = old.pure_input(k)
            _, _, meta = intervention(
                d["state"], d["positions"], old.body_directions(d)[:, d["positions"]]
            )
            saved = old.arrays(cross.OUT / "capture" / (k + "_design.npz"))
            for name in meta:
                np.testing.assert_array_equal(meta[name], saved[name])
    out.mkdir(exist_ok=False)
    old.write(out / "private_jobs.json", jobs)
    code = [
        PLAN,
        Path(__file__),
        ROOT / "grain_research/vector_mechanisms/coupled_pooling.py",
        ROOT / "tests/test_grain_coupled_pooling.py",
        ROOT / "grain_research/vector_mechanisms/cross_token_reading.py",
        ROOT / "grain_research/vector_mechanisms/full_context_geometry.py",
    ]
    inputs = [cross.OUT / "capture" / n for n in capture["files"]]
    models = [prior.prior.model_path(prior.OUT, q) for q in old.QUERIES]
    old.verify_hashes(
        prior.OUT / "models", {p.name: fm["files"][p.name] for p in models}
    )
    lineage = [
        prior.OUT / "models/freeze.json",
        prior.OUT / "predictions/freeze.json",
        prior.OUT / "analysis/summary.json",
        prior.OUT / "audit.json",
        cross.OUT / "audit.json",
        cross.OUT / "capture/status.json",
    ]
    old.write(
        out / "manifest.json",
        {
            **m,
            "protocol": "GRAIN_COUPLED_POOLING_V1",
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "code": m["code"] | {str(p.relative_to(ROOT)): old.sha(p) for p in code},
            "upstream_hashes": m["upstream_hashes"]
            | cm["inputs"]
            | {str(p.relative_to(ROOT)): old.sha(p) for p in inputs + models},
            "baseline_freezes": m["baseline_freezes"]
            | {str(p.relative_to(ROOT)): old.sha(p) for p in lineage},
            "jobs_sha256": old.sha(out / "private_jobs.json"),
            "planned_fits": 4,
            "max_cpu_seconds": CAP,
            "prepare_cpu_seconds": time.process_time() - start,
            "new_neural_calls": 0,
            "human_labels_read": False,
        },
    )
    print({"status": "PREPARED", "planned_fits": 4, "views_per_source": 4}, flush=True)


def fit_query(out, query):
    """One isolated query process; two matched fits with all new8 inputs/targets denied."""
    start = time.process_time()
    m = manifest(out)
    jobs = old.read(out / "private_jobs.json")
    src = [j for j in jobs if j["cohort"] == "source4"]
    qi = old.QUERIES.index(query)
    reads = [
        p
        for j in src
        for k in [j["query_keys"][qi]]
        for v in program.VIEWS
        for p in (input_path(j, v, k), *teacher_paths(j, v, k))
    ]
    baseline = prior.prior.model_path(prior.OUT, query)
    old.check_paths(m, reads + [baseline])
    status = out / "models" / (query + "_status.json")
    outputs = [model_path(out, query, arm) for arm in ARMS]
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
    denied = install(
        reads + [baseline],
        outputs + [status],
        probes,
        [ROOT / "scripts", ROOT / "grain_research"],
    )
    models, diagnostics = program.fit(source_records(src, qi), old.arrays(baseline))
    for arm, model in zip(ARMS, models, strict=True):
        validate_members(model, [0, 1, 2, 3])
        np.savez(model_path(out, query, arm), **model)
    old.write(
        status,
        {
            "status": "COMPLETE",
            "fits": 2,
            "diagnostics": diagnostics,
            "cpu_seconds": time.process_time() - start,
            "denial_probes": len(probes),
            "extra_denials": len(denied) - len(probes),
        },
    )
    print({"status": "FITTED", "query": query, "diagnostics": diagnostics}, flush=True)


def fit(out):
    """Four total solves; stop on projected budget, without restarting completed fits."""
    start = time.process_time()
    m = manifest(out)
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
            raise TimeoutError("Preserve partial fits; no automatic rescue")
    cpu = (
        time.process_time()
        - start
        + m["prepare_cpu_seconds"]
        + sum(s["cpu_seconds"] for s in statuses)
    )
    if cpu > CAP or any(s["extra_denials"] for s in statuses):
        raise ValueError("Fit budget/access contract differs")
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
    """Apply both frozen scalars on pure input only, saving all outputs before evaluation."""
    start = time.process_time()
    m = manifest(out)
    fm = old.frozen(out, "models")
    jobs = old.read(out / "private_jobs.json")
    entries = items(jobs)
    inputs = [input_path(j, v, k) for j, v, q, k in entries]
    old.check_paths(m, inputs)
    models = [model_path(out, q, a) for q in old.QUERIES for a in ARMS]
    dst = out / folder
    dst.mkdir(exist_ok=False)
    outputs = [dst / (k + "_" + v + ".npz") for j, v, q, k in entries]
    probes = old.forbidden() + list(
        dict.fromkeys(p for j, v, q, k in entries for p in teacher_paths(j, v, k))
    )
    probes += [
        prior.OUT / "analysis" / ("private_" + n + ".csv")
        for n in ("scores", "actions", "fields")
    ]
    priorcpu = (
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
    for j, view, q, k in entries:
        data = pure_input(j, view, k)
        pos = data["positions"]
        b, last = data["state"][pos].astype(float), data["state"][-1].astype(float)
        ds = old.body_directions(data)[:, pos]
        rows = []
        for arm in ARMS:
            model = bank[q, arm]
            score, density = program.predict(b, last, model)
            g = hidden_gradient(density, model)
            rows.append(
                {
                    "score": score,
                    "density": density,
                    "tau": np.einsum("th,fth->f", g, ds),
                    "allocation": np.einsum(
                        "th,fth->f", g, ds - ds.mean(1, keepdims=True)
                    ),
                }
            )
        np.savez(
            dst / (k + "_" + view + ".npz"),
            **{n: np.stack([r[n] for r in rows]) for n in rows[0]},
        )
    cpu = time.process_time() - start
    if cpu + fm["cpu_seconds"] + priorcpu > CAP or len(denied) != len(probes):
        raise ValueError("Application budget/access boundary differs")
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
    """Join frozen predictions with teachers; retain all original comparison rows."""
    start = time.process_time()
    m = manifest(out)
    old.frozen(out, folder)
    old.frozen(out, "models")
    jobs = old.read(out / "private_jobs.json")
    entries = items(jobs)
    old.check_paths(m, [p for j, v, q, k in entries for p in teacher_paths(j, v, k)])
    ps = old.read(prior.OUT / "analysis/summary.json")
    old.verify_hashes(prior.OUT / "analysis", ps["files"])
    originals = {
        n: pd.read_csv(prior.OUT / "analysis" / ("private_" + n + ".csv"))
        for n in ("scores", "actions", "fields")
    }
    new = {n: [] for n in originals}
    cross_rows, cross_actions = [], []
    bank = {
        (q, a): old.arrays(model_path(out, q, a)) for q in old.QUERIES for a in ARMS
    }
    for j, view, q, k in entries:
        data = pure_input(j, view, k)
        ds = old.body_directions(data)[:, data["positions"]]
        gp, yp = teacher_paths(j, view, k)
        g = old.selected(gp, ("body_gradient",))["body_gradient"].astype(float)
        score = float(old.selected(yp, ("score",))["score"])
        tau = np.einsum("th,fth->f", g, ds)
        allocation = np.einsum("th,fth->f", g, ds - ds.mean(1, keepdims=True))
        pred = old.arrays(out / folder / (k + "_" + view + ".npz"))
        for ai, arm in enumerate(ARMS):
            model = bank[q, arm]
            meta = {
                "author_key": j["key"],
                "cohort": "source4_infit" if j["cohort"] == "source4" else "new8",
                "view": view,
                "query_name": q,
                "candidate": arm,
            }
            new["scores"].append(
                meta
                | {
                    "actual": score,
                    "constant": float(model["origin"]),
                    "predicted": float(pred["score"][ai]),
                }
            )
            for fi in range(4):
                new["actions"].append(
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
            t = len(g) * (g @ model["basis"]) * model["scale"]
            new["fields"].append(
                meta
                | {
                    "field_error_ratio": float(
                        np.sum((estimate - g) ** 2) / np.sum(g * g)
                    ),
                    "normalized_density_loss": float(
                        np.mean((pred["density"][ai] - t) ** 2)
                        / model["density_scale"] ** 2
                    ),
                }
            )
        if view != "plus":
            continue
        minus = old.arrays(out / folder / (k + "_minus.npz"))
        gm = old.selected(teacher_paths(j, "minus", k)[0], ("body_gradient",))[
            "body_gradient"
        ].astype(float)
        cut = len(g) // 2
        delta = (g[cut:] - gm[cut:]) / 2
        actual = np.einsum("th,fth->f", delta, ds[:, cut:])
        for ai, arm in enumerate(ARMS):
            model = bank[q, arm]
            density = (pred["density"][ai] - minus["density"][ai]) / 2
            estimate = hidden_gradient(density, model)[cut:]
            t = len(g) * (delta @ model["basis"]) * model["scale"]
            meta = {"author_key": j["key"], "query_name": q, "candidate": arm}
            cross_rows.append(
                meta
                | {
                    "field_sse_over_zero": float(
                        np.sum((estimate - delta) ** 2) / np.sum(delta * delta)
                    ),
                    "coordinate_error": float(np.mean((density[cut:] - t) ** 2)),
                    "coordinate_energy": float(np.mean(t * t)),
                    "predicted_field_norm": float(np.linalg.norm(estimate)),
                    "actual_field_norm": float(np.linalg.norm(delta)),
                }
            )
            action = np.einsum("th,fth->f", estimate, ds[:, cut:])
            for fi in range(4):
                cross_actions.append(
                    meta
                    | {
                        "factor": f"whole:F{fi + 1:02d}",
                        "actual": float(actual[fi]),
                        "predicted": float(action[fi]),
                    }
                )
    sd, ad, fd = [
        pd.concat((originals[n], pd.DataFrame(new[n])), ignore_index=True)
        for n in ("scores", "actions", "fields")
    ]
    aggregate = prior.previous.prior.aggregate
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
    cd, ca = pd.DataFrame(cross_rows), pd.DataFrame(cross_actions)
    rows = []
    for (q, arm), g in cd.groupby(["query_name", "candidate"], sort=True):
        a = ca[(ca.query_name == q) & (ca.candidate == arm)]
        rows.append(
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
    tables["cross_response"] = pd.DataFrame(rows)
    tables["source_diagnostics"] = pd.DataFrame(
        [
            d | {"query_name": q}
            for q in old.QUERIES
            for d in old.read(out / "models" / (q + "_status.json"))["diagnostics"]
        ]
    )
    dst = out / ("analysis" if folder == "predictions" else "replay_analysis")
    dst.mkdir(exist_ok=False)
    for n, f in (
        ("scores", sd),
        ("actions", ad),
        ("fields", fd),
        ("cross_fields", cd),
        ("cross_actions", ca),
    ):
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
    print({"status": "EVALUATED", "cross_response": rows}, flush=True)


def audit(out):
    """Verify replay, actual directional derivatives, matched objectives and retained controls."""
    start = time.process_time()
    manifest(out)
    freezes = [old.frozen(out, n) for n in ("models", "predictions", "replay")]
    jobs = old.read(out / "private_jobs.json")
    bank = {
        (q, a): old.arrays(model_path(out, q, a)) for q in old.QUERIES for a in ARMS
    }
    maximum, checks = 0.0, 0
    for j, view, q, k in items(jobs):
        pred, replay = [
            old.arrays(out / n / (k + "_" + view + ".npz"))
            for n in ("predictions", "replay")
        ]
        for name in pred:
            np.testing.assert_array_equal(pred[name], replay[name])
        d = pure_input(j, view, k)
        b, last = d["state"][d["positions"]].astype(float), d["state"][-1].astype(float)
        ds = old.body_directions(d)[:, d["positions"]]
        for ai, arm in enumerate(ARMS):
            for fi in range(4):
                val = (
                    program.scalar(b + 1e-5 * ds[fi], last, bank[q, arm])
                    - program.scalar(b - 1e-5 * ds[fi], last, bank[q, arm])
                ) / 2e-5
                maximum = max(maximum, abs(val - pred["tau"][ai, fi]))
                checks += 1
    summaries = [
        old.read(out / n / "summary.json") for n in ("analysis", "replay_analysis")
    ]
    for name, s in zip(("analysis", "replay_analysis"), summaries, strict=True):
        old.verify_hashes(out / name, s["files"])
    if summaries[0]["files"] != summaries[1]["files"]:
        raise ValueError("CSV replay mismatch")
    for name in ("scores", "actions", "fields"):
        current = pd.read_csv(out / "analysis" / ("private_" + name + ".csv"))
        before = pd.read_csv(prior.OUT / "analysis" / ("private_" + name + ".csv"))
        pd.testing.assert_frame_equal(
            current[~current.candidate.isin(ARMS)].reset_index(drop=True),
            before,
            check_exact=False,
            atol=1e-14,
            rtol=1e-14,
        )
    error = 0.0
    src = [j for j in jobs if j["cohort"] == "source4"]
    for qi, q in enumerate(old.QUERIES):
        records = source_records(src, qi)
        status = old.read(out / "models" / (q + "_status.json"))
        for arm, diag in zip(ARMS, status["diagnostics"], strict=True):
            m = bank[q, arm]
            loss = program.objective(records, m, program.energies(records, m))
            expected = [
                diag[n]
                for n in (
                    "source_score_loss",
                    "source_density_loss",
                    "source_last_loss",
                    "source_cross_loss",
                    "source_penalty",
                )
            ]
            error = max(error, float(np.max(abs(loss - expected))))
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
        or len(freezes[1]["files"]) != 48
        or checks != 384
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
    """Run each fitting/application stage as a bounded explicit process."""
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
