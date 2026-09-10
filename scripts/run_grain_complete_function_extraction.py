#!/usr/bin/env python3
"""NEXT57: source-only full scalar extraction with no current suffix or teacher."""

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
from grain_research.vector_mechanisms.complete_function_extraction import (
    OBJECTIVES,
    apply_scalar,
    features,
    fit_scalar,
)
from grain_research.vector_mechanisms.stage_file_guard import install
from scripts.run_grain_source_state_transport import selected
from scripts.run_grain_whole_path_assembly import (
    BASE,
    ENDPOINT,
    QUERIES,
    arrays,
    body_directions,
    error_account,
    fidelity,
    forbidden,
    frozen,
    paired_queries,
    read,
    ref,
    score_metrics,
    sha,
    source_set,
    verify,
    verify_hashes,
    write,
)

OLD = BASE / "logic_source_offset_mechanism_v1"
PLAN = ROOT / "docs/GRAIN_COMPLETE_FUNCTION_EXTRACTION_PLAN_V1.md"


def model_path(out, reference, query, objective):
    """Address an explicitly query-conditioned, source-fitted scalar function."""
    return out / "models" / f"{reference}_{query}_{objective}.npz"


def source_jobs(jobs, reference):
    """Resolve the frozen reference without looking at any score or derivative."""
    representatives = [j for j in jobs if ref(j) == reference]
    if not representatives:
        raise ValueError("Unknown reference")
    first = representatives[0]
    members = source_set(first["cohort"], first["index"])
    return [j for j in jobs if j["cohort"] == "source4" and j["index"] in members]


def basis_path(out, reference):
    """Use only the frozen pointwise PCA mean/basis, not its old fitted transport."""
    return out / "private_bases" / f"joint_{reference}_pointwise.npz"


def feature_packet(out, key, basis):
    """Construct pure current features and directions without target/gradient fields."""
    data = arrays(out / "private_inputs" / (key + ".npz"))
    if set(data) != {"state", "grain", "positions", "cos", "sin", "mask"}:
        raise ValueError("Pure H16/GRAIN/layout inputs required")
    phi, dp = features(
        data["state"],
        body_directions(data),
        data["positions"],
        basis["mean"],
        basis["basis"],
    )
    if phi.shape != (64,) or dp.shape != (4, 64) or np.any(dp[:, 32:] != 0):
        raise ValueError(
            "Frozen 64-feature and zero last-token derivative contract differs"
        )
    return phi, dp


def training_arrays(out, jobs, basis, qi):
    """Read only original declared source L/tau fields, not current-author targets."""
    rows = []
    for j in jobs:
        key = j["query_keys"][qi]
        x, d = feature_packet(out, key, basis)
        t = selected(ENDPOINT / "capture_source" / (key + ".npz"), ("score", "tau"))
        rows.append((x, d, t["score"], t["tau"]))
    return [np.asarray(z) for z in zip(*rows, strict=True)]


def prepare(out):
    """Freeze lineage and source references before any new scalar fit or outcome."""
    start = time.process_time()
    old = verify(OLD)
    frozen(OLD, "predictions")
    jobs = read(OLD / "private_jobs.json")
    if len(jobs) != 12 or len({j["key"] for j in jobs}) != 12:
        raise ValueError("Exact twelve fixed materials required")
    out.mkdir(exist_ok=False)
    (out / "private_inputs").symlink_to(OLD / "private_inputs")
    (out / "private_bases").symlink_to(OLD / "private_models")
    write(out / "private_jobs.json", jobs)
    paths = [out / "private_jobs.json"]
    for reference in sorted({ref(j) for j in jobs}):
        p = basis_path(out, reference)
        b = selected(p, ("mean", "basis", "members"))
        expected = [j["index"] for j in source_jobs(jobs, reference)]
        np.testing.assert_array_equal(b["members"], expected)
        if b["basis"].shape != (1536, 32):
            raise ValueError("Frozen PCA32 required")
        paths.append(p)
    paths += [
        out / "private_inputs" / (k + ".npz") for j in jobs for k in j["query_keys"]
    ]
    lineage = dict(old["inputs"])
    code = [
        PLAN,
        Path(__file__),
        ROOT / "grain_research/vector_mechanisms/complete_function_extraction.py",
        ROOT / "tests/test_grain_complete_function_extraction.py",
        OLD / "manifest.json",
    ]
    lineage.update({str(p.relative_to(ROOT)): sha(p) for p in code})
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_COMPLETE_FUNCTION_EXTRACTION_V1",
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "inputs": lineage,
            "prepared": {str(p.relative_to(out)): sha(p) for p in paths},
            "evaluation": old["evaluation"],
            "max_cpu_seconds": 180,
            "max_fits": 20,
            "alpha": 0.01,
            "prepare_cpu_seconds": time.process_time() - start,
            "human_labels_read": False,
            "new_neural_calls": 0,
        },
    )
    print({"status": "PREPARED", "references": 5, "contexts": 24}, flush=True)


def fit_reference(out, reference):
    """Fit four functions in a process that cannot open excluded source targets."""
    start = time.process_time()
    m = verify(out)
    jobs = read(out / "private_jobs.json")
    src = source_jobs(jobs, reference)
    bp = basis_path(out, reference)
    targets = [
        ENDPOINT / "capture_source" / (k + ".npz") for j in src for k in j["query_keys"]
    ]
    verify_hashes(
        ROOT,
        {
            str(p.relative_to(ROOT)): m["evaluation"][str(p.relative_to(ROOT))]
            for p in targets
        },
    )
    outputs = [model_path(out, reference, q, o) for q in QUERIES for o in OBJECTIVES]
    status = out / "models" / (reference + "_status.json")
    probes = forbidden() + [
        ENDPOINT / "capture_source" / (k + ".npz")
        for j in jobs
        if j["cohort"] == "source4" and j not in src
        for k in j["query_keys"]
    ]
    denied = install(
        [
            bp,
            *targets,
            *[
                out / "private_inputs" / (k + ".npz")
                for j in src
                for k in j["query_keys"]
            ],
        ],
        outputs + [status],
        probes,
        [ROOT / "scripts", ROOT / "grain_research"],
    )
    basis = selected(bp, ("mean", "basis", "members"))
    diagnostics = []
    for qi, q in enumerate(QUERIES):
        x, d, y, t = training_arrays(out, src, basis, qi)
        for o in OBJECTIVES:
            model, diag = fit_scalar(
                x, d, y, t, o, [j["index"] for j in src], m["alpha"]
            )
            np.savez(model_path(out, reference, q, o), **model)
            diagnostics.append(
                {"reference": reference, "query_name": q, "candidate": o, **diag}
            )
    write(
        status,
        {
            "status": "COMPLETE",
            "fits": 4,
            "cpu_seconds": time.process_time() - start,
            "denial_probes": len(probes),
            "extra_denials": len(denied) - len(probes),
            "source_target_reads": len(targets),
            "current_target_reads": 0,
            "diagnostics": diagnostics,
            "files": {p.name: sha(p) for p in outputs},
        },
    )


def fit(out):
    """Run five isolated references once; preserve partial artifacts on any failure."""
    start = time.process_time()
    wall = time.monotonic()
    m = verify(out)
    jobs = read(out / "private_jobs.json")
    dst = out / "models"
    dst.mkdir(exist_ok=False)
    statuses = []
    for reference in sorted({ref(j) for j in jobs}):
        remaining = m["max_cpu_seconds"] - (time.monotonic() - wall)
        if remaining <= 0:
            raise TimeoutError("Bounded fit wall cap reached")
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
            timeout=remaining,
            cwd=ROOT,
        )
        statuses.append(read(dst / (reference + "_status.json")))
    cpu = (
        time.process_time()
        - start
        + sum(s["cpu_seconds"] for s in statuses)
        + m["prepare_cpu_seconds"]
    )
    if (
        cpu > 180
        or sum(s["fits"] for s in statuses) != 20
        or any(s["extra_denials"] for s in statuses)
    ):
        raise RuntimeError("Fit budget or membership boundary differs")
    files = {p.name: sha(p) for p in dst.iterdir() if p.suffix in (".npz", ".json")}
    write(
        dst / "freeze.json",
        {
            "status": "COMPLETE",
            "fits": 20,
            "cpu_seconds": cpu,
            "files": files,
            "current_target_reads": 0,
            "new_neural_calls": 0,
        },
    )
    print({"status": "FITTED", "fits": 20, "cpu_seconds": cpu}, flush=True)


def predict(out, folder):
    """Current application reads only pure H16/GRAIN, PCA bases and frozen functions."""
    start = time.process_time()
    m, fm = verify(out), frozen(out, "models")
    jobs = read(out / "private_jobs.json")
    dst = out / folder
    dst.mkdir(exist_ok=False)
    outputs = [dst / (k + ".npz") for j in jobs for k in j["query_keys"]]
    models = [out / "models" / n for n in fm["files"] if n.endswith(".npz")]
    probes = forbidden() + [ENDPOINT / "capture_source/source4_a00_neuroticism.npz"]
    denied = install(
        [out / n for n in m["prepared"]] + models,
        outputs + [dst / "freeze.json"],
        probes,
        [ROOT / "scripts", ROOT / "grain_research"],
    )
    for j in jobs:
        b = selected(basis_path(out, ref(j)), ("mean", "basis"))
        for q, k in zip(QUERIES, j["query_keys"], strict=True):
            phi, dp = feature_packet(out, k, b)
            predictions = [
                apply_scalar(phi, dp, arrays(model_path(out, ref(j), q, o)))
                for o in OBJECTIVES
            ]
            np.savez(
                dst / (k + ".npz"),
                score=np.asarray([p[0] for p in predictions]),
                tau=np.asarray([p[1] for p in predictions]),
            )
    cpu = time.process_time() - start
    if cpu + fm["cpu_seconds"] > m["max_cpu_seconds"] or len(denied) != len(probes):
        raise RuntimeError("Prediction budget or access boundary differs")
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
    print({"status": "PREDICTED", "contexts": 24, "cpu_seconds": cpu}, flush=True)


def evaluate(out, folder):
    """Join targets only after predictions freeze; no score calibration or selection."""
    m = verify(out)
    frozen(out, folder)
    verify_hashes(ROOT, m["evaluation"])
    jobs = read(out / "private_jobs.json")
    dst = out / ("analysis" if folder == "predictions" else "replay_analysis")
    dst.mkdir(exist_ok=False)
    scores, actions = [], []
    for j in jobs:
        f = "capture_source" if j["cohort"] == "source4" else "capture_new"
        for q, key in zip(QUERIES, j["query_keys"], strict=True):
            p = arrays(out / folder / (key + ".npz"))
            t = selected(ENDPOINT / f / (key + ".npz"), ("score", "tau"))
            for oi, o in enumerate(OBJECTIVES):
                meta = {
                    "author_key": j["key"],
                    "cohort": j["cohort"],
                    "query_name": q,
                    "candidate": o,
                }
                model = arrays(model_path(out, ref(j), q, o))
                scores.append(
                    meta
                    | {
                        "actual": float(t["score"]),
                        "predicted": p["score"][oi],
                        "constant": float(model["origin"]),
                    }
                )
                actions += [
                    meta
                    | {
                        "factor": f"whole:F{fi + 1:02d}",
                        "actual": t["tau"][fi],
                        "predicted": p["tau"][oi, fi],
                    }
                    for fi in range(4)
                ]
    sd, ad = pd.DataFrame(scores), pd.DataFrame(actions)
    keys = ["author_key", "cohort", "candidate"]
    sc = paired_queries(sd, keys, ["actual", "predicted", "constant"])
    ac = paired_queries(ad, keys + ["factor"], ["actual", "predicted"])
    if (len(sd), len(ad), len(sc), len(ac)) != (48, 192, 24, 96):
        raise ValueError("Frozen scalar/action inventory differs")
    tables = {}
    for frame, n in (
        (sd, "scores"),
        (ad, "actions"),
        (sc, "score_contrasts"),
        (ac, "action_contrasts"),
    ):
        frame.to_csv(dst / ("private_" + n + ".csv"), index=False)
    definitions = [
        (sd, ["cohort", "query_name"], "scores", True),
        (sc, ["cohort"], "score_contrasts", True),
        (ad, ["cohort", "query_name"], "actions", False),
        (ad, ["cohort", "query_name", "factor"], "factor_actions", False),
        (ac, ["cohort"], "action_contrasts", False),
        (ac, ["cohort", "factor"], "factor_contrasts", False),
    ]
    for tab, group, n, is_score in definitions:
        metrics, effects = [], []
        for k, g in tab.groupby(group + ["candidate"], sort=True):
            r = (
                score_metrics(g.actual, g.predicted, g.constant)
                if is_score
                else fidelity(g.actual.to_numpy(), g.predicted.to_numpy())
            )
            metrics.append(
                dict(zip(group + ["candidate"], k, strict=True))
                | r
                | {"authors": g.author_key.nunique()}
            )
        for k, g in tab.groupby(group, sort=True):
            k = k if isinstance(k, tuple) else (k,)
            unit = ["author_key"] + (
                ["factor"] if not is_score and "factor" not in group else []
            )
            a, b = [
                g[g.candidate == o].set_index(unit).sort_index() for o in OBJECTIVES
            ]
            if not a.index.equals(b.index):
                raise ValueError("Paired error membership mismatch")
            np.testing.assert_array_equal(a.actual, b.actual)
            effects.append(
                dict(zip(group, k, strict=True))
                | error_account(a.actual, a.predicted, b.predicted)
                | {"authors": g.author_key.nunique()}
            )
        tables[n], tables[n + "_joint_effect"] = (
            pd.DataFrame(metrics),
            pd.DataFrame(effects),
        )
    diagnostics = [
        row
        for p in sorted((out / "models").glob("*_status.json"))
        for row in read(p)["diagnostics"]
    ]
    tables["source_diagnostics"] = pd.DataFrame(diagnostics)
    for n, table in tables.items():
        table.to_csv(dst / (n + ".csv"), index=False)
    decisions = {}
    for o in OBJECTIVES:
        s = tables["scores"].query("cohort=='new8' and candidate==@o")
        a = tables["actions"].query("cohort=='new8' and candidate==@o")
        decisions[o] = {
            "score_usable": bool(
                ((s.score_r > 0) & (s.score_mse_over_source_constant < 1)).all()
            ),
            "response_usable": bool(
                ((a.action_r > 0) & (a.action_mse_over_zero < 1)).all()
            ),
        }
    write(
        dst / "summary.json",
        {
            "status": "COMPLETE",
            "decisions": decisions,
            "fits": 0,
            "new_neural_calls": 0,
            "human_labels_read": False,
            "files": {p.name: sha(p) for p in sorted(dst.glob("*.csv"))},
        },
    )
    print({"status": "EVALUATED", "decisions": decisions}, flush=True)


def main():
    """Keep source fitting, current application and target joins in separate processes."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--stage",
        choices=("prepare", "fit", "fit-reference", "predict", "evaluate"),
        required=True,
    )
    p.add_argument(
        "--output", type=Path, default=BASE / "logic_complete_function_extraction_v1"
    )
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
