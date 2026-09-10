#!/usr/bin/env python3
"""NEXT42: fixed early exits and signed layer-update accounting, without new fits."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.vector_mechanisms.complete_transport import body_direction
from grain_research.vector_mechanisms.context_carrier import cached_suffix_kwargs
from grain_research.vector_mechanisms.layer_accumulation import (
    accumulation_jvp,
    stage_labels,
    truncated_jvp,
)
from scripts.analyze_grain_position_reader import action_metrics
from scripts.run_grain_complete_transport_jvp import (
    inputs,
    install_guard,
    load_observer,
    verify,
)
from scripts.run_grain_reader_new_author_transport import packet_arrays, verify_hashes
from scripts.run_grain_real_author_path import BASE, read, sha, write

SOURCE = BASE / "logic_complete_transport_jvp_v1_r2"
EXITS = (20, 23, 26)
CHANNELS = ("after20", "after23", "after26", "full")
PLAN = ROOT / "docs/GRAIN_LAYER_ACCUMULATION_PLAN_V1.md"


def prepare(out):
    """Copy fixed forward-only inputs/source means; hash old targets without reading them."""
    source = verify(SOURCE)
    sf = read(SOURCE / "source_freeze.json")
    pf = read(SOURCE / "prediction_freeze.json")
    verify_hashes(SOURCE, sf["files"])
    verify_hashes(SOURCE, pf["files"])
    if pf["runtime"]["status"] != "COMPLETE":
        raise ValueError("Completed predecessor required")
    jobs = read(SOURCE / "private_jobs.json")
    if len(jobs) != 24 or len({j["key"] for j in jobs}) != 24:
        raise ValueError("Exact24 unique predecessor units required")
    out.mkdir(exist_ok=False)
    prepared = []
    for folder in ("private_inputs", "private_models", "private_predictions"):
        (out / folder).mkdir()
    for job in jobs:
        for folder in ("private_inputs", "private_models"):
            dst = out / folder / (job["key"] + ".npz")
            shutil.copy2(SOURCE / folder / dst.name, dst)
            prepared.append(dst)
    write(out / "private_jobs.json", jobs)
    prepared.append(out / "private_jobs.json")
    files = [
        PLAN,
        Path(__file__),
        ROOT / "grain_research/vector_mechanisms/layer_accumulation.py",
        ROOT / "tests/test_grain_layer_accumulation.py",
        ROOT / "scripts/run_grain_complete_transport_jvp.py",
        ROOT / "scripts/analyze_grain_position_reader.py",
        ROOT / "scripts/run_grain_reader_new_author_transport.py",
        ROOT / "scripts/run_grain_real_author_path.py",
        SOURCE / "manifest.json",
        SOURCE / "source_freeze.json",
        SOURCE / "prediction_freeze.json",
        SOURCE / "analysis/private_actions.csv",
        SOURCE / "analysis/summary.json",
    ] + [ROOT / p for p in source["inputs"]]
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_LAYER_ACCUMULATION_V1",
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "exits": list(EXITS),
            "channels": list(CHANNELS),
            "stages": stage_labels(),
            "model_sha256": source["model_sha256"],
            "max_seconds": 300,
            "max_jvps": 100,
            "source_fits": 0,
            "source_forwards": 0,
            "source_backwards": 0,
            "inputs": {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(files))},
            "prepared": {str(p.relative_to(out)): sha(p) for p in prepared},
        },
    )
    print({"status": "PREPARED", "units": 24, "targets_exported": False}, flush=True)


def predict(out):
    """Freeze all exits before target access; enforce the declared finite execution budget."""
    m = verify(out)
    if (out / "prediction_freeze.json").exists():
        print("Already frozen; no additional model execution")
        return
    jobs = read(out / "private_jobs.json")
    prior = (
        read(out / "prediction_status.json")
        if (out / "prediction_status.json").exists()
        else {
            "seconds": 0,
            "completed": [],
            "jvps": 0,
            "short_jvps": 0,
        }
    )
    if prior.get("status") == "ERROR":
        raise ValueError("Inspect failed packet before any explicit correction")
    observer, loading = load_observer(m)
    mh = sha(out / "manifest.json")
    before = not (out / "analysis").exists()
    torch.set_grad_enabled(False)
    denied = install_guard(out, jobs)
    for ext in (".npz", ".json"):
        try:
            with open(
                SOURCE / "private_predictions" / ("new8_a00_neuroticism" + ext), "rb"
            ):
                pass
        except PermissionError:
            continue
        raise AssertionError("Predecessor response denial failed")
    start, done = time.monotonic(), list(prior["completed"])
    jvps, short_jvps = prior["jvps"], prior["short_jvps"]

    def status(kind):
        value = {
            "status": kind,
            "completed": done.copy(),
            "jvps": jvps,
            "short_jvps": short_jvps,
            "seconds": prior["seconds"] + time.monotonic() - start,
            "load_seconds": loading,
            "denial_probes": 4,
            "extra_denials": len(denied) - 4,
            "current_scores": 0,
            "current_backwards": 0,
            "fits": 0,
        }
        write(out / "prediction_status.json", value)
        print(
            {k: v for k, v in value.items() if k != "completed"} | {"units": len(done)},
            flush=True,
        )
        return value

    try:
        for job in jobs:
            if job["key"] in done:
                continue
            short = (
                job["cohort"] == "source4"
                and job["index"] == 0
                and job["query"] == "neuroticism"
            )
            if (
                prior["seconds"] + time.monotonic() - start >= m["max_seconds"]
                or jvps + 4 + 4 * short > m["max_jvps"]
            ):
                status("PAUSED_RESOURCE_BOUNDARY")
                return
            dst = out / "private_predictions" / (job["key"] + ".npz")
            if dst.exists():
                raise ValueError("Unregistered existing packet; inspect before resume")
            state, grain, positions = inputs(out, job)
            kwargs = cached_suffix_kwargs(observer.model, state)
            with np.load(out / "private_models" / dst.name, allow_pickle=False) as z:
                means = dict(z)
            ends, updates, initials, short_errors = [], [], [], []
            for f in range(4):
                direction = body_direction(state, grain, positions, f)
                jvps += 1
                _, tangent = accumulation_jvp(observer.model, state, kwargs, direction)
                ends.append(tangent[0][:, 0].cpu().numpy())
                updates.append(tangent[1][:, 0].cpu().numpy())
                initials.append(tangent[2][0].cpu().numpy())
                if short:
                    jvps += 1
                    short_jvps += 1
                    result = (
                        truncated_jvp(observer.model, state, kwargs, direction)[0]
                        .cpu()
                        .numpy()
                    )
                    err = result.astype(float) - ends[-1][0]
                    short_errors.append(
                        {
                            "factor": f"whole:F{f + 1:02d}",
                            "relative_l2": float(
                                np.linalg.norm(err) / max(np.linalg.norm(result), 1e-30)
                            ),
                            "absolute": float(abs(err).max()),
                        }
                    )
            arrays = {
                "endpoints": np.asarray(ends),
                "updates": np.asarray(updates),
                "initial": np.asarray(initials),
            }
            for mode in ("own", "pooled"):
                for kind in ("endpoints", "updates", "initial"):
                    arrays[mode + "_" + kind] = arrays[kind].astype(float) @ means[
                        mode + "_t"
                    ].astype(float)
            err = (
                arrays["updates"].astype(float).sum(1)
                + arrays["initial"]
                - arrays["endpoints"][:, -1]
            )
            relative = float(
                np.linalg.norm(err)
                / max(np.linalg.norm(arrays["endpoints"][:, -1]), 1e-30)
            )
            record = {
                "key": job["key"],
                "telescoping_relative_l2": relative,
                "telescoping_max_absolute": float(abs(err).max()),
                "short_checks": short_errors,
            }
            if not all(np.isfinite(a).all() for a in arrays.values()):
                raise ValueError("Nonfinite accumulation")
            np.savez_compressed(dst, **arrays)
            write(dst.with_suffix(".json"), record | {"arrays_sha256": sha(dst)})
            done.append(job["key"])
            status("RUNNING")
            if np.count_nonzero(arrays["initial"]) or not (
                relative <= 1e-4 or abs(err).max() <= 1e-5
            ):
                raise ValueError("Initial direction or update telescoping differs")
            if any(
                not (r["relative_l2"] <= 1e-4 or r["absolute"] <= 1e-5)
                for r in short_errors
            ):
                raise ValueError("Actual truncated program differs")
            if len(done) == 2 and not prior["completed"]:
                forecast = (time.monotonic() - start) * len(jobs) / 2
                write(
                    out / "resource_forecast.json",
                    {"forecast_seconds": forecast, "native_cap": m["max_seconds"]},
                )
                if forecast > m["max_seconds"]:
                    status("PAUSED_FORECAST")
                    return
        final = status("COMPLETE")
        if (
            final["extra_denials"]
            or jvps != 100
            or short_jvps != 4
            or final["seconds"] > m["max_seconds"]
        ):
            raise ValueError("Frozen execution boundary violated")
        files = [
            out / "private_predictions" / (j["key"] + ext)
            for j in jobs
            for ext in (".npz", ".json")
        ]
        write(
            out / "prediction_freeze.json",
            {
                "manifest_sha256": mh,
                "before_evaluation": before,
                "runtime": final,
                "files": {str(p.relative_to(out)): sha(p) for p in files},
            },
        )
        print("PREDICTIONS_FROZEN", flush=True)
    except BaseException:
        status("ERROR")
        raise


def evaluate(out, name):
    """Read native targets only after freeze; report every fixed exit and signed update."""
    verify(out)
    freeze = read(out / "prediction_freeze.json")
    if not freeze["before_evaluation"] or freeze["manifest_sha256"] != sha(
        out / "manifest.json"
    ):
        raise ValueError("Predictions must precede evaluation")
    verify_hashes(out, freeze["files"])
    verify_hashes(SOURCE, read(SOURCE / "prediction_freeze.json")["files"])
    target_path = SOURCE / "analysis/private_actions.csv"
    targets = pd.read_csv(target_path)
    targets = targets[(targets["mode"] == "own") & (targets.channel == "response")]
    if len(targets) != 96 or targets.duplicated(["key", "factor"]).any():
        raise ValueError("Exact unique native target rows required")
    dest = out / name
    dest.mkdir(exist_ok=False)
    actions, changes, checks = [], [], []
    for job in read(out / "private_jobs.json"):
        rec, a = packet_arrays(out / "private_predictions" / (job["key"] + ".json"))
        _, old = packet_arrays(SOURCE / "private_predictions" / (job["key"] + ".json"))
        err = a["endpoints"][:, -1].astype(float) - old["response"]
        rel = float(np.linalg.norm(err) / max(np.linalg.norm(old["response"]), 1e-30))
        if not (rel <= 1e-4 or abs(err).max() <= 1e-5):
            raise ValueError("Full response differs from frozen predecessor")
        truth = targets[targets.key == job["key"]].set_index("factor").actual
        with np.load(
            out / "private_models" / (job["key"] + ".npz"), allow_pickle=False
        ) as z:
            means = dict(z)
        replay = 0.0
        for mode in ("own", "pooled"):
            for kind in ("endpoints", "updates", "initial"):
                replay = max(
                    replay,
                    float(
                        abs(
                            a[kind].astype(float) @ means[mode + "_t"].astype(float)
                            - a[mode + "_" + kind]
                        ).max()
                    ),
                )
            for f in range(4):
                factor = f"whole:F{f + 1:02d}"
                meta = job | {"mode": mode, "factor": factor}
                for c, channel in enumerate(CHANNELS):
                    actions.append(
                        meta
                        | {
                            "channel": channel,
                            "actual": float(truth[factor]),
                            "predicted": a[mode + "_endpoints"][f, c],
                        }
                    )
                cumulative = float(a[mode + "_initial"][f])
                full = float(a[mode + "_endpoints"][f, -1])
                for s, label in enumerate(stage_labels()):
                    value = float(a[mode + "_updates"][f, s])
                    cumulative += value
                    changes.append(
                        meta
                        | {
                            "stage_index": s,
                            "stage": label,
                            "update": value,
                            "cumulative": cumulative,
                            "full": full,
                            "actual": float(truth[factor]),
                        }
                    )
        checks.append(
            job
            | {
                "full_relative_l2": rel,
                "full_max_absolute": float(abs(err).max()),
                "contraction_replay_error": replay,
                **rec,
            }
        )
    actions, changes = pd.DataFrame(actions), pd.DataFrame(changes)
    comparison, factors, stages = [], [], []
    keys = ["cohort", "query", "mode", "channel"]
    for key, a in actions.groupby(keys, sort=True):
        meta = dict(zip(keys, key, strict=True))
        centered = a[["actual", "predicted"]] - a.groupby("factor")[
            ["actual", "predicted"]
        ].transform("mean")
        comparison.append(
            meta
            | {
                "authors": a["index"].nunique(),
                **action_metrics(a.actual, a.predicted),
                "factor_centered_action_r": action_metrics(
                    centered.actual, centered.predicted
                )["action_r"],
            }
        )
        for factor, b in a.groupby("factor", sort=True):
            factors.append(
                meta | {"factor": factor, **action_metrics(b.actual, b.predicted)}
            )
    keys = ["cohort", "query", "mode", "stage_index", "stage"]
    for key, a in changes.groupby(keys, sort=True):
        denominator = float(np.square(a.full).sum())
        stages.append(
            dict(zip(keys, key, strict=True))
            | {
                "signed_projection_share": float(
                    np.dot(a["update"], a.full) / denominator
                ),
                "update_rms": float(np.sqrt(np.square(a["update"]).mean())),
                "opposite_sign": int((a["update"] * a.full < 0).sum()),
                "cumulative_action_r": action_metrics(a.actual, a.cumulative)[
                    "action_r"
                ],
                "cumulative_sse_over_zero": action_metrics(a.actual, a.cumulative)[
                    "action_mse_over_zero"
                ],
            }
        )
    for filename, frame in (
        ("private_actions", actions),
        ("private_updates", changes),
        ("comparison", pd.DataFrame(comparison)),
        ("factor_actions", pd.DataFrame(factors)),
        ("stage_summary", pd.DataFrame(stages)),
    ):
        frame.to_csv(dest / (filename + ".csv"), index=False)
    write(dest / "private_checks.json", checks)
    summary = {
        "units": 24,
        "actions": len(actions),
        "signed_updates": len(changes),
        "full_max_relative_l2": max(r["full_relative_l2"] for r in checks),
        "full_max_absolute": max(r["full_max_absolute"] for r in checks),
        "contraction_replay_max_error": max(
            r["contraction_replay_error"] for r in checks
        ),
        "telescoping_max_relative_l2": max(
            r["telescoping_relative_l2"] for r in checks
        ),
        "telescoping_max_absolute": max(r["telescoping_max_absolute"] for r in checks),
        "short_checks": [c for r in checks for c in r["short_checks"]],
        "runtime": freeze["runtime"],
        "fits": 0,
        "human_labels_read": False,
        "target_file_sha256": sha(target_path),
        "files": {
            p.name: sha(p)
            for p in sorted(dest.glob("*.csv"))
            if not p.name.startswith("._")
        },
    }
    if summary["contraction_replay_max_error"] > 1e-12:
        raise ValueError("Saved source contractions differ on replay")
    write(dest / "summary.json", summary)
    print(
        {k: v for k, v in summary.items() if k not in ("files", "runtime")}, flush=True
    )


def main():
    """Keep target-denied transport in a separate process from evaluation."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "predict", "evaluate"))
    p.add_argument("--name", default="logic_layer_accumulation_v1")
    p.add_argument("--analysis-name", default="analysis")
    a = p.parse_args()
    if any(Path(n).name != n or n in (".", "..") for n in (a.name, a.analysis_name)):
        raise ValueError("Simple output names required")
    with threadpool_limits(limits=2):
        if a.stage == "prepare":
            prepare(BASE / a.name)
        elif a.stage == "predict":
            predict(BASE / a.name)
        else:
            evaluate(BASE / a.name, a.analysis_name)


if __name__ == "__main__":
    main()
