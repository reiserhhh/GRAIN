#!/usr/bin/env python3
"""NEXT43: analytic block25 gate/norm mechanism, with target-denied construction."""

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
from grain_research.vector_mechanisms.gated_update import (
    ARMS,
    alternatives,
    capture_jvp,
    export_weights,
    source_members,
)
from grain_research.vector_mechanisms.stage_file_guard import install
from scripts.analyze_grain_position_reader import action_metrics
from scripts.run_grain_complete_transport_jvp import inputs, load_observer, verify
from scripts.run_grain_reader_new_author_transport import packet_arrays, verify_hashes
from scripts.run_grain_real_author_path import BASE, read, sha, write

SOURCE = BASE / "logic_layer_accumulation_v1"
PLAN = ROOT / "docs/GRAIN_GATED_UPDATE_PLAN_V1.md"


def prepare(out):
    """Copy forward states and existing source readouts without native target fields."""
    old = verify(SOURCE)
    frozen = read(SOURCE / "prediction_freeze.json")
    verify_hashes(SOURCE, frozen["files"])
    if frozen["runtime"]["status"] != "COMPLETE":
        raise ValueError("Completed NEXT42 required")
    jobs = read(SOURCE / "private_jobs.json")
    if len(jobs) != 24 or len({j["key"] for j in jobs}) != 24:
        raise ValueError("Exact24 fixed units required")
    out.mkdir(exist_ok=False)
    prepared = []
    for folder in (
        "private_inputs",
        "private_models",
        "private_capture",
        "private_predictions",
    ):
        (out / folder).mkdir()
    for j in jobs:
        for folder in ("private_inputs", "private_models"):
            p = out / folder / (j["key"] + ".npz")
            shutil.copy2(SOURCE / folder / p.name, p)
            prepared.append(p)
    write(out / "private_jobs.json", jobs)
    prepared.append(out / "private_jobs.json")
    files = [
        PLAN,
        Path(__file__),
        SOURCE / "manifest.json",
        SOURCE / "prediction_freeze.json",
        ROOT / "grain_research/vector_mechanisms/gated_update.py",
        ROOT / "grain_research/vector_mechanisms/stage_file_guard.py",
        ROOT / "tests/test_grain_gated_update.py",
    ] + [ROOT / p for p in old["inputs"]]
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_GATED_UPDATE_V1",
            "block": 25,
            "arms": list(ARMS),
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "model_sha256": old["model_sha256"],
            "max_jvps": 96,
            "max_seconds": 300,
            "numeric_max_seconds": 120,
            "capture_dtype": "float32",
            "analytic_dtype": "float64",
            "fits": 0,
            "inputs": {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(files))},
            "prepared": {str(p.relative_to(out)): sha(p) for p in prepared},
        },
    )
    print(
        {"status": "PREPARED", "units": 24, "target_fields_copied": False}, flush=True
    )


def guard(read_paths, write_paths):
    """Keep both predecessor outputs and full native gradients unreadable during prediction."""
    probes = [
        BASE / folder / area / ("new8_a00_neuroticism" + ext)
        for folder, area in (
            ("logic_layer_accumulation_v1", "private_predictions"),
            ("logic_fixed_seed_attention_v1", "private_packets"),
        )
        for ext in (".npz", ".json")
    ]
    return install(
        read_paths, write_paths, probes, [ROOT / "grain_research", ROOT / "scripts"]
    )


def capture(out):
    """Capture only pre-MLP x/d; all native calls stop at attention25."""
    m = verify(out)
    if (out / "capture_freeze.json").exists():
        print("Capture already frozen; zero extra JVPs")
        return
    jobs = read(out / "private_jobs.json")
    prior = (
        read(out / "capture_status.json")
        if (out / "capture_status.json").exists()
        else {"completed": [], "jvps": 0, "seconds": 0}
    )
    if prior.get("status") == "ERROR":
        raise ValueError("Inspect error before any correction/resume")
    observer, loading = load_observer(m)
    start = time.monotonic()
    mh = sha(out / "manifest.json")
    weights = export_weights(observer.model)
    for name in ("gate", "up", "down", "gamma"):
        if weights[name].dtype != np.float32:
            raise ValueError("Pinned float32 weights required")
    wp = out / "private_weights.npz"
    if not wp.exists():
        np.savez_compressed(wp, **weights)
    else:
        with np.load(wp, allow_pickle=False) as z:
            if set(z.files) != set(weights) or any(
                not np.array_equal(z[k], v) for k, v in weights.items()
            ):
                raise ValueError("Saved weight export differs")
    torch.set_grad_enabled(False)
    torch.func.jvp(
        lambda x: x.square(),
        (torch.ones(2, device="mps"),),
        (torch.ones(2, device="mps"),),
    )
    files = [
        out / "private_capture" / (j["key"] + ext)
        for j in jobs
        for ext in (".npz", ".json")
    ]
    write_paths = files + [
        out / n
        for n in (
            "capture_status.json",
            "capture_freeze.json",
            "resource_forecast.json",
        )
    ]
    read_paths = [wp] + [out / "private_inputs" / (j["key"] + ".npz") for j in jobs]
    denied = guard(read_paths, write_paths)
    done, jvps = list(prior["completed"]), prior["jvps"]

    def status(kind):
        result = {
            "status": kind,
            "completed": done.copy(),
            "jvps": jvps,
            "seconds": prior["seconds"] + time.monotonic() - start,
            "loading_seconds": loading,
            "denial_probes": 4,
            "extra_denials": len(denied) - 4,
            "current_scores": 0,
            "backwards": 0,
            "fits": 0,
        }
        write(out / "capture_status.json", result)
        print(
            {k: v for k, v in result.items() if k != "completed"}
            | {"units": len(done)},
            flush=True,
        )
        return result

    try:
        for j in jobs:
            if j["key"] in done:
                continue
            if (
                prior["seconds"] + time.monotonic() - start >= m["max_seconds"]
                or jvps + 4 > m["max_jvps"]
            ):
                status("PAUSED_RESOURCE_BOUNDARY")
                return
            dst = out / "private_capture" / (j["key"] + ".npz")
            if dst.exists():
                raise ValueError("Unregistered packet; inspect before resume")
            state, grain, pos = inputs(out, j)
            kw = cached_suffix_kwargs(observer.model, state)
            values, tangents = [], []
            for f in range(4):
                direction = body_direction(state, grain, pos, f)
                jvps += 1
                x, d = capture_jvp(observer.model, state, kw, direction)
                values.append(x[0].cpu().numpy())
                tangents.append(d[0].cpu().numpy())
            if any(not np.array_equal(v, values[0]) for v in values[1:]):
                raise ValueError("Primal context depends on tangent seed")
            if not np.isfinite(values).all() or not np.isfinite(tangents).all():
                raise ValueError("Nonfinite boundary state")
            np.savez_compressed(dst, x=values[0], d=np.asarray(tangents))
            write(
                dst.with_suffix(".json"),
                {"key": j["key"], "arrays_sha256": sha(dst), "primal_replay_max": 0.0},
            )
            done.append(j["key"])
            status("RUNNING")
            if len(done) == 2 and not prior["completed"]:
                forecast = (time.monotonic() - start) * len(jobs) / 2
                write(
                    out / "resource_forecast.json",
                    {"forecast_seconds": forecast, "cap_seconds": m["max_seconds"]},
                )
                if forecast > m["max_seconds"]:
                    status("PAUSED_FORECAST")
                    return
        final = status("COMPLETE")
        if jvps != 96 or final["extra_denials"] or final["seconds"] > m["max_seconds"]:
            raise ValueError("Frozen capture boundary violated")
        write(
            out / "capture_freeze.json",
            {
                "manifest_sha256": mh,
                "runtime": final,
                "files": {str(p.relative_to(out)): sha(p) for p in files + [wp]},
            },
        )
    except BaseException:
        status("ERROR")
        raise


def sources(out):
    """Compile source-only pre-MLP contexts without gradients, targets or label fitting."""
    verify(out)
    f = read(out / "capture_freeze.json")
    verify_hashes(out, f["files"])
    if f["manifest_sha256"] != sha(out / "manifest.json"):
        raise ValueError("Capture came from another construction")
    if (out / "source_freeze.json").exists():
        print("Source bank already frozen")
        return
    jobs = [j for j in read(out / "private_jobs.json") if j["cohort"] == "source4"]
    if len(jobs) != 8:
        raise ValueError("Four fixed sources x two queries required")
    xs = []
    for j in jobs:
        _, a = packet_arrays(out / "private_capture" / (j["key"] + ".json"))
        xs.append(a["x"])
    dst = out / "private_source_bank.npz"
    np.savez_compressed(
        dst,
        x=np.asarray(xs),
        index=np.asarray([j["index"] for j in jobs]),
        query=np.asarray([j["query"] for j in jobs]),
    )
    write(
        out / "source_freeze.json",
        {
            "capture_freeze_sha256": sha(out / "capture_freeze.json"),
            "source_keys": [j["key"] for j in jobs],
            "fits": 0,
            "files": {dst.name: sha(dst)},
        },
    )
    print(
        {"status": "SOURCE_BANK_FROZEN", "sources": 8, "targets_read": False},
        flush=True,
    )


def predict(out):
    """Run all four fixed numeric operators in a process denied native increment targets."""
    m = verify(out)
    capture_freeze, source_freeze = (
        read(out / "capture_freeze.json"),
        read(out / "source_freeze.json"),
    )
    verify_hashes(out, capture_freeze["files"])
    verify_hashes(out, source_freeze["files"])
    if source_freeze["capture_freeze_sha256"] != sha(out / "capture_freeze.json"):
        raise ValueError("Source/capture linkage differs")
    if (out / "prediction_freeze.json").exists():
        print("Predictions already frozen; zero extra computation")
        return
    if (out / "prediction_status.json").exists():
        raise ValueError("Inspect partial numeric output rather than silently retry")
    jobs = read(out / "private_jobs.json")
    files = [
        out / "private_predictions" / (j["key"] + ext)
        for j in jobs
        for ext in (".npz", ".json")
    ]
    mh, before = sha(out / "manifest.json"), not (out / "analysis").exists()
    read_paths = [out / "private_weights.npz", out / "private_source_bank.npz"] + [
        out / folder / (j["key"] + ext)
        for j in jobs
        for folder, ext in (
            ("private_capture", ".npz"),
            ("private_capture", ".json"),
            ("private_models", ".npz"),
        )
    ]
    write_paths = files + [
        out / "prediction_status.json",
        out / "prediction_freeze.json",
    ]
    denied = guard(read_paths, write_paths)
    start, done = time.monotonic(), []
    with np.load(out / "private_weights.npz", allow_pickle=False) as z:
        weights = {k: z[k].astype(float) for k in z.files}
    with np.load(out / "private_source_bank.npz", allow_pickle=False) as z:
        bank = dict(z)

    def status(kind):
        result = {
            "status": kind,
            "completed": done.copy(),
            "seconds": time.monotonic() - start,
            "denial_probes": 4,
            "extra_denials": len(denied) - 4,
            "current_scores": 0,
            "backwards": 0,
            "fits": 0,
            "neural_calls": 0,
        }
        write(out / "prediction_status.json", result)
        print(
            {k: v for k, v in result.items() if k != "completed"}
            | {"units": len(done)},
            flush=True,
        )
        return result

    try:
        for j in jobs:
            if time.monotonic() - start > m["numeric_max_seconds"]:
                status("PAUSED_RESOURCE_BOUNDARY")
                return
            _, a = packet_arrays(out / "private_capture" / (j["key"] + ".json"))
            if set(a) != {"x", "d"}:
                raise ValueError("Unexpected boundary schema")
            held = j["index"] if j["cohort"] == "source4" else None
            keep = source_members(bank["index"], bank["query"], j["query"], held)
            if len(keep) != (3 if held is not None else 4):
                raise ValueError("Source membership mismatch")
            terms = alternatives(a["x"], a["d"], bank["x"][keep], weights)
            arrays = {k + "_terms": v for k, v in terms.items()}
            arrays.update({k: v.sum(0) for k, v in terms.items()})
            with np.load(
                out / "private_models" / (j["key"] + ".npz"), allow_pickle=False
            ) as z:
                means = dict(z)
            for mode in ("own", "pooled"):
                for arm in ARMS:
                    arrays[mode + "_" + arm] = arrays[arm] @ means[mode + "_t"].astype(
                        float
                    )
                    arrays[mode + "_" + arm + "_terms"] = arrays[
                        arm + "_terms"
                    ] @ means[mode + "_t"].astype(float)
            if any(not np.isfinite(v).all() for v in arrays.values()):
                raise ValueError("Nonfinite analytic predictions")
            dst = out / "private_predictions" / (j["key"] + ".npz")
            np.savez_compressed(dst, **arrays)
            write(
                dst.with_suffix(".json"),
                {
                    "key": j["key"],
                    "source_indices": bank["index"][keep].tolist(),
                    "arrays_sha256": sha(dst),
                },
            )
            done.append(j["key"])
            status("RUNNING")
        final = status("COMPLETE")
        if final["extra_denials"] or final["seconds"] > m["numeric_max_seconds"]:
            raise ValueError("Numeric boundary violated")
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
    """Open only the declared NEXT42 block25 increment after prediction freeze."""
    verify(out)
    frozen = read(out / "prediction_freeze.json")
    if not frozen["before_evaluation"] or frozen["manifest_sha256"] != sha(
        out / "manifest.json"
    ):
        raise ValueError("Prediction freeze must precede evaluation")
    verify_hashes(out, frozen["files"])
    verify_hashes(SOURCE, read(SOURCE / "prediction_freeze.json")["files"])
    dest = out / name
    dest.mkdir(exist_ok=False)
    rows, vectors, checks, target_hashes = [], [], [], {}
    for j in read(out / "private_jobs.json"):
        rec, a = packet_arrays(out / "private_predictions" / (j["key"] + ".json"))
        path = SOURCE / "private_predictions" / (j["key"] + ".json")
        _, old = packet_arrays(path)
        native = old["updates"][:, 17].astype(float)
        target_hashes[str(path.relative_to(ROOT))] = sha(path)
        target_hashes[str(path.with_suffix(".npz").relative_to(ROOT))] = sha(
            path.with_suffix(".npz")
        )
        diff = a["current"] - native
        relative = float(np.linalg.norm(diff) / max(np.linalg.norm(native), 1e-30))
        if not (relative <= 1e-4 or abs(diff).max() <= 1e-5):
            raise ValueError(
                "Analytic current MLP differs from native; do not widen tolerance"
            )
        replay, sum_error = 0.0, 0.0
        with np.load(
            out / "private_models" / (j["key"] + ".npz"), allow_pickle=False
        ) as z:
            means = dict(z)
        for arm in ARMS:
            sum_error = max(
                sum_error, float(abs(a[arm] - a[arm + "_terms"].sum(0)).max())
            )
            for f in range(4):
                vectors.append(
                    j
                    | {
                        "arm": arm,
                        "factor": f"whole:F{f + 1:02d}",
                        "error_sq": float(np.square(a[arm][f] - native[f]).sum()),
                        "native_sq": float(np.square(native[f]).sum()),
                    }
                )
            for mode in ("own", "pooled"):
                t = means[mode + "_t"].astype(float)
                truth = native @ t
                predicted, terms = a[arm] @ t, a[arm + "_terms"] @ t
                replay = max(
                    replay,
                    float(abs(predicted - a[mode + "_" + arm]).max()),
                    float(abs(terms - a[mode + "_" + arm + "_terms"]).max()),
                )
                for f in range(4):
                    rows.append(
                        j
                        | {
                            "mode": mode,
                            "arm": arm,
                            "factor": f"whole:F{f + 1:02d}",
                            "actual": truth[f],
                            "predicted": predicted[f],
                            "input_term": terms[0, f],
                            "gate_term": terms[1, f],
                        }
                    )
        checks.append(
            j
            | {
                "analytic_relative_l2": relative,
                "analytic_max_absolute": float(abs(diff).max()),
                "contraction_replay": replay,
                "term_sum_error": sum_error,
                "source_indices": rec["source_indices"],
            }
        )
    actions, vector_rows = pd.DataFrame(rows), pd.DataFrame(vectors)
    comparisons, factors, components, vector_summary = [], [], [], []
    keys = ["cohort", "query", "mode", "arm"]
    for key, a in actions.groupby(keys, sort=True):
        meta = dict(zip(keys, key, strict=True))
        centered = a[["actual", "predicted"]] - a.groupby("factor")[
            ["actual", "predicted"]
        ].transform("mean")
        comparisons.append(
            meta
            | {
                "authors": a["index"].nunique(),
                **action_metrics(a.actual, a.predicted),
                "factor_centered_action_r": action_metrics(
                    centered.actual, centered.predicted
                )["action_r"],
            }
        )
        for f, b in a.groupby("factor", sort=True):
            factors.append(
                meta | {"factor": f, **action_metrics(b.actual, b.predicted)}
            )
        energy = float(np.square(a.predicted).sum())
        components.append(
            meta
            | {
                "input_projected_share": float(
                    np.dot(a.input_term, a.predicted) / energy
                ),
                "gate_projected_share": float(
                    np.dot(a.gate_term, a.predicted) / energy
                ),
                "signed_cross_energy": float(
                    2 * np.dot(a.input_term, a.gate_term) / energy
                ),
                "opposite_signs": int((a.input_term * a.gate_term < 0).sum()),
                "input_only_sse_over_zero": action_metrics(a.actual, a.input_term)[
                    "action_mse_over_zero"
                ],
                "gate_only_sse_over_zero": action_metrics(a.actual, a.gate_term)[
                    "action_mse_over_zero"
                ],
                "sum_max_absolute": float(
                    abs(a.input_term + a.gate_term - a.predicted).max()
                ),
            }
        )
    for key, a in vector_rows.groupby(["cohort", "query", "arm"], sort=True):
        vector_summary.append(
            dict(zip(("cohort", "query", "arm"), key, strict=True))
            | {
                "relative_l2": float(np.sqrt(a.error_sq.sum() / a.native_sq.sum())),
                "directions": len(a),
            }
        )
    frames = {
        "private_actions": actions,
        "private_vectors": vector_rows,
        "comparison": pd.DataFrame(comparisons),
        "factor_actions": pd.DataFrame(factors),
        "components": pd.DataFrame(components),
        "vector_summary": pd.DataFrame(vector_summary),
    }
    for filename, df in frames.items():
        df.to_csv(dest / (filename + ".csv"), index=False)
    write(dest / "private_checks.json", checks)
    summary = {
        "units": 24,
        "actions": len(actions),
        "analytic_max_relative_l2": max(r["analytic_relative_l2"] for r in checks),
        "analytic_max_absolute": max(r["analytic_max_absolute"] for r in checks),
        "contraction_replay_max": max(r["contraction_replay"] for r in checks),
        "term_sum_max_error": max(r["term_sum_error"] for r in checks),
        "capture_runtime": read(out / "capture_freeze.json")["runtime"],
        "numeric_runtime": frozen["runtime"],
        "human_labels_read": False,
        "fits": 0,
        "target_hashes": target_hashes,
        "files": {
            p.name: sha(p)
            for p in sorted(dest.glob("*.csv"))
            if not p.name.startswith("._")
        },
    }
    if (
        summary["contraction_replay_max"] > 1e-12
        or summary["term_sum_max_error"] > 1e-12
    ):
        raise ValueError("Saved analytic terms/contractions differ")
    write(dest / "summary.json", summary)
    print(
        {
            k: v
            for k, v in summary.items()
            if k not in ("files", "target_hashes", "capture_runtime", "numeric_runtime")
        },
        flush=True,
    )


def main():
    """Separate native capture, source-only compilation, denied prediction and evaluation."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "stage", choices=("prepare", "capture", "sources", "predict", "evaluate")
    )
    p.add_argument("--name", default="logic_gated_update_v1")
    p.add_argument("--analysis-name", default="analysis")
    a = p.parse_args()
    if any(Path(n).name != n or n in (".", "..") for n in (a.name, a.analysis_name)):
        raise ValueError("Simple output names required")
    with threadpool_limits(limits=2):
        if a.stage == "evaluate":
            evaluate(BASE / a.name, a.analysis_name)
        else:
            {
                "prepare": prepare,
                "capture": capture,
                "sources": sources,
                "predict": predict,
            }[a.stage](BASE / a.name)


if __name__ == "__main__":
    main()
