#!/usr/bin/env python3
"""NEXT40: freeze source-only attention programs, predict, then open targets."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.vector_mechanisms.native_material_reader import cosine, direction
from grain_research.vector_mechanisms.source_cotangent import source_means
from scripts.analyze_grain_position_reader import action_metrics
from scripts.run_grain_fixed_seed_attention import verify as verify_native
from scripts.run_grain_reader_new_author_transport import verify_hashes
from scripts.run_grain_reader_output_seed import summarize
from scripts.run_grain_real_author_path import BASE, read, sha, write

NATIVE = BASE / "logic_fixed_seed_attention_v1"
PLAN = ROOT / "docs/GRAIN_SOURCE_COTANGENT_READER_PLAN_V1.md"


def verify(out):
    """Check immutable code/source contracts before each stage."""
    m = read(out / "manifest.json")
    verify_hashes(ROOT, m["inputs"])
    verify_hashes(out, m["prepared"])
    return m


def checked_arrays(job):
    """Verify a native packet but leave its numerical members lazily accessible."""
    p = NATIVE / "private_packets" / (job["key"] + ".json")
    if sha(p.with_suffix(".npz")) != read(p)["arrays_sha256"]:
        raise ValueError("Native arrays changed")
    return p.with_suffix(".npz")


def prepare(out):
    """Export only forward features and source g, never current g as predictor input."""
    verify_native(NATIVE)
    native = read(NATIVE / "analysis/manifest.json")
    verify_hashes(NATIVE, native["native_packet_hashes"])
    if read(NATIVE / "run_status.json")["status"] != "COMPLETE":
        raise ValueError("Complete NEXT39 required")
    jobs = read(NATIVE / "private_jobs.json")
    if len(jobs) != 24:
        raise ValueError("Fixed24 units required")
    sources = {}
    for j in jobs:
        if j["cohort"] == "source4":
            with np.load(checked_arrays(j), allow_pickle=False) as z:
                sources[j["index"], j["query"]] = z["g"]
    out.mkdir(exist_ok=False)
    for d in ("private_inputs", "private_models", "private_predictions"):
        (out / d).mkdir()
    with np.load(NATIVE / "private_projections.npz", allow_pickle=False) as z:
        np.savez_compressed(out / "private_projections.npz", **dict(z))
    files = [
        PLAN,
        Path(__file__),
        NATIVE / "manifest.json",
        NATIVE / "analysis/manifest.json",
        NATIVE / "private_jobs.json",
        NATIVE / "private_projections.npz",
    ]
    files += [
        ROOT / p
        for p in (
            "grain_research/vector_mechanisms/source_cotangent.py",
            "grain_research/vector_mechanisms/fixed_seed_attention.py",
            "grain_research/vector_mechanisms/attention_pullback.py",
            "grain_research/vector_mechanisms/native_material_reader.py",
            "scripts/replay_grain_source_cotangent.py",
        )
    ]
    predicted_jobs, design = [], []
    for j in jobs:
        path = checked_arrays(j)
        files += [path, path.with_suffix(".json"), ROOT / j["input"]]
        with np.load(path, allow_pickle=False) as z:
            forward = {n: z[n] for n in ("q", "k", "v", "cos", "sin")}
            forward["hidden"] = z["norm_hidden"]
        with np.load(ROOT / j["input"], allow_pickle=False) as z:
            forward.update(grain=z["grain"], positions=z["positions"])
        inp = out / "private_inputs" / (j["key"] + ".npz")
        model = out / "private_models" / (j["key"] + ".npz")
        np.savez_compressed(inp, **forward)
        means = source_means(
            sources, j["query"], held=j["index"] if j["cohort"] == "source4" else None
        )
        np.savez_compressed(model, **means)
        predicted_jobs.append(
            {
                "key": j["key"],
                "inputs": str(inp),
                "model": str(model),
                "weights": str(out / "private_projections.npz"),
                "output": str(out / "private_predictions" / (j["key"] + ".npz")),
            }
        )
        design.append({k: j[k] for k in ("key", "cohort", "index", "query")})
    write(out / "private_jobs.json", predicted_jobs)
    write(out / "private_design.json", design)
    prepared = [
        out / "private_projections.npz",
        out / "private_jobs.json",
        out / "private_design.json",
    ]
    prepared += [Path(j[k]) for j in predicted_jobs for k in ("inputs", "model")]
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_SOURCE_COTANGENT_V1",
            "units": 24,
            "fits": 0,
            "neural_calls": 0,
            "human_labels_read": False,
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "inputs": {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(files))},
            "prepared": {
                str(p.relative_to(out)): sha(p) for p in sorted(set(prepared))
            },
        },
    )
    print(
        {
            "status": "PREPARED",
            "units": 24,
            "source_authors": 4,
            "new_author_g_exported": False,
        },
        flush=True,
    )


def predict(out, replay=False):
    """Freeze all outputs in a process with no native target-file permissions."""
    verify(out)
    if not replay and (out / "prediction_freeze.json").exists():
        raise ValueError("Already frozen; use explicit replay")
    jobs = read(out / "private_jobs.json")
    jobfile = out / "private_jobs.json"
    if replay:
        (out / "private_replay").mkdir(exist_ok=False)
        for j in jobs:
            j["output"] = str(out / "private_replay" / Path(j["output"]).name)
        jobfile = out / "private_replay_jobs.json"
        write(jobfile, jobs)
    forbidden = [
        NATIVE / "private_packets/new8_a00_neuroticism.npz",
        NATIVE / "private_jobs.json",
    ]
    command = [
        sys.executable,
        str(ROOT / "scripts/replay_grain_source_cotangent.py"),
        "--jobs",
        str(jobfile),
    ]
    for path in forbidden:
        command += ["--forbid", str(path)]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    import json

    runtime = json.loads(result.stdout)
    if (
        runtime["units"] != 24
        or runtime["denial_probes"] != 2
        or runtime["extra_denials"] != 0
    ):
        raise ValueError("Restricted runtime did not match protocol")
    freeze = {
        "manifest_sha256": sha(out / "manifest.json"),
        "runtime": runtime,
        "files": {
            str(Path(j["output"]).relative_to(out)): sha(Path(j["output"]))
            for j in jobs
        },
        "before_evaluation": not (out / "analysis").exists(),
    }
    if replay:
        old = read(out / "prediction_freeze.json")
        for j in jobs:
            base = "private_predictions/" + Path(j["output"]).name
            if (
                sha(Path(j["output"])) != old["files"][base]
                or sha(out / base) != old["files"][base]
            ):
                raise ValueError("Numerical prediction replay changed")
        freeze["exact_packet_replays"] = len(jobs)
    write(out / ("replay_freeze.json" if replay else "prediction_freeze.json"), freeze)
    print(
        runtime | {"status": "REPLAYED" if replay else "PREDICTIONS_FROZEN"}, flush=True
    )


def evaluate(out, name):
    """Only after prediction freeze, compare fields, actions and g geometry."""
    verify(out)
    freeze = read(out / "prediction_freeze.json")
    if (
        sha(out / "manifest.json") != freeze["manifest_sha256"]
        or not freeze["before_evaluation"]
    ):
        raise ValueError("Predictions not frozen before targets")
    verify_hashes(out, freeze["files"])
    dest = out / name
    dest.mkdir(exist_ok=False)
    rows, acts, geometries = [], [], []
    dual_error = 0.0
    for j in read(out / "private_design.json"):
        key = j["key"]
        with np.load(checked_arrays(j), allow_pickle=False) as z:
            target, g = z["norm_gradient"].astype(float), z["g"].astype(float)
        with np.load(out / "private_inputs" / (key + ".npz"), allow_pickle=False) as z:
            directions = np.stack(
                [direction(z["hidden"], z["grain"], f)[0] for f in range(4)]
            )
        with np.load(out / "private_models" / (key + ".npz"), allow_pickle=False) as z:
            means = dict(z)
        with np.load(
            out / "private_predictions" / (key + ".npz"), allow_pickle=False
        ) as z:
            predictions = dict(z)
        actual = np.einsum("fth,th->f", directions, target)
        native_forward = np.einsum(
            "fhd,hd->f", predictions["response_key"] + predictions["response_value"], g
        )
        dual_error = max(dual_error, float(abs(native_forward - actual).max()))
        for mode in ("own", "pooled"):
            gh = means[mode + "_g"]
            geometries.append(
                j
                | {
                    "mode": mode,
                    "g_cosine": cosine(g, gh),
                    "g_error_ratio": float(np.sum((g - gh) ** 2) / np.sum(g * g)),
                    "g_norm_ratio": float(np.linalg.norm(gh) / np.linalg.norm(g)),
                }
            )
            for channel in ("full", "value"):
                field = predictions[mode + "_value"].copy()
                action = predictions[mode + "_value_actions"].copy()
                if channel == "full":
                    field += predictions[mode + "_key"]
                    action += predictions[mode + "_key_actions"]
                rows.append(
                    j
                    | {
                        "mode": mode,
                        "channel": channel,
                        "field_cosine": cosine(target, field),
                        "error_energy": float(np.sum((field - target) ** 2)),
                        "target_energy": float(np.sum(target * target)),
                        "relative_error": float(
                            np.sum((field - target) ** 2) / np.sum(target * target)
                        ),
                    }
                )
                for f in range(4):
                    acts.append(
                        j
                        | {
                            "mode": mode,
                            "channel": channel,
                            "factor": f"whole:F{f + 1:02d}",
                            "actual": actual[f],
                            "predicted": action[f],
                            "value_action": predictions[mode + "_value_actions"][f],
                            "key_action": predictions[mode + "_key_actions"][f],
                        }
                    )
    fields, actions, geometry = (
        pd.DataFrame(rows),
        pd.DataFrame(acts),
        pd.DataFrame(geometries),
    )
    comps, factors = [], []
    keys = ["cohort", "query", "mode", "channel"]
    for key, a in actions.groupby(keys, sort=True):
        metadata = dict(zip(keys, key, strict=True))
        f = fields
        for col, value in metadata.items():
            f = f[f[col] == value]
        centered = a[["actual", "predicted"]] - a.groupby("factor")[
            ["actual", "predicted"]
        ].transform("mean")
        comps.append(
            metadata
            | {
                "authors": a["index"].nunique(),
                "field_cosine_mean": float(f.field_cosine.mean()),
                "field_relative_error_pooled": float(
                    f.error_energy.sum() / f.target_energy.sum()
                ),
                **action_metrics(a.actual, a.predicted),
                "factor_centered_action_r": action_metrics(
                    centered.actual, centered.predicted
                )["action_r"],
            }
        )
        for factor, b in a.groupby("factor", sort=True):
            factors.append(
                metadata | {"factor": factor, **action_metrics(b.actual, b.predicted)}
            )
    fields.to_csv(dest / "private_fields.csv", index=False)
    actions.to_csv(dest / "private_actions.csv", index=False)
    geometry.to_csv(dest / "private_geometry.csv", index=False)
    pd.DataFrame(comps).to_csv(dest / "comparison.csv", index=False)
    pd.DataFrame(factors).to_csv(dest / "factor_actions.csv", index=False)
    summarize(
        geometry,
        ["cohort", "query", "mode"],
        ["g_cosine", "g_error_ratio", "g_norm_ratio"],
    ).to_csv(dest / "source_geometry.csv", index=False)
    summary = {
        "units": 24,
        "programs": 48,
        "field_conditions": len(fields),
        "action_conditions": len(actions),
        "native_forward_duality_error_max": dual_error,
        "neural_calls": 0,
        "fits": 0,
        "human_labels_read": False,
        "files": {
            p.name: sha(p)
            for p in sorted(dest.glob("*.csv"))
            if not p.name.startswith("._")
        },
    }
    write(dest / "summary.json", summary)
    print({k: v for k, v in summary.items() if k != "files"}, flush=True)


def main():
    """Use distinct immutable construction, restricted prediction and evaluation."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "predict", "replay", "evaluate"))
    p.add_argument("--name", default="logic_source_cotangent_reader_v1")
    p.add_argument("--analysis-name", default="analysis")
    a = p.parse_args()
    if any(Path(n).name != n for n in (a.name, a.analysis_name)):
        raise ValueError("Simple output names required")
    out = BASE / a.name
    with threadpool_limits(limits=2):
        if a.stage == "prepare":
            prepare(out)
        elif a.stage in ("predict", "replay"):
            predict(out, replay=a.stage == "replay")
        else:
            evaluate(out, a.analysis_name)


if __name__ == "__main__":
    main()
