#!/usr/bin/env python3
"""NEXT34: source-only context/GRAIN coefficient readers, no new neural calls."""

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
from grain_research.vector_mechanisms.content_coefficients import (
    fit_head,
    fit_preprocessor,
    predict_coefficients,
)
from scripts.run_grain_native_material_reader import QUERIES, cached_unit
from scripts.run_grain_position_reader import SOURCE as NATIVE
from scripts.run_grain_position_reader import verified as old_verified
from scripts.run_grain_real_author_path import BASE, read, sha, write

SOURCE = BASE / "logic_position_reader_v1"
PLAN = ROOT / "docs/GRAIN_CONTENT_COEFFICIENT_READER_PLAN_V1.md"
ARMS = ("content", "content_grain")


def prepare(out):
    """Bind the previous shared bases, numerical contexts and new fixed program."""
    old = old_verified(SOURCE)
    f = read(SOURCE / "prediction_freeze.json")
    if f["manifest_sha256"] != sha(SOURCE / "manifest.json"):
        raise ValueError("Prior construction changed")
    assets = dict(old["inputs"])
    for p, digest in f["files"].items():
        if sha(SOURCE / p) != digest:
            raise ValueError("Prior frozen prediction changed")
        assets[str((SOURCE / p).relative_to(ROOT))] = digest
    for p in (SOURCE / "manifest.json", SOURCE / "prediction_freeze.json"):
        assets[str(p.relative_to(ROOT))] = sha(p)
    code = [
        PLAN,
        Path(__file__),
        ROOT / "scripts/replay_grain_content_coefficients.py",
        ROOT / "grain_research/vector_mechanisms/content_coefficients.py",
        ROOT / "grain_research/vector_mechanisms/position_reader.py",
    ]
    out.mkdir(exist_ok=False)
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_CONTENT_COEFFICIENTS_V1",
            "inputs": assets,
            "code_hashes": {str(p.relative_to(ROOT)): sha(p) for p in code},
            "queries": list(QUERIES),
            "arms": list(ARMS),
            "pca_rank": 32,
            "reader_rank": 8,
            "alpha": 0.01,
            "pca_seed": 4300,
            "threads": 4,
            "max_seconds": 240,
            "new_neural_calls": 0,
            "human_labels_read": False,
        },
    )
    write(out / "private_membership.json", read(SOURCE / "private_membership.json"))


def verified(out):
    """Verify source assets and current construction code before fitting/evaluation."""
    m = read(out / "manifest.json")
    for key in ("inputs", "code_hashes"):
        for p, digest in m[key].items():
            if sha(ROOT / p) != digest:
                raise ValueError("Frozen dependency changed: " + Path(p).name)
    return m


def construct(name):
    """Fit at most four PCAs and24 Ridge heads, then gradient-denied predictions."""
    out = BASE / name
    if not out.exists():
        prepare(out)
    m = verified(out)
    if (out / "prediction_freeze.json").exists():
        raise ValueError("Complete packet exists; use a new explicit replay name")
    members = read(out / "private_membership.json")
    indices = [r["index"] for r in members["authors"]]
    if len(indices) != 4 or len(set(indices)) != 4:
        raise ValueError("Four distinct old authors required")
    with np.load(NATIVE / "private_coordinates.npz") as z:
        coord = dict(z)
    hidden, grain = {}, {}
    for i in indices:
        with np.load(cached_unit(i, QUERIES[0]).with_suffix(".npz")) as z:
            hidden[i] = z["full_state"][coord[f"a{i}__positions"]].astype(float)
        grain[i] = coord[f"a{i}__grain"]
    paths, jobs, sources = {}, [], []
    start = time.monotonic()
    prep_freeze = out / "preprocessing_freeze.json"
    if prep_freeze.exists():
        for p, digest in read(prep_freeze)["files"].items():
            if sha(out / p) != digest:
                raise ValueError("Source preprocessing/input packet changed")
    elif any((out / f"private_preprocessor_{i}.npz").exists() for i in indices):
        raise ValueError("Unsealed partial preprocessing exists; inspect before retry")
    with threadpool_limits(limits=4):
        for slot, i in enumerate(indices):
            p = out / f"private_preprocessor_{i}.npz"
            if p.exists():
                with np.load(p) as z:
                    paths[i] = dict(z)
            else:
                src = [s for s in indices if s != i]
                paths[i] = fit_preprocessor(
                    [hidden[s] for s in src], [grain[s] for s in src], seed=4300 + slot
                )
                np.savez_compressed(p, **paths[i])
                np.savez_compressed(
                    out / f"private_inputs_{i}.npz",
                    hidden=hidden[i],
                    grain=grain[i],
                    positions=coord[f"a{i}__positions"],
                )
        if not prep_freeze.exists():
            names = [
                f"private_{kind}_{i}.npz"
                for kind in ("preprocessor", "inputs")
                for i in indices
            ]
            write(prep_freeze, {"files": {p: sha(out / p) for p in names}})
        for qi, q in enumerate(QUERIES):
            folder = out / q
            if (folder / "freeze.json").exists():
                saved = read(folder / "freeze.json")
                for p, digest in saved["files"].items():
                    if sha(folder / p) != digest:
                        raise ValueError("Saved complete source fits changed")
            else:
                if folder.exists():
                    raise ValueError("Partial unsealed query: inspect before retry")
                if qi and (time.monotonic() - start) / qi * 3 > m["max_seconds"]:
                    write(
                        out / "run_status.json",
                        {
                            "status": "PAUSED_RESOURCE_BOUNDARY",
                            "complete_queries": list(QUERIES[:qi]),
                            "seconds": time.monotonic() - start,
                        },
                    )
                    return
                folder.mkdir()
                with np.load(SOURCE / q / "private_factors.npz") as z:
                    bases = {i: z[f"a{i}__source_basis"] for i in indices}
                records = []
                for i in indices:
                    src = [s for s in indices if s != i]
                    # Never load this fold's held gradient for fitting or prediction.
                    js = []
                    for s in src:
                        with np.load(
                            NATIVE / "private_packets" / f"a{s:02d}_{q}.npz"
                        ) as z:
                            js.append(z["body_gradient"].astype(float))
                    for arm in ARMS:
                        model, fit = fit_head(
                            [hidden[s] for s in src],
                            [grain[s] for s in src],
                            js,
                            paths[i],
                            bases[i],
                            arm == "content_grain",
                        )
                        np.savez_compressed(
                            folder / f"private_model_{i}_{arm}.npz", **model
                        )
                        records.append({"query": q, "held": i, "arm": arm, **fit})
                pd.DataFrame(records).to_csv(
                    folder / "private_source_fit.csv", index=False
                )
                names = [
                    f"private_model_{i}_{arm}.npz" for i in indices for arm in ARMS
                ] + ["private_source_fit.csv"]
                write(
                    folder / "freeze.json",
                    {"files": {p: sha(folder / p) for p in names}},
                )
            sources.append(pd.read_csv(folder / "private_source_fit.csv"))
            for i in indices:
                for arm in ARMS:
                    jobs.append(
                        {
                            "key": f"{q}__a{i}__{arm}",
                            "model": str(folder / f"private_model_{i}_{arm}.npz"),
                            "inputs": str(out / f"private_inputs_{i}.npz"),
                        }
                    )
            print(
                {
                    "query": q,
                    "status": "SOURCE_FITS_COMPLETE",
                    "seconds": round(time.monotonic() - start, 3),
                },
                flush=True,
            )
        write(out / "private_jobs.json", jobs)
        prediction = out / "private_predictions.npz"
        command = [
            sys.executable,
            str(ROOT / "scripts/replay_grain_content_coefficients.py"),
            "--jobs",
            str(out / "private_jobs.json"),
            "--output",
            str(prediction),
        ]
        for p in (
            NATIVE / "private_packets" / f"a{indices[0]:02d}_{QUERIES[0]}.npz",
            SOURCE / QUERIES[0] / "private_actions.csv",
        ):
            command += ["--forbid", str(p)]
        replay = subprocess.run(
            command, capture_output=True, text=True, timeout=120, check=True
        )
        independent = read_json_output(replay.stdout)
        discrepancy = 0.0
        with np.load(prediction) as predictions:
            for j in jobs:
                with np.load(j["model"]) as z:
                    model = dict(z)
                with np.load(j["inputs"]) as z:
                    local = predict_coefficients(z["hidden"], z["grain"], model)
                discrepancy = max(
                    discrepancy, float(np.max(abs(local - predictions[j["key"]])))
                )
        if (
            discrepancy != 0
            or independent["extra_denials"]
            or independent["jobs"] != 24
        ):
            raise ValueError("Independent prediction verification failed")
    pd.concat(sources, ignore_index=True).to_csv(
        out / "private_source_fit.csv", index=False
    )
    write(
        out / "construction_summary.json",
        {
            "status": "COMPLETE",
            "authors": 4,
            "queries": 3,
            "pca_fits": 4,
            "ridge_fits": 24,
            "predictions": 24,
            "source_only_predictors": True,
            "runtime_replay_max": discrepancy,
            **independent,
            "new_neural_calls": 0,
            "human_labels_read": False,
        },
    )
    names = [
        "private_membership.json",
        "private_predictions.npz",
        "private_source_fit.csv",
        "construction_summary.json",
    ]
    names += [
        f"private_{kind}_{i}.npz"
        for kind in ("preprocessor", "inputs")
        for i in indices
    ]
    for q in QUERIES:
        names += [q + "/" + p for p in read(out / q / "freeze.json")["files"]]
    write(
        out / "prediction_freeze.json",
        {
            "manifest_sha256": sha(out / "manifest.json"),
            "runtime_jobs_sha256": sha(out / "private_jobs.json"),
            "files": {p: sha(out / p) for p in names},
        },
    )
    write(
        out / "run_status.json",
        {
            "status": "COMPLETE",
            "seconds": time.monotonic() - start,
            "new_neural_calls": 0,
        },
    )
    print(
        {
            "status": "COMPLETE",
            "fits": 24,
            "new_neural_calls": 0,
            "seconds": round(time.monotonic() - start, 3),
        },
        flush=True,
    )


def read_json_output(text):
    """Parse only the independent runtime's numeric completion record."""
    import json

    return json.loads(text.strip())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="logic_content_coefficients_v1")
    construct(p.parse_args().name)
