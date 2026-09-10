#!/usr/bin/env python3
"""NEXT35: matched variance/query residual input comparisons on frozen source roles."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.vector_mechanisms.query_residual_input import (
    coordinate_support,
    fit_extension,
    predict_extension,
    variance_basis,
)
from scripts.analyze_grain_content_coefficients import (
    verify_predictions as prior_verified,
)
from scripts.run_grain_content_coefficients import NATIVE
from scripts.run_grain_real_author_path import BASE, read, sha, write

SOURCE = BASE / "logic_content_coefficients_v1"
PLAN = ROOT / "docs/GRAIN_QUERY_RESIDUAL_INPUT_PLAN_V1.md"
ARMS = ("residual_variance", "residual_query")


def prepare(out):
    """Bind all numerical source inputs and immutable predecessor assets."""
    old = prior_verified(SOURCE)
    assets = dict(old["inputs"])
    for p, digest in read(SOURCE / "prediction_freeze.json")["files"].items():
        assets[str((SOURCE / p).relative_to(ROOT))] = digest
    for p in (SOURCE / "manifest.json", SOURCE / "prediction_freeze.json"):
        assets[str(p.relative_to(ROOT))] = sha(p)
    code = [
        PLAN,
        Path(__file__),
        ROOT / "scripts/replay_grain_query_residual.py",
        ROOT / "scripts/replay_grain_content_coefficients.py",
        ROOT / "grain_research/vector_mechanisms/query_residual_input.py",
        ROOT / "grain_research/vector_mechanisms/content_coefficients.py",
    ]
    out.mkdir(exist_ok=False)
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_QUERY_RESIDUAL_INPUT_V1",
            "code_hashes": {str(p.relative_to(ROOT)): sha(p) for p in code},
            "inputs": assets,
            "queries": old["queries"],
            "arms": list(ARMS),
            "extra_rank": 8,
            "alpha": 0.01,
            "seed": 4400,
            "max_seconds": 240,
            "threads": 4,
            "new_neural_calls": 0,
            "human_labels_read": False,
        },
    )
    write(out / "private_membership.json", read(SOURCE / "private_membership.json"))


def verified(out):
    """Verify frozen numerical dependencies before any reconstruction or analysis."""
    m = read(out / "manifest.json")
    for key in ("code_hashes", "inputs"):
        for p, digest in m[key].items():
            if sha(ROOT / p) != digest:
                raise ValueError("Frozen dependency changed: " + Path(p).name)
    return m


def construct(name):
    """Fit bounded source-only extensions and export gradient-denied predictions."""
    out = BASE / name
    if not out.exists():
        prepare(out)
    m = verified(out)
    if (out / "prediction_freeze.json").exists():
        raise ValueError("Completed packet exists; use an explicit replay name")
    indices = [r["index"] for r in read(out / "private_membership.json")["authors"]]
    if len(indices) != 4 or len(set(indices)) != 4:
        raise ValueError("Four old authors required")
    packets, bases = {}, {}
    for i in indices:
        with np.load(SOURCE / f"private_inputs_{i}.npz") as z:
            packets[i] = (z["hidden"], z["grain"])
        for q in m["queries"]:
            with np.load(SOURCE / q / f"private_model_{i}_content.npz") as z:
                bases[(q, i)] = dict(z)
    for i in indices:
        for q in m["queries"][1:]:
            for k in ("h_center", "h_basis"):
                if not np.array_equal(bases[(q, i)][k], bases[(m["queries"][0], i)][k]):
                    raise ValueError(
                        "Shared source preprocessing differs across queries"
                    )
    start = time.monotonic()
    variance, jobs, source_records = {}, [], []
    with threadpool_limits(limits=4):
        vf = out / "variance_freeze.json"
        if vf.exists():
            for p, digest in read(vf)["files"].items():
                if sha(out / p) != digest:
                    raise ValueError("Frozen source variance basis changed")
        elif any((out / f"private_variance_{i}.npz").exists() for i in indices):
            raise ValueError(
                "Partial unsealed variance construction; inspect before retry"
            )
        for slot, i in enumerate(indices):
            path = out / f"private_variance_{i}.npz"
            if path.exists():
                with np.load(path) as z:
                    variance[i] = dict(z)
            else:
                src = [s for s in indices if s != i]
                variance[i] = variance_basis(
                    [packets[s][0] for s in src],
                    bases[(m["queries"][0], i)],
                    seed=4400 + slot,
                )
                np.savez_compressed(path, **variance[i])
        if not vf.exists():
            write(
                vf,
                {
                    "files": {
                        f"private_variance_{i}.npz": sha(
                            out / f"private_variance_{i}.npz"
                        )
                        for i in indices
                    }
                },
            )
        for qi, q in enumerate(m["queries"]):
            folder = out / q
            if (folder / "freeze.json").exists():
                for p, digest in read(folder / "freeze.json")["files"].items():
                    if sha(folder / p) != digest:
                        raise ValueError("Complete query unit changed")
            else:
                if folder.exists():
                    raise ValueError("Partial unsealed query; inspect before retry")
                if qi and (time.monotonic() - start) / qi * 3 > m["max_seconds"]:
                    write(
                        out / "run_status.json",
                        {
                            "status": "PAUSED_RESOURCE_BOUNDARY",
                            "complete_queries": m["queries"][:qi],
                        },
                    )
                    return
                folder.mkdir()
                records = []
                for i in indices:
                    src = [s for s in indices if s != i]
                    js = []
                    for s in src:
                        with np.load(
                            NATIVE / "private_packets" / f"a{s:02d}_{q}.npz"
                        ) as z:
                            js.append(z["body_gradient"].astype(float))
                    for arm in ARMS:
                        model, record = fit_extension(
                            [packets[s][0] for s in src],
                            [packets[s][1] for s in src],
                            js,
                            bases[(q, i)],
                            variance[i] if arm == "residual_variance" else None,
                        )
                        np.savez_compressed(
                            folder / f"private_model_{i}_{arm}.npz", **model
                        )
                        records.append(
                            {
                                "query": q,
                                "arm": arm,
                                "held": i,
                                **record,
                                **coordinate_support(packets[i][0], model),
                            }
                        )
                pd.DataFrame(records).to_csv(
                    folder / "private_source_fit.csv", index=False
                )
                names = [
                    f"private_model_{i}_{a}.npz" for i in indices for a in ARMS
                ] + ["private_source_fit.csv"]
                write(
                    folder / "freeze.json",
                    {"files": {p: sha(folder / p) for p in names}},
                )
            source_records.append(pd.read_csv(folder / "private_source_fit.csv"))
            for i in indices:
                for arm in ARMS:
                    jobs.append(
                        {
                            "key": f"{q}__a{i}__{arm}",
                            "model": str(folder / f"private_model_{i}_{arm}.npz"),
                            "inputs": str(SOURCE / f"private_inputs_{i}.npz"),
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
        command = [
            sys.executable,
            str(ROOT / "scripts/replay_grain_query_residual.py"),
            "--jobs",
            str(out / "private_jobs.json"),
            "--output",
            str(out / "private_predictions.npz"),
        ]
        for p in (
            NATIVE / "private_packets" / f"a{indices[0]:02d}_{m['queries'][0]}.npz",
            SOURCE / "private_predictions.npz",
        ):
            command += ["--forbid", str(p)]
        run = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=120
        )
        record = json.loads(run.stdout)
        error = 0.0
        with np.load(out / "private_predictions.npz") as yp:
            for j in jobs:
                with np.load(j["model"]) as z:
                    model = dict(z)
                with np.load(j["inputs"]) as z:
                    pred = predict_extension(z["hidden"], z["grain"], model)
                error = max(error, float(np.max(abs(pred - yp[j["key"]]))))
        if error != 0 or record["extra_denials"] or record["jobs"] != 24:
            raise ValueError("Independent inference mismatch")
    pd.concat(source_records, ignore_index=True).to_csv(
        out / "private_source_fit.csv", index=False
    )
    write(
        out / "construction_summary.json",
        {
            "status": "COMPLETE",
            "authors": 4,
            "queries": 3,
            "variance_fits": 4,
            "query_covariance_fits": 12,
            "ridge_fits": 24,
            "runtime_replay_max": error,
            **record,
            "new_neural_calls": 0,
            "human_labels_read": False,
        },
    )
    names = [
        "private_membership.json",
        "private_predictions.npz",
        "private_source_fit.csv",
        "construction_summary.json",
    ] + [f"private_variance_{i}.npz" for i in indices]
    for q in m["queries"]:
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
            "seconds": round(time.monotonic() - start, 3),
            "new_neural_calls": 0,
        },
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="logic_query_residual_input_v1")
    construct(p.parse_args().name)
