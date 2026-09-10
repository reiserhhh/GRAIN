#!/usr/bin/env python3
"""NEXT36: exchange frozen query input directions in identical source folds."""

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
from grain_research.vector_mechanisms.query_direction_transfer import (
    fit_borrowed,
    subspace_geometry,
)
from grain_research.vector_mechanisms.query_residual_input import (
    coordinate_support,
    predict_extension,
)
from scripts.analyze_grain_query_residual_input import (
    verify_predictions as prior_verified,
)
from scripts.run_grain_content_coefficients import NATIVE
from scripts.run_grain_query_residual_input import SOURCE as CONTENT
from scripts.run_grain_real_author_path import BASE, read, sha, write

SOURCE = BASE / "logic_query_residual_input_v1_r2"
PLAN = ROOT / "docs/GRAIN_QUERY_DIRECTION_TRANSFER_PLAN_V1.md"


def prepare(out):
    """Bind source models, input packs, role history and unchanged numerical code."""
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
        ROOT / "grain_research/vector_mechanisms/query_direction_transfer.py",
        ROOT / "grain_research/vector_mechanisms/query_residual_input.py",
        ROOT / "grain_research/vector_mechanisms/content_coefficients.py",
    ]
    out.mkdir(exist_ok=False)
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_QUERY_DIRECTION_TRANSFER_V1",
            "code_hashes": {str(p.relative_to(ROOT)): sha(p) for p in code},
            "inputs": assets,
            "queries": old["queries"],
            "extra_rank": 8,
            "alpha": 0.01,
            "max_seconds": 240,
            "threads": 4,
            "new_neural_calls": 0,
            "human_labels_read": False,
        },
    )
    write(out / "private_membership.json", read(SOURCE / "private_membership.json"))


def verified(out):
    """Reject changed dependencies before fitting, replay or evaluation."""
    m = read(out / "manifest.json")
    for key in ("code_hashes", "inputs"):
        for p, digest in m[key].items():
            if sha(ROOT / p) != digest:
                raise ValueError("Frozen dependency changed: " + Path(p).name)
    return m


def construct(name):
    """Fit only off-diagonal readers, then freeze target-denied held predictions."""
    out = BASE / name
    if not out.exists():
        prepare(out)
    m = verified(out)
    if (out / "prediction_freeze.json").exists():
        raise ValueError("Complete packet exists; choose a new replay name")
    membership = read(out / "private_membership.json")
    indices = [r["index"] for r in membership["authors"]]
    roles = {r["held"]: r for r in membership["folds"]}
    if len(indices) != 4 or len(set(indices)) != 4 or set(roles) != set(indices):
        raise ValueError("Four old source/held roles required")
    packets, bases, donors = {}, {}, {}
    for i in indices:
        with np.load(CONTENT / f"private_inputs_{i}.npz") as z:
            packets[i] = (z["hidden"], z["grain"])
        for q in m["queries"]:
            with np.load(CONTENT / q / f"private_model_{i}_content.npz") as z:
                bases[q, i] = dict(z)
            with np.load(SOURCE / q / f"private_model_{i}_residual_query.npz") as z:
                donors[q, i] = dict(z)
    start = time.monotonic()
    jobs, records = [], []
    with threadpool_limits(limits=4):
        geometry = []
        for i in indices:
            for q in m["queries"]:
                for p in m["queries"]:
                    if p != q:
                        geometry.append(
                            {
                                "query": q,
                                "donor": p,
                                "held": i,
                                **subspace_geometry(
                                    donors[q, i]["extra_basis"],
                                    donors[p, i]["extra_basis"],
                                ),
                            }
                        )
        geometry_path = out / "private_geometry.csv"
        geometry_text = pd.DataFrame(geometry).to_csv(index=False)
        if geometry_path.exists():
            if geometry_path.read_text() != geometry_text:
                raise ValueError("Source geometry changed")
        else:
            geometry_path.write_text(geometry_text)
        write(out / "geometry_freeze.json", {"sha256": sha(geometry_path)})
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
                        out / "run_status.json", {"status": "PAUSED_RESOURCE_BOUNDARY"}
                    )
                    return
                folder.mkdir()
                source_rows = []
                for i in indices:
                    src = roles[i]["source"]
                    js = []
                    for s in src:
                        with np.load(
                            NATIVE / "private_packets" / f"a{s:02d}_{q}.npz"
                        ) as z:
                            js.append(z["body_gradient"].astype(float))
                    for p in m["queries"]:
                        if p == q:
                            continue
                        model, record = fit_borrowed(
                            [packets[s][0] for s in src],
                            [packets[s][1] for s in src],
                            js,
                            bases[q, i],
                            donors[p, i],
                            roles[i],
                            roles[i],
                            src,
                            alpha=m["alpha"],
                        )
                        np.savez_compressed(
                            folder / f"private_model_{i}_from_{p}.npz", **model
                        )
                        source_rows.append(
                            {
                                "query": q,
                                "donor": p,
                                "held": i,
                                **record,
                                **coordinate_support(packets[i][0], model),
                            }
                        )
                pd.DataFrame(source_rows).to_csv(
                    folder / "private_source_fit.csv", index=False
                )
                names = [
                    f"private_model_{i}_from_{p}.npz"
                    for i in indices
                    for p in m["queries"]
                    if p != q
                ] + ["private_source_fit.csv"]
                write(
                    folder / "freeze.json",
                    {"files": {p: sha(folder / p) for p in names}},
                )
            records.append(pd.read_csv(folder / "private_source_fit.csv"))
            for i in indices:
                for p in m["queries"]:
                    if p != q:
                        jobs.append(
                            {
                                "key": f"{q}__a{i}__from_{p}",
                                "model": str(
                                    folder / f"private_model_{i}_from_{p}.npz"
                                ),
                                "inputs": str(CONTENT / f"private_inputs_{i}.npz"),
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
        replay = json.loads(run.stdout)
        max_error = 0.0
        with np.load(out / "private_predictions.npz") as yp:
            for job in jobs:
                with np.load(job["model"]) as z:
                    model = dict(z)
                with np.load(job["inputs"]) as z:
                    pred = predict_extension(z["hidden"], z["grain"], model)
                max_error = max(max_error, float(np.max(abs(pred - yp[job["key"]]))))
        if max_error != 0 or replay != {
            "jobs": 24,
            "denial_probes": 2,
            "extra_denials": 0,
        }:
            raise ValueError("Independent target-denied replay failed")
    pd.concat(records, ignore_index=True).to_csv(
        out / "private_source_fit.csv", index=False
    )
    write(
        out / "construction_summary.json",
        {
            "status": "COMPLETE",
            "authors": 4,
            "queries": 3,
            "ridge_fits": 24,
            "new_basis_fits": 0,
            "ordered_geometry_pairs": 24,
            "runtime_replay_max": max_error,
            **replay,
            "new_neural_calls": 0,
            "human_labels_read": False,
        },
    )
    names = [
        "private_membership.json",
        "private_predictions.npz",
        "private_geometry.csv",
        "private_source_fit.csv",
        "construction_summary.json",
    ]
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
    status = {
        "status": "COMPLETE",
        "seconds": time.monotonic() - start,
        "new_neural_calls": 0,
    }
    write(out / "run_status.json", status)
    print(status, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="logic_query_direction_transfer_v1")
    construct(parser.parse_args().name)
