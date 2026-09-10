#!/usr/bin/env python3
"""Publish NEXT65 and a separate no-fit paired average/difference objective account."""

import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_grain_full_context_geometry import (
    ARM,
    OUT,
    manifest,
    model_path,
    old,
    previous,
)


def paired_channel_account(target0, target1, pred0, pred1, scale):
    """Exact equal-author parallelogram decomposition; no refit or weight selection."""
    n = len(target0)
    if not n or any(len(v) != n for v in (target1, pred0, pred1)) or scale <= 0:
        raise ValueError("Aligned paired arrays and positive original scale required")
    energies = []
    errors = []
    for a, b, p, q in zip(target0, target1, pred0, pred1, strict=True):
        a, b, p, q = [np.asarray(x, float) for x in (a, b, p, q)]
        if len({x.shape for x in (a, b, p, q)}) != 1:
            raise ValueError("Pair shape mismatch")
        m, d = (a + b) / 2, (b - a) / 2
        pm, pd = (p + q) / 2, (q - p) / 2
        energies.append([np.mean(m * m), np.mean(d * d)])
        errors.append(
            [
                np.mean((pm - m) ** 2),
                np.mean((pd - d) ** 2),
                0.5 * (np.mean((p - a) ** 2) + np.mean((q - b) ** 2)),
            ]
        )
    energy = np.mean(energies, axis=0) / scale**2
    loss = np.mean(errors, axis=0) / scale**2
    return {
        "target_average_energy": float(energy[0]),
        "target_half_difference_energy": float(energy[1]),
        "target_difference_fraction": float(energy[1] / energy.sum()),
        "average_loss": float(loss[0]),
        "half_difference_loss": float(loss[1]),
        "pair_loss": float(loss[2]),
        "half_difference_sse_over_zero": float(loss[1] / energy[1])
        if energy[1] > 0
        else float("nan"),
        "difference_fraction_of_pair_loss": float(loss[1] / loss[2])
        if loss[2] > 0
        else float("nan"),
        "parallelogram_error": float(abs(loss[0] + loss[1] - loss[2])),
    }


def source_account():
    """Inspect existing source views only; diagnostic added after primary results."""
    start = time.process_time()
    m = manifest(OUT)
    jobs = sorted(
        [j for j in old.read(OUT / "private_jobs.json") if j["cohort"] == "source4"],
        key=lambda j: j["index"],
    )
    old.frozen(OUT, "models")
    old.frozen(OUT, "predictions")
    old.frozen(previous.OUT, "predictions")
    paths = [
        p
        for j in jobs
        for k in j["query_keys"]
        for p in (
            old.input_path(k),
            old.gradient_path(k),
            previous.intervention.OUT / "capture" / (k + ".npz"),
        )
    ]
    old.check_paths(m, paths)
    records = []
    for qi, q in enumerate(old.QUERIES):
        model = old.arrays(model_path(OUT, q))
        t0 = []
        t1 = []
        pred = {a: [[], []] for a in (*previous.ARMS, ARM)}
        for j in jobs:
            k = j["query_keys"][qi]
            _, h, _ = old.pure_input(k)
            grads = [
                old.selected(p, ("body_gradient",))["body_gradient"].astype(float)
                for p in (
                    old.gradient_path(k),
                    previous.intervention.OUT / "capture" / (k + ".npz"),
                )
            ]
            t0.append(len(h) * (grads[0] @ model["basis"]) * model["scale"])
            t1.append(len(h) * (grads[1] @ model["basis"]) * model["scale"])
            for vi, v in enumerate(("original", "shifted")):
                before = old.arrays(
                    previous.OUT / "predictions" / (k + "_" + v + ".npz")
                )
                for ai, a in enumerate(previous.ARMS):
                    pred[a][vi].append(before["density"][ai])
                pred[ARM][vi].append(
                    old.arrays(OUT / "predictions" / (k + "_" + v + ".npz"))["density"]
                )
        for a, (p0, p1) in pred.items():
            records.append(
                {
                    "query_name": q,
                    "candidate": a,
                    "source_authors": 4,
                    **paired_channel_account(
                        t0, t1, p0, p1, float(model["density_scale"])
                    ),
                }
            )
    return pd.DataFrame(records), time.process_time() - start


def publish():
    """Copy verified aggregates only, with post-outcome analysis clearly separated."""
    audit = old.read(OUT / "audit.json")
    summary = old.read(OUT / "analysis/summary.json")
    if audit["status"] != "PASS":
        raise ValueError("Primary audit required")
    dst = ROOT / "reports/grain_vector_mechanisms/full_context_geometry_v1"
    dst.mkdir(exist_ok=True)
    files = {}
    for name, digest in summary["files"].items():
        if name.startswith("private_"):
            continue
        source = OUT / "analysis" / name
        if old.sha(source) != digest:
            raise ValueError("Aggregate changed")
        if set(pd.read_csv(source, nrows=0).columns) & {
            "author_key",
            "text",
            "user_id",
            "tau",
        }:
            raise ValueError("Private data in publication")
        shutil.copyfile(source, dst / name)
        files[name] = old.sha(dst / name)
    table, cpu = source_account()
    table.to_csv(dst / "paired_objective_account.csv", index=False)
    files["paired_objective_account.csv"] = old.sha(
        dst / "paired_objective_account.csv"
    )
    old.write(
        dst / "publication.json",
        {
            "primary_commit": old.read(OUT / "manifest.json")["commit"],
            "audit": audit,
            "files": files,
            "promotion": summary["promotion"],
            "criteria": summary["criteria"],
            "post_outcome_analysis": "Source paired-objective arithmetic only; zero fits, no new8 diagnostic selection",
            "post_outcome_cpu_seconds": cpu,
            "post_outcome_code": {
                str(p.relative_to(ROOT)): old.sha(p)
                for p in (
                    Path(__file__),
                    ROOT / "tests/test_grain_paired_objective_account.py",
                )
            },
            "scope": "source4 in-fit/exposed new8 source-disjoint; model L and GRAIN actions, no humanY",
        },
    )
    print({"public_tables": len(files), "post_account_cpu": cpu, "directory": str(dst)})


if __name__ == "__main__":
    publish()
