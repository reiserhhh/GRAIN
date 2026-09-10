#!/usr/bin/env python3
"""NEXT54: frozen state/tangent crossings, then exact endpoint error accounting."""

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
from grain_research.vector_mechanisms.compiled_query_transfer import QUERIES, fidelity
from grain_research.vector_mechanisms.response_repair import CELLS, allocate, summarize
from grain_research.vector_mechanisms.selection_endpoint import key_vectors
from grain_research.vector_mechanisms.source_query_selection import evidence_inputs
from grain_research.vector_mechanisms.stage_file_guard import install
from scripts.run_grain_source_state_transport import (
    ARMS,
    BASE,
    ENDPOINT,
    arrays,
    forbidden,
    frozen,
    read,
    reconstruct,
    ref,
    sha,
    verify,
    verify_hashes,
    write,
)

OLD = BASE / "logic_response_aware_state_v1"
OBJECTIVES = ("normalized_value", "joint")
PLAN = ROOT / "docs/GRAIN_RESPONSE_REPAIR_MECHANISM_PLAN_V1.md"


def prepare(out):
    """Link only frozen numerical candidates; native outcomes are evaluation-only."""
    old = verify(OLD)
    jobs = read(OLD / "private_jobs.json")
    if len(jobs) != 12:
        raise ValueError("Exact twelve paired materials required")
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    files, inputs = [], [OLD / "manifest.json", PLAN, Path(__file__)]
    for name in (
        "private_inputs",
        "private_fixed",
        "private_weights.npz",
        "private_jobs.json",
    ):
        (out / name).symlink_to(OLD / name)
    files += [out / "private_weights.npz", out / "private_jobs.json"]
    files += [out / "private_fixed" / (j["key"] + ".npz") for j in jobs]
    files += [
        out / "private_inputs" / (k + ".npz") for j in jobs for k in j["query_keys"]
    ]
    for kind, oldkind in (
        ("private_candidates", "predictions"),
        ("private_maps", "models"),
    ):
        (out / kind).mkdir()
        for o in OBJECTIVES:
            f = frozen(OLD / o, oldkind)
            (out / kind / o).symlink_to(OLD / o / oldkind)
            files += [out / kind / o / n for n in f["files"]]
            inputs += [OLD / o / oldkind / "freeze.json"] + [
                OLD / o / oldkind / n for n in f["files"]
            ]
    inputs += [
        ROOT / "grain_research/vector_mechanisms/response_repair.py",
        ROOT / "tests/test_grain_response_repair.py",
        ROOT / "scripts/run_grain_source_state_transport.py",
    ]
    lineage = dict(old["inputs"])
    lineage.update({str(p.relative_to(ROOT)): sha(p) for p in inputs})
    m = {
        "protocol": "GRAIN_RESPONSE_REPAIR_V1",
        "max_seconds": 60,
        "max_cross_evaluations": 96,
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "prepare_seconds": time.monotonic() - started,
        "fits": 0,
        "native_calls": 0,
        "inputs": lineage,
        "prepared": {str(p.relative_to(out)): sha(p) for p in files},
    }
    write(out / "manifest.json", m)
    print({"status": "PREPARED", "seconds": m["prepare_seconds"]}, flush=True)


def predict(out, name):
    """Freeze all four cells without native current states, gradients or endpoints."""
    m = verify(out)
    jobs = read(out / "private_jobs.json")
    dst = out / name
    dst.mkdir(exist_ok=False)
    paths = [dst / (k + ".npz") for j in jobs for k in j["query_keys"]]
    reads = [out / n for n in m["prepared"]]
    denied = install(
        reads,
        paths + [dst / "freeze.json"],
        forbidden(),
        [ROOT / "grain_research", ROOT / "scripts"],
    )
    started = time.monotonic()
    w = arrays(out / "private_weights.npz")
    diag_error, cross_calls = 0.0, 0
    for j in jobs:
        fixed = arrays(out / "private_fixed" / (j["key"] + ".npz"))
        models = {
            o: {
                a: arrays(out / "private_maps" / o / f"{ref(j)}_{a}.npz")
                for a in ARMS[1:]
            }
            for o in OBJECTIVES
        }
        for qi, k in enumerate(j["query_keys"]):
            data = arrays(out / "private_inputs" / (k + ".npz"))
            packet = {
                o: arrays(out / "private_candidates" / o / (k + ".npz"))
                for o in OBJECTIVES
            }
            vv, aa = [], []
            for ai, family in enumerate(ARMS[1:], start=1):
                states = [
                    reconstruct(data, packet[o], models[o], ai) for o in OBJECTIVES
                ]
                vectors, actions = [], []
                for si in range(2):
                    vr, ar = [], []
                    for di in range(2):
                        x, d = states[si][0], states[di][1]
                        cap = {n: data[n] for n in ("cos", "sin", "mask")} | {
                            "x": x,
                            "d": d,
                        }
                        e = evidence_inputs(cap, w, np.zeros((2, x.shape[1])))
                        del e["g"]
                        u = key_vectors(fixed["q"][qi][None], e, w["o"])[0]
                        action = u @ fixed["r"][qi]
                        if si == di:
                            source = packet[OBJECTIVES[si]]
                            err = max(
                                float(np.max(abs(u - source["vectors"][ai]))),
                                float(np.max(abs(action - source["actions"][ai]))),
                            )
                            diag_error = max(diag_error, err)
                            if err > 1e-11:
                                raise ValueError("Frozen diagonal formula changed")
                        else:
                            cross_calls += 1
                        vr.append(u)
                        ar.append(action)
                    vectors.append(vr)
                    actions.append(ar)
                vv.append(vectors)
                aa.append(actions)
            np.savez(dst / (k + ".npz"), vectors=np.asarray(vv), actions=np.asarray(aa))
            if time.monotonic() - started + m["prepare_seconds"] > m["max_seconds"]:
                raise TimeoutError("Crossing budget exceeded")
    if cross_calls != 96:
        raise ValueError("Exact fixed cross inventory required")
    r = {
        "status": "COMPLETE",
        "seconds": time.monotonic() - started,
        "cross_evaluations": cross_calls,
        "diagonal_evaluations": 96,
        "diagonal_max_error": diag_error,
        "fits": 0,
        "native_calls": 0,
        "human_labels_read": False,
        "current_target_reads": 0,
        "denial_probes": 4,
        "extra_denials": len(denied) - 4,
        "files": {p.name: sha(p) for p in paths},
    }
    write(dst / "freeze.json", r)
    print({k: v for k, v in r.items() if k != "files"}, flush=True)


def effects(frame, groups):
    """Calculate effects after pairing queries, not by subtracting squared errors."""
    rows = []
    for key, g in frame.groupby(groups, sort=True):
        if len(g) != 4 or set(g.cell) != set(CELLS) or g.actual.nunique() != 1:
            raise ValueError(
                "Four exactly paired cells with one actual target required"
            )
        a = allocate(
            g.set_index("cell").loc[list(CELLS), "predicted"].to_numpy().reshape(2, 2),
            np.asarray(g.actual.iloc[0]),
        )
        rows.append(
            dict(zip(groups, key, strict=True)) | {k: float(v) for k, v in a.items()}
        )
    return pd.DataFrame(rows)


def evaluate(out, name, predictions):
    """Open targets only after the crossing freeze and keep all signed allocations."""
    verify(out)
    frozen(out, predictions)
    prior = read(ENDPOINT / "analysis/summary.json")
    verify_hashes(ENDPOINT / "analysis", prior["files"])
    target = pd.read_csv(ENDPOINT / "analysis/private_actions.csv")
    target = target[target.arm == "source_own_Q"].set_index(
        ["author_key", "query_name", "factor"]
    )
    if len(target) != 96 or not target.index.is_unique:
        raise ValueError("Unique native endpoint targets required")
    dst = out / name
    dst.mkdir(exist_ok=False)
    rows = []
    for j in read(out / "private_jobs.json"):
        for q, k in zip(QUERIES, j["query_keys"], strict=True):
            p = arrays(out / predictions / (k + ".npz"))["actions"]
            if p.shape != (2, 2, 2, 4):
                raise ValueError("Fixed family/state/tangent/factor inventory required")
            for ai, family in enumerate(ARMS[1:]):
                for ci, cell in enumerate(CELLS):
                    for f in range(4):
                        factor = f"whole:F{f + 1:02d}"
                        rows.append(
                            {
                                "author_key": j["key"],
                                "cohort": j["cohort"],
                                "family": family,
                                "query_name": q,
                                "cell": cell,
                                "factor": factor,
                                "actual": float(
                                    target.loc[(j["key"], q, factor), "actual"]
                                ),
                                "predicted": float(p[ai, ci // 2, ci % 2, f]),
                            }
                        )
    df = pd.DataFrame(rows)
    idx = ["author_key", "cohort", "family", "cell", "factor"]
    cf = (
        df[df.query_name == QUERIES[0]].set_index(idx)[["actual", "predicted"]]
        - df[df.query_name == QUERIES[1]].set_index(idx)[["actual", "predicted"]]
    ).reset_index()
    if len(df) != 768 or len(cf) != 384 or cf.isna().any().any():
        raise ValueError("Cross-cell paired query inventory mismatch")
    er = effects(df, ["author_key", "cohort", "family", "query_name", "factor"])
    ec = effects(cf, ["author_key", "cohort", "family", "factor"])
    files = []

    def save(tab, filename):
        """Write deterministic private or aggregate tables for replay."""
        tab.to_csv(dst / filename, index=False)
        files.append(filename)

    for tab, n in (
        (df, "private_cells.csv"),
        (cf, "private_contrast_cells.csv"),
        (er, "private_effects.csv"),
        (ec, "private_contrast_effects.csv"),
    ):
        save(tab, n)
    for tab, groups, n in (
        (df, ["cohort", "family", "query_name", "cell"], "query_cells.csv"),
        (df, ["cohort", "family", "query_name", "cell", "factor"], "factor_cells.csv"),
        (cf, ["cohort", "family", "cell"], "contrast_cells.csv"),
        (cf, ["cohort", "family", "cell", "factor"], "contrast_factor_cells.csv"),
    ):
        save(
            pd.DataFrame(
                [
                    dict(zip(groups, key, strict=True))
                    | {"authors": g.author_key.nunique()}
                    | fidelity(g.actual.to_numpy(), g.predicted.to_numpy())
                    for key, g in tab.groupby(groups, sort=True)
                ]
            ),
            n,
        )
    max_alloc, max_mse = 0.0, 0.0
    fields = list(allocate(np.zeros((2, 2)), np.asarray(0.0)))
    for tab, groups, n in (
        (er, ["cohort", "family", "query_name"], "query_effects.csv"),
        (er, ["cohort", "family", "query_name", "factor"], "factor_effects.csv"),
        (ec, ["cohort", "family"], "contrast_effects.csv"),
        (ec, ["cohort", "family", "factor"], "contrast_factor_effects.csv"),
    ):
        values = []
        for key, g in tab.groupby(groups, sort=True):
            r = summarize({k: g[k].to_numpy() for k in fields})
            max_alloc = max(max_alloc, r["allocation_max_error"])
            max_mse = max(max_mse, r["mse_accounting_error"])
            values.append(
                dict(zip(groups, key, strict=True))
                | {"authors": g.author_key.nunique()}
                | r
            )
        save(pd.DataFrame(values), n)
    if max_alloc > 1e-11 or max_mse > 1e-11:
        raise ValueError("Allocation/endpoint MSE identity failed")
    r = {
        "status": "COMPLETE",
        "cells": len(df),
        "contrasts": len(cf),
        "effects": len(er),
        "contrast_effects": len(ec),
        "allocation_max_error": max_alloc,
        "mse_max_error": max_mse,
        "fits": 0,
        "native_calls": 0,
        "human_labels_read": False,
        "files": {n: sha(dst / n) for n in files},
    }
    write(dst / "summary.json", r)
    print({k: v for k, v in r.items() if k != "files"}, flush=True)


def main():
    """Separate frozen construction, crossing and target evaluation into processes."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "predict", "evaluate"))
    p.add_argument("--output", type=Path, default=BASE / "logic_response_repair_v1")
    p.add_argument("--name")
    p.add_argument("--predictions", default="predictions")
    a = p.parse_args()
    with threadpool_limits(limits=2):
        if a.stage == "prepare":
            prepare(a.output)
        elif a.stage == "predict":
            predict(a.output, a.name or "predictions")
        else:
            evaluate(a.output, a.name or "analysis", a.predictions)


if __name__ == "__main__":
    main()
