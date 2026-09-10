#!/usr/bin/env python3
"""NEXT50: fixed source-query anchors selecting current evidence, zero model calls."""

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
from grain_research.vector_mechanisms.compiled_query_transfer import (
    MODES,
    QUERIES,
    fidelity,
)
from grain_research.vector_mechanisms.source_query_selection import (
    ARMS,
    anchors,
    evidence_inputs,
    key_actions,
    source_query,
)
from grain_research.vector_mechanisms.stage_file_guard import install
from scripts.run_grain_compiled_query_transfer import (
    BASE,
    arrays,
    read,
    sha,
    verify,
    verify_hashes,
    write,
)
from scripts.run_grain_upstream_attention_channels import (
    completed_pairs,
    verify_evaluation,
)

SOURCE = BASE / "logic_upstream_attention_channels_v1"
PLAN = ROOT / "docs/GRAIN_SOURCE_QUERY_SELECTION_PLAN_V1.md"


def prepare(out):
    """Export pure current evidence and source-only query bank; no target actions."""
    verify_evaluation(SOURCE)
    jobs = completed_pairs(SOURCE)
    if (
        len(jobs) != 12
        or read(SOURCE / "capture_freeze.json")["runtime"]["status"] != "COMPLETE"
    ):
        raise ValueError("Complete twelve-pair NEXT49 capture required")
    out.mkdir(exist_ok=False)
    (out / "private_inputs").mkdir()
    weights = arrays(SOURCE / "private_weights.npz")
    if int(weights["block"]) != 22:
        raise ValueError("Actual attention22 weights required")
    prepared = []
    lineage = [
        SOURCE / "manifest.json",
        SOURCE / "capture_freeze.json",
        SOURCE / "analysis_code_amendment.json",
        SOURCE / "private_weights.npz",
    ]
    bank = {"index": [], "query": [], "q": []}
    start = time.monotonic()
    for j in jobs:
        shared_path = SOURCE / "private_readers" / (j["key"] + ".npz")
        shared = arrays(shared_path)["shared"]
        lineage.append(shared_path)
        for q, key in zip(QUERIES, j["query_keys"], strict=True):
            src = SOURCE / "private_capture" / (key + ".npz")
            capture = arrays(src)
            dst = out / "private_inputs" / (key + ".npz")
            np.savez(dst, **evidence_inputs(capture, weights, shared))
            prepared.append(dst)
            lineage.append(src)
            if j["cohort"] == "source4":
                bank["index"].append(j["index"])
                bank["query"].append(q)
                bank["q"].append(source_query(capture["x"][-1], weights))
    np.savez(
        out / "private_source_bank.npz", **{k: np.asarray(v) for k, v in bank.items()}
    )
    write(out / "private_jobs.json", jobs)
    prepared.extend([out / "private_source_bank.npz", out / "private_jobs.json"])
    code = [PLAN, Path(__file__), ROOT / "tests/test_grain_source_query_selection.py"]
    code += [
        ROOT / "grain_research/vector_mechanisms" / (n + ".py")
        for n in (
            "source_query_selection",
            "upstream_attention_channels",
            "fixed_seed_attention",
            "compiled_query_transfer",
            "stage_file_guard",
        )
    ]
    code += [
        ROOT / "scripts" / (n + ".py")
        for n in (
            "run_grain_upstream_attention_channels",
            "run_grain_compiled_query_transfer",
        )
    ]
    seconds = time.monotonic() - start
    if seconds > 60:
        raise TimeoutError("Preparation numeric cap exceeded")
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_SOURCE_QUERY_SELECTION_V1",
            "block": 22,
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "prepare_seconds": seconds,
            "max_seconds": 60,
            "native_calls": 0,
            "fits": 0,
            "human_labels_read": False,
            "current_Q_exported": False,
            "target_fields_copied": False,
            "inputs": {
                str(p.relative_to(ROOT)): sha(p) for p in sorted(set(code + lineage))
            },
            "prepared": {str(p.relative_to(out)): sha(p) for p in prepared},
        },
    )
    print(
        {"status": "PREPARED", "seconds": seconds, "source_Q_records": len(bank["q"])},
        flush=True,
    )


def predict(out, name):
    """Freeze all three source-only anchor predictions without target/current-Q access."""
    m = verify(out)
    jobs = read(out / "private_jobs.json")
    dst = out / name
    dst.mkdir(exist_ok=False)
    reads = [out / "private_source_bank.npz"] + [
        out / "private_inputs" / (k + ".npz") for j in jobs for k in j["query_keys"]
    ]
    files = [dst / (j["key"] + ".npz") for j in jobs]
    probes = [
        SOURCE / "analysis_r2/private_actions.csv",
        SOURCE / "analysis_r2/private_contrasts.csv",
        SOURCE / "private_capture/new8_a00_neuroticism.npz",
        SOURCE / "predictions/new8_a00.npz",
    ]
    denied = install(
        reads,
        files + [dst / "freeze.json"],
        probes,
        [ROOT / "grain_research", ROOT / "scripts"],
    )
    start = time.monotonic()
    bank = arrays(reads[0])
    checks = []
    for j, path in zip(jobs, files, strict=True):
        if time.monotonic() - start > m["max_seconds"]:
            raise TimeoutError("Numeric cap reached")
        values = []
        vectors = []
        for q, key in zip(QUERIES, j["query_keys"], strict=True):
            anchor, members = anchors(bank, j["cohort"], j["index"], q)
            values.append(
                key_actions(anchor, arrays(out / "private_inputs" / (key + ".npz")))
            )
            vectors.append(anchor)
            checks.append({"author_key": j["key"], "query_name": q, "sources": members})
        a = np.stack(values)
        np.savez_compressed(
            path, actions=a, contrasts=a[0] - a[1], anchors=np.stack(vectors)
        )
    seconds = time.monotonic() - start
    if seconds > m["max_seconds"]:
        raise TimeoutError("Numeric cap exceeded")
    r = {
        "status": "COMPLETE",
        "seconds": seconds,
        "native_calls": 0,
        "fits": 0,
        "human_labels_read": False,
        "target_reads": 0,
        "current_Q_reads": 0,
        "denial_probes": 4,
        "extra_denials": len(denied) - 4,
        "checks": checks,
        "files": {p.name: sha(p) for p in files},
    }
    write(dst / "freeze.json", r)
    print({k: v for k, v in r.items() if k not in ("checks", "files")}, flush=True)


def evaluate(out, name, predictions):
    """Compare frozen anchors with verified NEXT49 key targets using safe identity names."""
    verify(out)
    jobs = read(out / "private_jobs.json")
    fr = read(out / predictions / "freeze.json")
    if fr["status"] != "COMPLETE":
        raise ValueError("Complete prediction freeze required")
    verify_hashes(out / predictions, fr["files"])
    sf = read(SOURCE / "predictions/prediction_freeze.json")
    verify_hashes(SOURCE / "predictions", sf["files"])
    dst = out / name
    dst.mkdir(exist_ok=False)
    rows = []
    contrasts = []
    for j in jobs:
        p = arrays(out / predictions / (j["key"] + ".npz"))
        target = arrays(SOURCE / "predictions" / (j["key"] + ".npz"))["actions"][
            :, :, 1, :
        ]
        for a, arm in enumerate(ARMS):
            for m, mode in enumerate(MODES):
                for f in range(4):
                    meta = {
                        "author_key": j["key"],
                        "cohort": j["cohort"],
                        "arm": arm,
                        "mode": mode,
                        "factor": f"whole:F{f + 1:02d}",
                    }
                    contrasts.append(
                        meta
                        | {
                            "actual": float(target[0, m, f] - target[1, m, f]),
                            "predicted": float(p["contrasts"][a, m, f]),
                        }
                    )
                    for q, query in enumerate(QUERIES):
                        rows.append(
                            meta
                            | {
                                "query_name": query,
                                "actual": float(target[q, m, f]),
                                "predicted": float(p["actions"][q, a, m, f]),
                            }
                        )
    df = pd.DataFrame(rows)
    cf = pd.DataFrame(contrasts)
    if (
        len(df) != 576
        or len(cf) != 288
        or df.author_key.nunique() != 12
        or set(df.query_name) != set(QUERIES)
    ):
        raise ValueError("Fixed inventory/identity mismatch")
    if df.duplicated(["author_key", "query_name", "mode", "factor", "arm"]).any():
        raise ValueError("Duplicate action")
    df.to_csv(dst / "private_actions.csv", index=False)
    cf.to_csv(dst / "private_contrasts.csv", index=False)
    names = ["private_actions.csv", "private_contrasts.csv"]
    for frame, groups, filename in (
        (df, ["cohort", "mode", "arm", "query_name"], "query_summary.csv"),
        (df, ["cohort", "mode", "arm", "query_name", "factor"], "factor_summary.csv"),
        (cf, ["cohort", "mode", "arm"], "contrast_summary.csv"),
        (cf, ["cohort", "mode", "arm", "factor"], "contrast_factors.csv"),
    ):
        result = []
        for key, g in frame.groupby(groups, sort=True):
            result.append(
                dict(zip(groups, key, strict=True))
                | {"authors": g.author_key.nunique()}
                | fidelity(g.actual.to_numpy(), g.predicted.to_numpy())
            )
        pd.DataFrame(result).to_csv(dst / filename, index=False)
        names.append(filename)
    summary = pd.read_csv(dst / "query_summary.csv")
    new = summary[(summary.cohort == "new8") & (summary["mode"] == "own")]
    retained = {
        arm: bool((new[new.arm == arm].action_mse_over_zero < 1).all()) for arm in ARMS
    }
    if any(len(new[new.arm == arm]) != 2 for arm in ARMS):
        raise ValueError("Both recipient queries required")
    r = {
        "status": "COMPLETE",
        "actions": len(df),
        "contrasts": len(cf),
        "retained_anchors": retained,
        "native_calls": 0,
        "fits": 0,
        "human_labels_read": False,
        "files": {n: sha(dst / n) for n in names},
    }
    write(dst / "summary.json", r)
    print({k: v for k, v in r.items() if k != "files"}, flush=True)


def main():
    """Run preparation, guarded prediction or frozen-target evaluation separately."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "predict", "evaluate"))
    p.add_argument(
        "--output", type=Path, default=BASE / "logic_source_query_selection_v1"
    )
    p.add_argument("--name", default="predictions")
    p.add_argument("--predictions", default="predictions")
    a = p.parse_args()
    with threadpool_limits(limits=2):
        if a.stage == "prepare":
            prepare(a.output)
        elif a.stage == "predict":
            predict(a.output, a.name)
        else:
            evaluate(a.output, a.name, a.predictions)


if __name__ == "__main__":
    main()
