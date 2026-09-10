#!/usr/bin/env python3
"""NEXT47: fixed source-common MLP24 reader, no native executions or current x."""

from __future__ import annotations

import argparse
import shutil
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
from grain_research.vector_mechanisms.source_common_counteraction import (
    apply_readers,
    compile_readers,
    direct_actions,
    errors,
    select_contexts,
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

SOURCE = BASE / "logic_query_counteraction_v1"
NATIVE = BASE / "logic_layer_accumulation_v1"
PLAN = ROOT / "docs/GRAIN_SOURCE_COMMON_COUNTERACTION_PLAN_V1.md"


def prepare(out):
    """Copy pure d/w and source4 x; never open new8 x or local output targets."""
    m, cf = read(SOURCE / "manifest.json"), read(SOURCE / "capture_freeze.json")
    if cf["runtime"]["status"] != "COMPLETE" or cf["block"] != 24:
        raise ValueError("Complete actual block24 capture required")
    jp = SOURCE / "private_jobs.json"
    if sha(jp) != m["prepared"][jp.name]:
        raise ValueError("Job inventory changed")
    jobs = read(jp)
    expected = {(c, i) for c, n in (("source4", 4), ("new8", 8)) for i in range(n)}
    if len(jobs) != 12 or {(j["cohort"], j["index"]) for j in jobs} != expected:
        raise ValueError("Exact12-author inventory required")
    out.mkdir(exist_ok=False)
    (out / "private_inputs").mkdir()
    lineage = [SOURCE / "manifest.json", SOURCE / "capture_freeze.json", jp]
    prepared, xs, indices, queries = [], [], [], []
    for j in jobs:
        if j["key"] != f"{j['cohort']}_a{j['index']:02d}" or j["query_keys"] != [
            j["key"] + "_" + q for q in QUERIES
        ]:
            raise ValueError("Frozen author/query addresses differ")
        p = SOURCE / "private_prefix" / (j["key"] + ".npz")
        if sha(p) != m["prepared"][str(p.relative_to(SOURCE))]:
            raise ValueError("Pure input changed")
        a = arrays(p)
        if set(a) != {"d", "shared"}:
            raise ValueError("Forbidden or missing input field")
        dst = out / "private_inputs" / p.name
        shutil.copy2(p, dst)
        prepared.append(dst)
        lineage.append(p)
        if j["cohort"] != "source4":
            continue
        for q, key in zip(QUERIES, j["query_keys"], strict=True):
            if key != j["key"] + "_" + q:
                raise ValueError("Query pairing differs")
            p = SOURCE / "private_capture" / (key + ".npz")
            if sha(p) != cf["files"][str(p.relative_to(SOURCE))]:
                raise ValueError("Source x changed")
            x = arrays(p)
            if set(x) != {"x"}:
                raise ValueError("Source input schema differs")
            xs.append(x["x"])
            indices.append(j["index"])
            queries.append(q)
            lineage.append(p)
    bank = out / "private_source_bank.npz"
    np.savez_compressed(
        bank, x=np.asarray(xs), index=np.asarray(indices), query=np.asarray(queries)
    )
    prepared.append(bank)
    wp = SOURCE / "private_weights.npz"
    if sha(wp) != cf["weights_sha256"]:
        raise ValueError("Block24 weights changed")
    shutil.copy2(wp, out / wp.name)
    prepared.append(out / wp.name)
    lineage.append(wp)
    write(out / "private_jobs.json", jobs)
    prepared.append(out / "private_jobs.json")
    code = [
        PLAN,
        Path(__file__),
        ROOT / "tests/test_grain_source_common_counteraction.py",
        ROOT / "scripts/run_grain_compiled_query_transfer.py",
    ]
    code += [
        ROOT / "grain_research/vector_mechanisms" / (n + ".py")
        for n in (
            "source_common_counteraction",
            "compiled_gated_readout",
            "compiled_query_transfer",
            "gated_update",
            "stage_file_guard",
        )
    ]
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_SOURCE_COMMON_COUNTERACTION_V1",
            "block": 24,
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "max_seconds": 60,
            "model_sha256": m["model_sha256"],
            "new8_x_read": False,
            "native_targets_read": False,
            "fits": 0,
            "inputs": {
                str(p.relative_to(ROOT)): sha(p) for p in sorted(set(lineage + code))
            },
            "prepared": {str(p.relative_to(out)): sha(p) for p in prepared},
        },
    )
    print({"status": "PREPARED", "pairs": 12, "source_x": 8, "current_x_copied": False})


def compile_stage(out, name):
    """Compile only source-context adjoints, then apply using a d/v-only inner function."""
    m = verify(out)
    jobs = read(out / "private_jobs.json")
    dst = out / name
    dst.mkdir(exist_ok=False)
    wp, bp = out / "private_weights.npz", out / "private_source_bank.npz"
    writes = [dst / (j["key"] + ".npz") for j in jobs] + [dst / "freeze.json"]
    reads = [wp, bp] + [out / "private_inputs" / (j["key"] + ".npz") for j in jobs]
    probes = [
        SOURCE / "private_capture/new8_a00_neuroticism.npz",
        SOURCE / "private_predictions/new8_a00_neuroticism.npz",
        SOURCE / "analysis/private_actions.csv",
        NATIVE / "private_predictions/new8_a00_neuroticism.npz",
    ]
    denials = install(
        reads, writes, probes, [ROOT / "grain_research", ROOT / "scripts"]
    )
    start = time.monotonic()
    weights = {k: np.asarray(v, float) for k, v in arrays(wp).items()}
    bank, records = arrays(bp), []
    for j in jobs:
        if time.monotonic() - start > m["max_seconds"]:
            raise TimeoutError("Fixed60s numeric cap reached")
        x, members = select_contexts(bank, j["cohort"], j["index"])
        a = arrays(out / "private_inputs" / (j["key"] + ".npz"))
        v = compile_readers(x, a["shared"], weights)
        pred, contrast = apply_readers(v, a["d"])
        direct = direct_actions(x, a["shared"], a["d"], weights)
        err = float(abs(pred - direct).max())
        if err > 1e-10:
            raise ValueError("Source compilation identity failed")
        f = dst / (j["key"] + ".npz")
        np.savez_compressed(f, v=v, actions=pred, contrast=contrast)
        records.append(
            {"key": j["key"], "source_indices": members, "compile_error": err}
        )
    seconds = time.monotonic() - start
    if seconds > m["max_seconds"]:
        raise TimeoutError("Numeric cap exceeded")
    write(
        dst / "freeze.json",
        {
            "status": "COMPLETE",
            "seconds": seconds,
            "native_calls": 0,
            "fits": 0,
            "target_reads": 0,
            "new8_x_reads": 0,
            "denial_probes": len(probes),
            "extra_denials": len(denials) - len(probes),
            "records": records,
            "files": {p.name: sha(p) for p in writes if p.suffix == ".npz"},
        },
    )
    print(
        {
            "status": "FROZEN",
            "seconds": seconds,
            "actions": 192,
            "contrasts": 96,
            "compile_error": max(r["compile_error"] for r in records),
        }
    )


def score_summary(g, target, prediction):
    """Finite descriptive comparisons keep amplitude, baseline and signed action."""
    y, p = g[target].to_numpy(), g[prediction].to_numpy()
    return fidelity(y, p) | {
        "authors": g.key.nunique(),
        "target_mean": float(y.mean()),
        "prediction_mean": float(p.mean()),
        "final_T_projection": float(p @ g.final_T / (g.final_T @ g.final_T)),
    }


def evaluate(out, name, compiled):
    """Evaluate frozen values; retain native parity separately from the two model errors."""
    verify(out)
    freeze = read(out / compiled / "freeze.json")
    verify_hashes(out / compiled, freeze["files"])
    old = read(SOURCE / "analysis/summary.json")
    verify_hashes(SOURCE / "analysis", old["files"])
    targets = pd.read_csv(SOURCE / "analysis/private_actions.csv")
    nf = read(NATIVE / "prediction_freeze.json")
    dst = out / name
    dst.mkdir(exist_ok=False)
    rows, actions = [], []
    for j in read(out / "private_jobs.json"):
        p = arrays(out / compiled / (j["key"] + ".npz"))
        natives = []
        shared = arrays(out / "private_inputs" / (j["key"] + ".npz"))["shared"]
        for q, key in enumerate(j["query_keys"]):
            f = NATIVE / "private_predictions" / (key + ".npz")
            if sha(f) != nf["files"][str(f.relative_to(NATIVE))]:
                raise ValueError("Target changed")
            natives.append(np.einsum("mh,fh->mf", shared, arrays(f)["updates"][:, 15]))
        for m, mode in enumerate(MODES):
            for k in range(4):
                factor = f"whole:F{k + 1:02d}"
                ref = targets[
                    (targets.key == j["key"])
                    & (targets["mode"] == mode)
                    & (targets.factor == factor)
                ]
                if len(ref) != 1:
                    raise ValueError("Unique target linkage required")
                r = ref.iloc[0]
                if abs((natives[0] - natives[1])[m, k] - r.native) > 1e-10:
                    raise ValueError("Contrast target differs")
                e = errors(p["contrast"][m, k], r.carried, r.operator, r.native)
                row = {
                    "key": j["key"],
                    "cohort": j["cohort"],
                    "mode": mode,
                    "factor": factor,
                    "native": r.native,
                    "carried": r.carried,
                    "operator": r.operator,
                    "predicted": p["contrast"][m, k],
                    "final_T": r.final_T,
                }
                row.update({key: float(v) for key, v in e.items()})
                rows.append(row)
                for q, query in enumerate(QUERIES):
                    actions.append(
                        {
                            "key": j["key"],
                            "cohort": j["cohort"],
                            "mode": mode,
                            "factor": factor,
                            "query": query,
                            "native": natives[q][m, k],
                            "predicted": p["actions"][q, m, k],
                        }
                    )
    df, af = pd.DataFrame(rows), pd.DataFrame(actions)
    df.to_csv(dst / "private_contrasts.csv", index=False)
    af.to_csv(dst / "private_actions.csv", index=False)
    files = [dst / "private_contrasts.csv", dst / "private_actions.csv"]
    for keys in (("cohort", "mode"), ("cohort", "mode", "factor")):
        summaries = []
        for key, g in df.groupby(list(keys), sort=True):
            for arm in ("predicted", "carried"):
                summaries.append(
                    dict(zip(keys, key, strict=True))
                    | {"arm": arm}
                    | score_summary(g, "native", arm)
                )
        f = dst / ("summary.csv" if len(keys) == 2 else "factors.csv")
        pd.DataFrame(summaries).to_csv(f, index=False)
        files.append(f)
    action_rows = []
    for key, g in af.groupby(["cohort", "mode", "query"], sort=True):
        action_rows.append(
            dict(zip(("cohort", "mode", "query"), key, strict=True))
            | fidelity(g.native.to_numpy(), g.predicted.to_numpy())
        )
    f = dst / "actions_by_query.csv"
    pd.DataFrame(action_rows).to_csv(f, index=False)
    files.append(f)
    account = []
    for key, g in df.groupby(["cohort", "mode"], sort=True):
        a, b, c = (
            g[n].to_numpy()
            for n in ("source_error", "missing_operator", "parity_remainder")
        )
        e = g.total_error.to_numpy()
        target_energy = float(np.mean(g.native**2))
        row = dict(zip(("cohort", "mode"), key, strict=True))
        for n, v in (
            ("source_error", a),
            ("missing_operator", b),
            ("parity_remainder", c),
            ("total_error", e),
        ):
            row[n + "_rms"] = float(np.sqrt(np.mean(v * v)))
            row[n + "_mean"] = float(v.mean())
        row.update(
            covariance=float(np.cov(a, b, ddof=0)[0, 1]),
            cross_energy_over_target=float(2 * np.mean(a * b) / target_energy),
            error_closure_max=float(abs(a + b + c - e).max()),
            total_mse_over_zero=float(np.mean(e * e) / target_energy),
        )
        account.append(row)
    f = dst / "error_accounting.csv"
    pd.DataFrame(account).to_csv(f, index=False)
    files.append(f)
    summary = pd.read_csv(dst / "summary.csv")
    s = summary[(summary.cohort == "new8") & (summary.arm == "predicted")]
    gate = bool(
        (s.final_T_projection < 0).all()
        and (s.action_mse_over_zero < 1).all()
        and len(s) == 2
    )
    write(
        dst / "summary.json",
        {
            "status": "COMPLETE",
            "local_candidate_retained": gate,
            "actions": len(af),
            "contrasts": len(df),
            "fits": 0,
            "human_labels_read": False,
            "files": {p.name: sha(p) for p in files},
        },
    )
    print(
        {
            "status": "COMPLETE",
            "local_candidate_retained": gate,
            "actions": len(af),
            "contrasts": len(df),
        }
    )


def main():
    """Run a fixed source-only local mechanism packet; no capture/training stage exists."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "compile", "evaluate"))
    p.add_argument(
        "--output", type=Path, default=BASE / "logic_source_common_counteraction_v1"
    )
    p.add_argument("--name", default="compiled")
    p.add_argument("--compiled", default="compiled")
    args = p.parse_args()
    with threadpool_limits(limits=2):
        if args.stage == "prepare":
            prepare(args.output)
        elif args.stage == "compile":
            compile_stage(args.output, args.name)
        else:
            evaluate(args.output, args.name, args.compiled)


if __name__ == "__main__":
    main()
