#!/usr/bin/env python3
"""NEXT48: one source-only query correction; zero native calls or residual fitting."""

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
from grain_research.vector_mechanisms.source_common_counteraction import direct_actions
from grain_research.vector_mechanisms.source_query_correction import (
    apply_correction,
    error_account,
    query_readers,
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

SOURCE = BASE / "logic_source_common_counteraction_v1"
PLAN = ROOT / "docs/GRAIN_SOURCE_QUERY_CORRECTION_PLAN_V1.md"


def prepare(out):
    """Freeze pure source inputs plus old common reader; copy no evaluated target."""
    m = verify(SOURCE)
    cf = read(SOURCE / "compiled/freeze.json")
    verify_hashes(SOURCE / "compiled", cf["files"])
    if cf["status"] != "COMPLETE" or m["block"] != 24:
        raise ValueError("Complete block24 source-common packet required")
    out.mkdir(exist_ok=False)
    (out / "private_inputs").mkdir()
    jobs = read(SOURCE / "private_jobs.json")
    prepared, lineage = [], [SOURCE / "manifest.json", SOURCE / "compiled/freeze.json"]
    for j in jobs:
        src = SOURCE / "private_inputs" / (j["key"] + ".npz")
        prior = SOURCE / "compiled" / src.name
        a = arrays(src)
        with np.load(prior, allow_pickle=False) as z:
            common = z["v"]
        if set(a) != {"d", "shared"}:
            raise ValueError("Only d/shared inputs permitted")
        dst = out / "private_inputs" / src.name
        np.savez_compressed(dst, **a, previous_common=common)
        prepared.append(dst)
        lineage.extend([src, prior])
    for name in ("private_source_bank.npz", "private_weights.npz", "private_jobs.json"):
        shutil.copy2(SOURCE / name, out / name)
        prepared.append(out / name)
        lineage.append(SOURCE / name)
    code = [
        PLAN,
        Path(__file__),
        ROOT / "tests/test_grain_source_query_correction.py",
        ROOT / "scripts/run_grain_compiled_query_transfer.py",
    ]
    code += [
        ROOT / "grain_research/vector_mechanisms" / (n + ".py")
        for n in (
            "source_query_correction",
            "source_common_counteraction",
            "compiled_gated_readout",
            "gated_update",
            "compiled_query_transfer",
            "stage_file_guard",
        )
    ]
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_SOURCE_QUERY_CORRECTION_V1",
            "block": 24,
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "max_seconds": 60,
            "native_calls": 0,
            "fits": 0,
            "human_labels_read": False,
            "inputs": {
                str(p.relative_to(ROOT)): sha(p) for p in sorted(set(code + lineage))
            },
            "prepared": {str(p.relative_to(out)): sha(p) for p in prepared},
        },
    )
    print({"status": "PREPARED", "pairs": len(jobs), "target_fields_copied": False})


def predict(out, name):
    """Compile paired source adjoints and freeze all actions before evaluating o/n."""
    m = verify(out)
    jobs = read(out / "private_jobs.json")
    dst = out / name
    dst.mkdir(exist_ok=False)
    reads = [out / "private_source_bank.npz", out / "private_weights.npz"]
    reads += [out / "private_inputs" / (j["key"] + ".npz") for j in jobs]
    files = [dst / (j["key"] + ".npz") for j in jobs]
    probes = [
        SOURCE / "analysis/private_contrasts.csv",
        SOURCE / "analysis/private_actions.csv",
        BASE / "logic_query_counteraction_v1/private_capture/new8_a00_neuroticism.npz",
        BASE
        / "logic_layer_accumulation_v1/private_predictions/new8_a00_neuroticism.npz",
    ]
    denials = install(
        reads,
        files + [dst / "freeze.json"],
        probes,
        [ROOT / "grain_research", ROOT / "scripts"],
    )
    start = time.monotonic()
    bank = arrays(reads[0])
    weights = {k: np.asarray(v, float) for k, v in arrays(reads[1]).items()}
    checks = []
    for j, f in zip(jobs, files, strict=True):
        if time.monotonic() - start > m["max_seconds"]:
            raise TimeoutError("Fixed numeric cap")
        a = arrays(out / "private_inputs" / f.name)
        v, x, members = query_readers(
            bank, j["cohort"], j["index"], a["shared"], weights
        )
        p = apply_correction(v, a["d"])
        ce = float(abs(p["common_v"] - a["previous_common"]).max())
        de = max(
            float(
                abs(
                    p["actions"][q]
                    - direct_actions(x[q], a["shared"], a["d"], weights)[q]
                ).max()
            )
            for q in range(2)
        )
        se = float(abs(p["contrast"] - (p["actions"][0] - p["actions"][1])).max())
        if max(ce, de, se) > 1e-10:
            raise ValueError("Fixed compilation/common/split identity failed")
        np.savez_compressed(f, v=v, **p)
        checks.append(
            {
                "key": j["key"],
                "members": members,
                "common_error": ce,
                "direct_error": de,
                "split_error": se,
            }
        )
    seconds = time.monotonic() - start
    if seconds > m["max_seconds"]:
        raise TimeoutError("Fixed numeric cap exceeded")
    result = {
        "status": "COMPLETE",
        "seconds": seconds,
        "native_calls": 0,
        "fits": 0,
        "human_labels_read": False,
        "target_reads": 0,
        "denial_probes": 4,
        "extra_denials": len(denials) - 4,
        "checks": checks,
        "files": {p.name: sha(p) for p in files},
    }
    write(dst / "freeze.json", result)
    print({k: v for k, v in result.items() if k not in ("checks", "files")})


def evaluate(out, name, predictions):
    """Score the missing operator independently from the total native contrast."""
    verify(out)
    pf = read(out / predictions / "freeze.json")
    verify_hashes(out / predictions, pf["files"])
    ref = read(SOURCE / "analysis/summary.json")
    verify_hashes(SOURCE / "analysis", ref["files"])
    old = pd.read_csv(SOURCE / "analysis/private_contrasts.csv")
    old_actions = pd.read_csv(SOURCE / "analysis/private_actions.csv")
    dst = out / name
    dst.mkdir(exist_ok=False)
    rows = []
    action_rows = []
    for j in read(out / "private_jobs.json"):
        p = arrays(out / predictions / (j["key"] + ".npz"))
        for m, mode in enumerate(MODES):
            for k in range(4):
                factor = f"whole:F{k + 1:02d}"
                g = old[
                    (old.key == j["key"])
                    & (old["mode"] == mode)
                    & (old.factor == factor)
                ]
                if len(g) != 1:
                    raise ValueError("Unique contrast linkage required")
                r = g.iloc[0]
                if abs(p["common"][m, k] - r.predicted) > 1e-10:
                    raise ValueError("Old common prediction changed")
                row = {
                    "key": j["key"],
                    "cohort": j["cohort"],
                    "mode": mode,
                    "factor": factor,
                    "native": r.native,
                    "carried": r.carried,
                    "operator": r.operator,
                    "final_T": r.final_T,
                    "common": p["common"][m, k],
                    "correction": p["correction"][m, k],
                    "predicted": p["contrast"][m, k],
                }
                e = error_account(
                    row["common"], row["correction"], r.carried, r.operator, r.native
                )
                row.update({n: float(v) for n, v in e.items()})
                rows.append(row)
                for q, query in enumerate(QUERIES):
                    a = old_actions[
                        (old_actions.key == j["key"])
                        & (old_actions["mode"] == mode)
                        & (old_actions.factor == factor)
                        & (old_actions["query"] == query)
                    ]
                    if len(a) != 1:
                        raise ValueError("Unique action linkage required")
                    action_rows.append(
                        {
                            "key": j["key"],
                            "cohort": j["cohort"],
                            "mode": mode,
                            "factor": factor,
                            "query": query,
                            "native": a.iloc[0].native,
                            "predicted": p["actions"][q, m, k],
                        }
                    )
    df, af = pd.DataFrame(rows), pd.DataFrame(action_rows)
    files = []

    def save(name, values):
        f = dst / name
        pd.DataFrame(values).to_csv(f, index=False)
        files.append(f)

    save("private_contrasts.csv", df)
    save("private_actions.csv", af)
    for keys in (("cohort", "mode"), ("cohort", "mode", "factor")):
        summaries = []
        ops = []
        for key, g in df.groupby(list(keys), sort=True):
            addr = dict(zip(keys, key, strict=True))
            for arm in ("common", "predicted"):
                summaries.append(
                    addr
                    | {"arm": arm, "authors": g.key.nunique()}
                    | fidelity(g.native.to_numpy(), g[arm].to_numpy())
                    | {
                        "final_T_projection": float(
                            g[arm] @ g.final_T / (g.final_T @ g.final_T)
                        )
                    }
                )
            ops.append(
                addr
                | {"authors": g.key.nunique()}
                | fidelity(g.operator.to_numpy(), g.correction.to_numpy())
                | {
                    "final_T_projection": float(
                        g.correction @ g.final_T / (g.final_T @ g.final_T)
                    )
                }
            )
        suffix = "summary" if len(keys) == 2 else "factors"
        save(suffix + ".csv", summaries)
        save("operator_" + suffix + ".csv", ops)
    actions = []
    for key, g in af.groupby(["cohort", "mode", "query"], sort=True):
        actions.append(
            dict(zip(("cohort", "mode", "query"), key, strict=True))
            | fidelity(g.native.to_numpy(), g.predicted.to_numpy())
        )
    save("actions_by_query.csv", actions)
    accounts = []
    for key, g in df.groupby(["cohort", "mode"], sort=True):
        row = dict(zip(("cohort", "mode"), key, strict=True))
        for n in (
            "source_error",
            "query_error",
            "parity_error",
            "old_error",
            "new_error",
        ):
            row[n + "_rms"] = float(np.sqrt(np.mean(g[n] ** 2)))
        row.update(
            error_closure=float(
                abs(g.source_error + g.query_error + g.parity_error - g.new_error).max()
            ),
            mse_change=float(np.mean(g.new_error**2 - g.old_error**2)),
            mse_change_cross=float(g.mse_change_cross.mean()),
            mse_change_self=float(g.mse_change_self.mean()),
            covariance=float(np.cov(g.source_error, g.query_error, ddof=0)[0, 1]),
        )
        if (
            abs(row["mse_change"] - row["mse_change_cross"] - row["mse_change_self"])
            > 1e-12
        ):
            raise ValueError("Squared error change identity failed")
        accounts.append(row)
    save("error_accounting.csv", accounts)
    op = pd.read_csv(dst / "operator_summary.csv")
    op = op[op.cohort == "new8"]
    gate = bool(len(op) == 2 and (op.action_mse_over_zero < 1).all())
    s = pd.read_csv(dst / "summary.csv")
    s = s[s.cohort == "new8"].pivot(
        index="mode", columns="arm", values="action_mse_over_zero"
    )
    result = {
        "status": "COMPLETE",
        "operator_candidate_retained": gate,
        "total_error_improves_both_new8_modes": bool((s.predicted < s.common).all()),
        "actions": len(af),
        "contrasts": len(df),
        "native_calls": 0,
        "fits": 0,
        "human_labels_read": False,
        "files": {p.name: sha(p) for p in files},
    }
    write(dst / "summary.json", result)
    print({k: v for k, v in result.items() if k != "files"})


def main():
    """Run a single frozen source-query candidate, with no capture or tuning entry."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "predict", "evaluate"))
    p.add_argument(
        "--output", type=Path, default=BASE / "logic_source_query_correction_v1"
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
