#!/usr/bin/env python3
"""NEXT29: no-fit spread/alignment account on the frozen same-material interface."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.factor_catalog import bh_adjust
from grain_research.vector_mechanisms.material_coupling import row_correlations
from grain_research.vector_mechanisms.material_geometry import (
    local_geometry,
    product_components,
)
from scripts.run_grain_material_coupling import (
    CRITERIA,
    FACTORS,
    FC,
    SEED,
    SLICES,
    SOURCE,
    interval,
)
from scripts.run_grain_material_coupling import (
    verify as verify_previous,
)
from scripts.run_grain_real_author_path import BASE, read, sha, write

PREVIOUS = BASE / "logic_material_coupling_v1_r2"
PLAN = ROOT / "docs/GRAIN_MATERIAL_GEOMETRY_PLAN_V1.md"
ARMS = ("sg", "sp", "A", "rho", "S", "shifted_S", "shifted_rho", "matched_excess")
COMPONENTS = ("spread_component", "alignment_component", "interaction_component")


def source_feature_index(meta, addresses):
    """Bind only original source views, preserving physical NPZ row positions."""
    frame = meta.copy()
    frame["feature_index"] = np.arange(len(frame))
    frame = frame[
        frame.group.isin(["big5_only", "mbti_only"]) & frame.edit.eq("source")
    ].set_index(["author_id", "view"])
    if frame.index.duplicated().any() or set(frame.index) != set(addresses):
        raise ValueError("Source embedding addresses do not match frozen cohort")
    return frame


def verify(out):
    """Check both frozen source linkage and this numerical implementation."""
    verify_previous(PREVIOUS)
    m = read(out / "manifest.json")
    for name, expected in m["hashes"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("Changed source: " + name)
    for name, expected in read(PREVIOUS / "prediction_freeze.json")["files"].items():
        if sha(PREVIOUS / name) != expected:
            raise ValueError("Prior numerical source changed")
    return m


def prepare(out):
    """Freeze an immutable diagnostic before recomputing any criterion relation."""
    verify_previous(PREVIOUS)
    files = [
        PLAN,
        Path(__file__),
        ROOT / "grain_research/vector_mechanisms/material_geometry.py",
        PREVIOUS / "manifest.json",
        PREVIOUS / "prediction_freeze.json",
        PREVIOUS / "private_coupling_views.csv",
        PREVIOUS / "private_local_maps.npz",
        ROOT / "scripts/run_grain_material_coupling.py",
    ]
    out.mkdir(parents=True, exist_ok=False)
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_MATERIAL_GEOMETRY_V1",
            "hashes": {str(p.relative_to(ROOT)): sha(p) for p in files},
            "criterion_sha256": sha(CRITERIA),
            "fits": 0,
            "new_neural_calls": 0,
            "bootstrap": 999,
            "seed": SEED,
            "associations": 288,
            "reference": "label-free query/factor source-cohort means; descriptive, not deployment norms",
        },
    )
    print({"status": "PREPARED", "fits": 0, "new_neural_calls": 0}, flush=True)


def construct(out):
    """Expose paired geometry and freeze products before accessing source criteria."""
    verify(out)
    if (out / "prediction_freeze.json").exists():
        raise ValueError("Preserve frozen output; choose a new directory")
    start = time.monotonic()
    old = pd.read_csv(PREVIOUS / "private_coupling_views.csv", dtype={"author_id": str})
    keys = ["author_id", "view", "query", "fold"]
    if old.duplicated([*keys, "factor"]).any():
        raise ValueError("Duplicate prior address")
    meta = pd.read_csv(SOURCE / "private_feature_rows.csv", dtype={"author_id": str})
    meta = source_feature_index(
        meta, old[["author_id", "view"]].itertuples(index=False, name=None)
    )
    slices = pd.read_csv(SLICES / "private_factor_slices.csv", dtype={"author_id": str})
    means = slices.groupby(["author_id", "view", "comment"], sort=True)[FC].mean()
    lookup = {key: group.sort_index() for key, group in means.groupby(level=[0, 1])}
    records, replay_max, identity_max = [], 0.0, 0.0
    with (
        np.load(SOURCE / "private_features.npz", allow_pickle=False) as z,
        np.load(PREVIOUS / "private_local_maps.npz", allow_pickle=False) as models,
    ):
        flat, edges = z["bags"], z["offsets"]
        for (uid, view, q, fold), group in old.groupby(keys, sort=True):
            frame = lookup[(uid, view)]
            comment = frame.index.get_level_values("comment").to_numpy(int)
            index = int(meta.loc[(uid, view), "feature_index"])
            bag = flat[edges[index] : edges[index + 1]]
            if bag.shape != (12, 1024) or comment.min() < 0 or comment.max() >= 12:
                raise ValueError("Changed comment coverage")
            p = bag @ models[f"{q}__{fold}__coef"]
            account = local_geometry(frame.to_numpy(float), p[comment])
            identity_max = max(identity_max, account["identity_error"])
            by_factor = group.set_index("factor")
            if set(by_factor.index) != set(FACTORS):
                raise ValueError("Incomplete old coordinates")
            for j, factor in enumerate(FACTORS):
                original = by_factor.loc[factor]
                errors = [
                    abs(original.S - account["S"][j]),
                    abs(original.g_variance - account["sg"][j] ** 2),
                    abs(original.p_variance - account["sp"][j] ** 2),
                ]
                replay_max = max(replay_max, *errors)
                row = dict(zip(keys, (uid, view, q, fold), strict=True))
                row.update({"factor": factor, "eligible_comments": len(comment)})
                row.update({k: float(account[k][j]) for k in ARMS})
                row["degenerate"] = bool(account["degenerate"][j])
                records.append(row)
    if len(records) != len(old) or replay_max > 1e-10 or identity_max > 1e-12:
        raise ValueError("Prior account changed")
    table = pd.DataFrame(records)
    references = []
    for (q, factor), group in table.groupby(["query", "factor"], sort=True):
        if group.groupby("author_id").size().ne(3).any():
            raise ValueError("Three source views per author required")
        a0, r0 = float(group.A.mean()), float(group.rho.mean())
        terms = product_components(group.A, group.rho, a0, r0)
        for k, v in terms.items():
            table.loc[group.index, k] = v
        identity_max = max(
            identity_max, float(np.max(abs(sum(terms.values()) - group.S)))
        )
        references.append({"query": q, "factor": factor, "a0": a0, "r0": r0})
    if identity_max > 1e-12:
        raise ValueError("Reference product account failed")
    table.to_csv(out / "private_geometry_views.csv", index=False)
    pd.DataFrame(references).to_csv(out / "source_references.csv", index=False)
    files = ["private_geometry_views.csv", "source_references.csv"]
    write(
        out / "prediction_freeze.json",
        {
            "files": {n: sha(out / n) for n in files},
            "human_labels_read": False,
        },
    )
    write(
        out / "construction_summary.json",
        {
            "status": "COMPLETE",
            "source_authors": table.author_id.nunique(),
            "rows": len(table),
            "combinations": len(references),
            "degenerate_rows": int(table.degenerate.sum()),
            "fits": 0,
            "new_neural_calls": 0,
            "replay_max": replay_max,
            "identity_max": identity_max,
            "seconds": time.monotonic() - start,
        },
    )
    print(read(out / "construction_summary.json"), flush=True)


def analyze(out, name):
    """Describe every old-source criterion link without fitting or selecting a head."""
    m = verify(out)
    for n, expected in read(out / "prediction_freeze.json")["files"].items():
        if sha(out / n) != expected:
            raise ValueError("Frozen statistics changed")
    if sha(CRITERIA) != m["criterion_sha256"]:
        raise ValueError("Changed criteria")
    dest = out / name
    dest.mkdir(exist_ok=False)
    table = pd.read_csv(out / "private_geometry_views.csv", dtype={"author_id": str})
    criteria = pd.read_csv(CRITERIA, dtype={"author_id": str}).set_index(
        ["author_id", "query"]
    )
    if criteria.index.duplicated().any():
        raise ValueError("Ambiguous criterion source")
    means = (
        table.groupby(["author_id", "query", "factor"])
        .mean(numeric_only=True)
        .reset_index()
    )
    links, deltas, contributions, geometry = [], [], [], []
    account_max = 0.0
    for (q, factor), frame in means.groupby(["query", "factor"], sort=True):
        frame = frame.sort_values("author_id")
        y = np.array([criteria.loc[(uid, q), "y_true"] for uid in frame.author_id])
        if not np.isfinite(y).all():
            raise ValueError("Missing source criterion")
        rng = np.random.default_rng(SEED)
        draw = rng.integers(len(y), size=(m["bootstrap"], len(y)))
        boot, point = {}, {}
        for arm in ARMS:
            x = frame[arm].to_numpy()
            statistic = pearsonr(x, y)
            point[arm] = float(statistic.statistic)
            boot[arm] = row_correlations(x[draw], y[draw])
            lo, hi, count = interval(boot[arm])
            links.append(
                {
                    "query": q,
                    "factor": factor,
                    "arm": arm,
                    "n": len(y),
                    "r": point[arm],
                    "p": float(statistic.pvalue),
                    "low": lo,
                    "high": hi,
                    "usable_bootstrap": count,
                }
            )
        for arm, control in (("S", "shifted_S"), ("rho", "shifted_rho")):
            lo, hi, count = interval(boot[arm] - boot[control])
            deltas.append(
                {
                    "query": q,
                    "factor": factor,
                    "arm": arm,
                    "control": control,
                    "n": len(y),
                    "delta_r": point[arm] - point[control],
                    "low": lo,
                    "high": hi,
                    "usable_bootstrap": count,
                }
            )
        s = frame.S.to_numpy()
        denom = np.std(s) * np.std(y)
        yd = y[draw] - y[draw].mean(1, keepdims=True)
        sd = s[draw] - s[draw].mean(1, keepdims=True)
        denom_draw = np.sqrt(np.mean(yd**2, 1) * np.mean(sd**2, 1))
        total = 0.0
        for component in COMPONENTS:
            x = frame[component].to_numpy()
            c = (
                np.mean((x - x.mean()) * (y - y.mean())) / denom
                if denom > 1e-14
                else float("nan")
            )
            xd = x[draw] - x[draw].mean(1, keepdims=True)
            draws = np.divide(
                np.mean(xd * yd, 1),
                denom_draw,
                out=np.full(len(draw), np.nan),
                where=denom_draw > 1e-14,
            )
            lo, hi, count = interval(draws)
            contributions.append(
                {
                    "query": q,
                    "factor": factor,
                    "component": component,
                    "n": len(y),
                    "normalized_covariance": float(c),
                    "low": lo,
                    "high": hi,
                    "usable_bootstrap": count,
                }
            )
            total += c
        if np.isfinite(total):
            account_max = max(account_max, abs(total - point["S"]))
        geometry.append(
            {
                "query": q,
                "factor": factor,
                "n": len(y),
                **{
                    f"mean_{k}": float(frame[k].mean())
                    for k in (
                        "sg",
                        "sp",
                        "A",
                        "rho",
                        "shifted_rho",
                        "eligible_comments",
                    )
                },
                "criterion_account_error": abs(total - point["S"]),
                "degenerate_view_fraction": float(frame.degenerate.mean()),
            }
        )
    links = pd.DataFrame(links)
    links["p_fdr_288"] = bh_adjust(links.p.to_numpy())
    if (
        len(links) != 288
        or len(deltas) != 72
        or len(contributions) != 108
        or account_max > 1e-10
    ):
        raise ValueError("Incomplete or invalid accounting family")
    outputs = {
        "criterion_links.csv": links,
        "paired_contrasts.csv": pd.DataFrame(deltas),
        "criterion_components.csv": pd.DataFrame(contributions),
        "geometry_summary.csv": pd.DataFrame(geometry),
    }
    for n, t in outputs.items():
        t.to_csv(dest / n, index=False)
    write(
        dest / "summary.json",
        {
            "status": "COMPLETE",
            "fits": 0,
            "new_neural_calls": 0,
            "criterion_account_max": account_max,
            "construction": read(out / "construction_summary.json"),
            "inference_unit": "whole old source author; fixed maps and descriptive references",
            "files": {n: sha(dest / n) for n in outputs},
        },
    )
    print(
        links.groupby("arm")
        .agg(
            max_abs_r=("r", lambda x: abs(x).max()),
            fdr_links=("p_fdr_288", lambda x: (x < 0.05).sum()),
        )
        .to_string()
    )


def main():
    """Execute one bounded phase without overwriting frozen numerical evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "construct", "analyze"))
    parser.add_argument(
        "--output-dir", type=Path, default=BASE / "logic_material_geometry_v1_r2"
    )
    parser.add_argument("--name", default="analysis_r1")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if (
        not out.is_relative_to(BASE.resolve())
        or not args.name.replace("_", "").isalnum()
    ):
        raise ValueError("Use ignored results and a simple analysis name")
    if args.phase == "analyze":
        analyze(out, args.name)
    else:
        {"prepare": prepare, "construct": construct}[args.phase](out)


if __name__ == "__main__":
    main()
