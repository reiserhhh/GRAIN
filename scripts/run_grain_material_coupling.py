#!/usr/bin/env python3
"""NEXT27: source-OOF comment functions aligned with frozen lexical GRAIN slices."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.configuration_extraction import InputProgram
from grain_research.vector_mechanisms.direct_traits import AXES, BIG5
from grain_research.vector_mechanisms.factor_catalog import bh_adjust
from grain_research.vector_mechanisms.material_coupling import (
    local_coupling,
    row_correlations,
)
from scripts.run_grain_real_author_path import BASE, read, sha, write

PLAN = ROOT / "docs/GRAIN_SAME_MATERIAL_INTERPRETATION_PLAN_V1.md"
SOURCE = BASE / "logic_configuration_extraction_v1"
OLD = BASE / "logic_reading_levels_v1"
DYNAMIC = BASE / "dynamic_path_channels_v1"
SLICES = DYNAMIC / "within_comment"
MATERIAL = BASE / "source_remainder_s9_r1/private_material.json"
CRITERIA = BASE / "all_factor_source_m29_v1/private_external_links.csv"
SEED = 20260910
QUERIES = (*AXES, *BIG5)
FC = [f"SU7-FC-{i:04d}@v1_z" for i in range(1, 5)]
FACTORS = [f"whole:F{i:02d}" for i in range(1, 5)]


def verify(out):
    """Check the finite input/code manifest without reading the criterion table."""
    manifest = read(out / "manifest.json")
    for name, expected in manifest["hashes"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("Changed input: " + name)
    return manifest


def prepare(out):
    """Bind existing matching materials and numerical observations before new fitting."""
    fm = read(SOURCE / "feature_manifest.json")
    sm = read(SLICES / "manifest.json")
    key = str(MATERIAL.relative_to(ROOT))
    if not (fm["hashes"][key] == sm["sources"][key] == sha(MATERIAL)):
        raise ValueError(
            "GRAIN slices and embedding cache do not use the same material"
        )
    old = read(OLD / "prediction_freeze.json")
    if sha(OLD / "private_source_oof.csv") != old["files"]["private_source_oof.csv"]:
        raise ValueError("Old OOF predictions changed")
    files = [
        PLAN,
        Path(__file__),
        MATERIAL,
        ROOT / "grain_research/vector_mechanisms/material_coupling.py",
        ROOT / "grain_research/vector_mechanisms/configuration_extraction.py",
        SOURCE / "feature_manifest.json",
        SOURCE / "private_feature_rows.csv",
        SOURCE / "private_features.npz",
        SLICES / "manifest.json",
        SLICES / "private_factor_slices.csv",
        SLICES / "private_comment_descriptors.csv",
        DYNAMIC / "private_path_views.csv",
        OLD / "prediction_freeze.json",
        OLD / "private_source_oof.csv",
    ]
    out.mkdir(parents=True, exist_ok=False)
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_SAME_MATERIAL_COUPLING_V1",
            "hashes": {str(p.relative_to(ROOT)): sha(p) for p in files},
            "criterion_sha256": sha(CRITERIA),
            "fits": 45,
            "new_neural_calls": 0,
            "source_authors": 276,
            "queries": list(QUERIES),
            "label_training": False,
            "seed": SEED,
            "unit": "comment mean of four lexical slices; not a transformer token effect",
        },
    )
    print({"status": "PREPARED", "fits": 45, "new_neural_calls": 0}, flush=True)


def construct(out):
    """Rebuild the identical source-fold configuration reader, then expose its local map."""
    verify(out)
    if (out / "prediction_freeze.json").exists():
        raise ValueError("Preserve frozen output; use a new directory")
    started = time.monotonic()
    meta = pd.read_csv(SOURCE / "private_feature_rows.csv", dtype={"author_id": str})
    with np.load(SOURCE / "private_features.npz", allow_pickle=False) as z:
        flat, edges = z["bags"], z["offsets"]
    bags = [flat[edges[i] : edges[i + 1]] for i in range(len(meta))]
    paths = pd.read_csv(DYNAMIC / "private_path_views.csv", dtype={"author_id": str})
    slices = pd.read_csv(SLICES / "private_factor_slices.csv", dtype={"author_id": str})
    keys = ["author_id", "view", "comment"]
    if slices.duplicated([*keys, "slice"]).any():
        raise ValueError("Duplicate slice address")
    grouped = slices.groupby(keys, sort=True)
    if not grouped["slice"].agg(lambda x: tuple(sorted(x)) == (0, 1, 2, 3)).all():
        raise ValueError("Four distinct lexical slices required per selected comment")
    means = grouped[FC].mean()
    desc = pd.read_csv(
        SLICES / "private_comment_descriptors.csv", dtype={"author_id": str}
    ).set_index(keys)
    if desc.index.duplicated().any() or set(means.index) != set(desc.index):
        raise ValueError("Mismatch between slice and comment address sets")
    error = np.max(
        abs(
            means.to_numpy()
            - desc.loc[means.index, ["level::" + f for f in FACTORS]].to_numpy()
        )
    )
    if error > 1e-10:
        raise ValueError("Slice aggregation does not replay old comment scores")
    lookup = {k: g.sort_index().to_numpy(float) for k, g in means.groupby(level=[0, 1])}
    comments = {
        k: g.index.get_level_values("comment").to_numpy(int)
        for k, g in means.groupby(level=[0, 1])
    }
    old = pd.read_csv(OLD / "private_source_oof.csv", dtype={"author_id": str})
    old = old[old.arm == "pooled"].set_index(["author_id", "view", "query", "fold"])
    records, predictions, models = [], [], {}
    replay_max, identity_max, n_fits = 0.0, 0.0, 0
    for q in QUERIES:
        group = "mbti_only" if q in AXES else "big5_only"
        data = meta[meta.group == group].sort_values(["author_id", "view"]).copy()
        data["feature_index"] = data.index
        data = data.merge(
            paths[paths["query"] == q][["author_id", "view", "configuration"]],
            on=["author_id", "view"],
            validate="one_to_one",
        )
        ids = data.author_id.unique()
        if (
            len(ids) != (63 if q in AXES else 213)
            or data.groupby("author_id").size().ne(3).any()
        ):
            raise ValueError("Complete original source cohort required")
        for fold, (train, test) in enumerate(
            KFold(5, shuffle=True, random_state=SEED).split(ids)
        ):
            training = data[data.author_id.isin(ids[train])]
            testing = data[data.author_id.isin(ids[test])]
            if set(training.author_id) & set(testing.author_id):
                raise ValueError("Author overlap")
            y = training.configuration.to_numpy()
            model = InputProgram("e5_mean", np.zeros((2, 1024)), penalty=0.01).fit(
                [bags[i] for i in training.feature_index], None, np.column_stack([y, y])
            )
            coef, intercept = model.linear_instance_map()
            n_fits += 1
            models[f"{q}__{fold}__coef"] = coef[0]
            models[f"{q}__{fold}__intercept"] = intercept[:1]
            for r in testing.itertuples():
                bag = bags[r.feature_index]
                if bag.shape != (12, 1024):
                    raise ValueError("Exact12-comment source view required")
                psi = bag @ coef[0]
                prediction = float(intercept[0] + psi.mean())
                previous = old.loc[(r.author_id, r.view, q, fold)].predicted_c
                replay_max = max(replay_max, abs(prediction - previous))
                predictions.append(
                    {
                        "author_id": r.author_id,
                        "view": r.view,
                        "query": q,
                        "fold": fold,
                        "prediction": prediction,
                        "previous": previous,
                    }
                )
                key = (r.author_id, r.view)
                g, index = lookup[key], comments[key]
                if (
                    len(index) < 2
                    or len(set(index)) != len(index)
                    or index.min() < 0
                    or index.max() >= 12
                ):
                    raise ValueError("Invalid within-view comment join")
                account = local_coupling(g, psi[index])
                identity_max = max(identity_max, account["identity_error"])
                for j, factor in enumerate(FACTORS):
                    record = {
                        "author_id": r.author_id,
                        "view": r.view,
                        "query": q,
                        "fold": fold,
                        "factor": factor,
                    }
                    record.update(
                        {
                            k: account[k][j]
                            for k in ("U", "W", "D", "S", "shifted_W", "g_variance")
                        }
                    )
                    record.update(
                        {
                            k: account[k]
                            for k in (
                                "p_variance",
                                "effective_comments",
                                "eligible_comments",
                            )
                        }
                    )
                    records.append(record)
        print({"query": q, "fits": n_fits, "OOF_replay_max": replay_max}, flush=True)
    if (
        n_fits != 45
        or len(predictions) != len(old)
        or replay_max > 1e-9
        or identity_max > 1e-10
    ):
        raise ValueError("Source OOF replay or mathematical identity changed")
    pd.DataFrame(records).to_csv(out / "private_coupling_views.csv", index=False)
    pd.DataFrame(predictions).to_csv(out / "private_oof_replay.csv", index=False)
    np.savez_compressed(out / "private_local_maps.npz", **models)
    files = [
        "private_coupling_views.csv",
        "private_oof_replay.csv",
        "private_local_maps.npz",
    ]
    write(
        out / "prediction_freeze.json",
        {"files": {n: sha(out / n) for n in files}, "human_labels_read": False},
    )
    write(
        out / "construction_summary.json",
        {
            "status": "COMPLETE",
            "fits": n_fits,
            "seconds": time.monotonic() - started,
            "new_neural_calls": 0,
            "source_authors": 276,
            "new_participants": 0,
            "slice_rows": len(slices),
            "eligible_comments": len(means),
            "input_comments": 276 * 36,
            "replayed_views": len(predictions),
            "OOF_replay_max": replay_max,
            "slice_aggregation_replay_max": float(error),
            "coupling_identity_max": identity_max,
        },
    )
    print(read(out / "construction_summary.json"), flush=True)


def interval(draws):
    """Return the pointwise finite-bootstrap interval and actual usable draw count."""
    finite = np.asarray(draws)[np.isfinite(draws)]
    return (
        [*np.quantile(finite, [0.025, 0.975]).tolist(), len(finite)]
        if len(finite)
        else [float("nan"), float("nan"), 0]
    )


def analyze(out, name):
    """Join old source questionnaires after freeze; no criterion fitting or selection."""
    manifest = verify(out)
    freeze = read(out / "prediction_freeze.json")
    for path, expected in freeze["files"].items():
        if sha(out / path) != expected:
            raise ValueError("Prediction freeze changed")
    if sha(CRITERIA) != manifest["criterion_sha256"]:
        raise ValueError("Criterion source changed")
    dest = out / name
    dest.mkdir(exist_ok=False)
    data = pd.read_csv(out / "private_coupling_views.csv", dtype={"author_id": str})
    criteria = pd.read_csv(CRITERIA, dtype={"author_id": str}).set_index(
        ["author_id", "query"]
    )
    if criteria.index.duplicated().any():
        raise ValueError("Nonunique source labels")
    means = (
        data.groupby(["author_id", "query", "factor"])
        .mean(numeric_only=True)
        .reset_index()
    )
    links, deltas, local, coverage = [], [], [], []
    for (q, factor), frame in means.groupby(["query", "factor"], sort=True):
        frame = frame.sort_values("author_id")
        target = np.array([criteria.loc[(uid, q)].y_true for uid in frame.author_id])
        if not np.isfinite(target).all():
            raise ValueError("Incomplete criterion join")
        rng = np.random.default_rng(SEED)
        draws = rng.integers(len(frame), size=(999, len(frame)))
        boot = {}
        for arm in ("U", "W", "D", "S", "shifted_W"):
            x = frame[arm].to_numpy()
            r = pearsonr(x, target)
            boot[arm] = row_correlations(x[draws], target[draws])
            lo, hi, n_boot = interval(boot[arm])
            links.append(
                {
                    "query": q,
                    "factor": factor,
                    "arm": arm,
                    "n": len(x),
                    "r": float(r.statistic),
                    "p": float(r.pvalue),
                    "low": lo,
                    "high": hi,
                    "usable_bootstrap": n_boot,
                }
            )
        for control in ("U", "shifted_W"):
            delta = float(
                pearsonr(frame.W, target).statistic
                - pearsonr(frame[control], target).statistic
            )
            lo, hi, n_boot = interval(boot["W"] - boot[control])
            deltas.append(
                {
                    "query": q,
                    "factor": factor,
                    "control": control,
                    "n": len(frame),
                    "delta_r": delta,
                    "low": lo,
                    "high": hi,
                    "usable_bootstrap": n_boot,
                }
            )
        s, vg, vp = (frame[k].to_numpy() for k in ("S", "g_variance", "p_variance"))
        denom = np.sqrt(vg.mean() * vp.mean())
        corr = s.mean() / denom if denom > 1e-14 else float("nan")
        denom_draw = np.sqrt(vg[draws].mean(1) * vp[draws].mean(1))
        values = np.divide(
            s[draws].mean(1),
            denom_draw,
            out=np.full(999, np.nan),
            where=denom_draw > 1e-14,
        )
        lo, hi, n_boot = interval(values)
        local.append(
            {
                "query": q,
                "factor": factor,
                "n": len(frame),
                "centered_material_r": corr,
                "low": lo,
                "high": hi,
                "usable_bootstrap": n_boot,
            }
        )
        if factor == FACTORS[0]:
            coverage.append(
                {
                    "query": q,
                    "authors": len(frame),
                    "mean_eligible_comments": float(frame.eligible_comments.mean()),
                    "mean_effective_comments": float(frame.effective_comments.mean()),
                }
            )
    table = pd.DataFrame(links)
    table["p_fdr_180"] = bh_adjust(table.p.to_numpy())
    outputs = {
        "criterion_links.csv": table,
        "paired_association_deltas.csv": pd.DataFrame(deltas),
        "centered_material_links.csv": pd.DataFrame(local),
        "coverage.csv": pd.DataFrame(coverage),
    }
    for n, t in outputs.items():
        t.to_csv(dest / n, index=False)
    write(
        dest / "summary.json",
        {
            "status": "COMPLETE",
            "criterion_fits": 0,
            "new_neural_calls": 0,
            "criterion_rows": len(table),
            "local_links": len(local),
            "paired_contrasts": len(deltas),
            "construction": read(out / "construction_summary.json"),
            "inference_unit": "whole source author; fixed OOF maps; exploratory reused source labels",
            "files": {n: sha(dest / n) for n in outputs},
        },
    )
    print(
        table.groupby("arm")
        .agg(
            max_abs_r=("r", lambda x: abs(x).max()),
            fdr_links=("p_fdr_180", lambda x: (x < 0.05).sum()),
        )
        .to_string()
    )
    print(
        pd.DataFrame(deltas)
        .groupby("control")
        .agg(
            mean_delta=("delta_r", "mean"),
            positive=("delta_r", lambda x: (x > 0).sum()),
            positive_interval=("low", lambda x: (x > 0).sum()),
        )
        .to_string()
    )


def main():
    """Run one finite phase in an immutable ignored output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "construct", "analyze"))
    parser.add_argument(
        "--output-dir", type=Path, default=BASE / "logic_material_coupling_v1"
    )
    parser.add_argument("--name", default="analysis_r1")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if (
        not out.is_relative_to(BASE.resolve())
        or not args.name.replace("_", "").isalnum()
    ):
        raise ValueError("Ignored result directory and simple analysis name required")
    with threadpool_limits(limits=2):
        if args.phase == "analyze":
            analyze(out, args.name)
        else:
            {"prepare": prepare, "construct": construct}[args.phase](out)


if __name__ == "__main__":
    main()
