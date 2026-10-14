#!/usr/bin/env python3
"""Extend the completed reference table with source sJIVE and evidence diagnostics."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.predecessors import TRAITS, mean_pearson
from grain_research.vector_mechanisms.external_heads import metrics
from scripts.prepare_grain_predecessor_reference import BASE, DEFAULT, digest, write
from scripts.run_grain_predecessor_reference import paired_delta, read_material


def lowrank_views(sources, train, test, dimensions=16):
    """Fit source-only PCA/scalers and balance two views by feature count."""
    fit, held, transforms = [], [], []
    for source in sources:
        pca = PCA(
            n_components=min(dimensions, len(train) - 1, source.shape[1]),
            svd_solver="full",
        )
        x = pca.fit_transform(source[train])
        scaler = StandardScaler().fit(x)
        divisor = np.sqrt(x.shape[1])
        fit.append(scaler.transform(x) / divisor)
        held.append(scaler.transform(pca.transform(source[test])) / divisor)
        transforms.append((pca, scaler))
    return fit, held, transforms


def evidence_diagnostic(data, items, vectors, out):
    """Describe retrieval geometry without changing thresholds or consulting labels."""
    q, h = vectors[: len(items)], vectors[len(items) :]
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    h = h / np.linalg.norm(h, axis=1, keepdims=True)
    rng = np.random.default_rng(731)
    top, margin, random = [], [], []
    for start in range(0, len(h), 1024):
        s = h[start : start + 1024] @ q.T
        ordered = np.partition(s, -2, axis=1)[:, -2:]
        top.extend(ordered[:, 1])
        margin.extend(ordered[:, 1] - ordered[:, 0])
        random.extend(s[np.arange(len(s)), rng.integers(0, len(q), len(s))])
    diagnostics = {
        "candidates": len(h),
        "threshold": 0.6,
        "top1_fraction_above_threshold": float(np.mean(np.asarray(top) >= 0.6)),
        "random_item_fraction_above_threshold": float(
            np.mean(np.asarray(random) >= 0.6)
        ),
        "top1_quantiles": np.quantile(top, [0.05, 0.5, 0.95]).tolist(),
        "random_item_quantiles": np.quantile(random, [0.05, 0.5, 0.95]).tolist(),
        "top1_top2_margin_quantiles": np.quantile(margin, [0.05, 0.5, 0.95]).tolist(),
        "interpretation": "retrieval geometry only; random items are not ground-truth negative sentences",
        "scope": "E5_adapter_threshold_not_calibrated_for_paraphrase_equivalence",
    }
    write(out / "evidence_diagnostic.json", diagnostics)
    # Fixed label-blind audit queue, not a claim that the matches are correct.
    selected = rng.choice(len(h), size=min(60, len(h)), replace=False)
    rows = []
    for i in selected:
        s = h[i] @ q.T
        order = np.argsort(-s)[:2]
        sentence = data["sentences"][int(i)]
        rows.append(
            {
                "sentence_index": int(i),
                "sentence": sentence["text"],
                "author_index": sentence["author_index"],
                "comment_id": sentence["comment_id"],
                "start": sentence["start"],
                "end": sentence["end"],
                "item_id": items[int(order[0])]["id"],
                "item_text": items[int(order[0])]["text"],
                "runner_up_item_text": items[int(order[1])]["text"],
                "similarity": float(s[order[0]]),
                "self_attribution": "",
                "polarity_correct": "",
                "trait_relevance": "",
                "rater": "",
            }
        )
    pd.DataFrame(rows).to_csv(out / "private_blind_evidence_audit.csv", index=False)


def run(args):
    """Run one declared fixed-rank reference; failures remain visible in status."""
    import joblib

    out = args.output
    if out.exists():
        raise ValueError("Use a new output directory, preserving previous results")
    out.mkdir(parents=True)
    _, a, data, items, vectors = read_material(args.material)
    upstream = json.loads((args.upstream / "upstream_manifest.json").read_text())
    # Verify the retrieved source against its archived hash before executing it.
    for name, info in upstream["files"].items():
        if digest(args.upstream / name) != info["sha256"]:
            raise ValueError("Upstream source changed")
    write(
        out / "contract.json",
        {
            "material": digest(args.material / "material_manifest.json"),
            "upstream": digest(args.upstream / "upstream_manifest.json"),
            "code": {
                p.name: digest(p)
                for p in [
                    Path(__file__),
                    ROOT / "scripts/run_grain_predecessor_joint.R",
                ]
            },
            "protocol": "official_5fold_624_source_authors_exploratory_reuse",
            "fixed_rankJ": 2,
            "fixed_rankA": [2, 2],
            "fixed_eta": 0.5,
            "pca_per_view": 16,
            "fit_iteration_limit": 1000,
            "fit_threshold": 0.001,
            "predict_iteration_limit": 500,
            "rank_or_eta_selection": False,
            "frozen_grain_scope": "VM05_48_lexical_mean_std_not_full_GRAIN",
            "stage": "I2_I3_local_adaptation_not_original_paper_replication",
        },
    )
    sources = [a["comments"].mean(1), a["grain"]]
    y, folds = a["y"], a["folds"]
    concat = np.full_like(y, np.nan)
    with threadpool_limits(limits=4):
        evidence_diagnostic(data, items, vectors, out)
        for fold in range(5):
            tr, te = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
            inner = {
                alpha: np.full_like(y[tr], np.nan)
                for alpha in (1.0, 10.0, 100.0, 1000.0)
            }
            for i, j in KFold(3, shuffle=True, random_state=731).split(tr):
                xs, zs, _ = lowrank_views(sources, tr[i], tr[j])
                x, z = np.concatenate(xs, axis=1), np.concatenate(zs, axis=1)
                for alpha, values in inner.items():
                    values[j] = Ridge(alpha=alpha).fit(x, y[tr[i]]).predict(z)
            chosen = max(inner, key=lambda k: mean_pearson(y[tr], inner[k]))
            xs, zs, transforms = lowrank_views(sources, tr, te)
            head = Ridge(alpha=chosen).fit(np.concatenate(xs, axis=1), y[tr])
            concat[te] = head.predict(np.concatenate(zs, axis=1))
            joblib.dump(
                {"transforms": transforms, "head": head, "alpha": chosen},
                out / f"private_concat_fold{fold}.joblib",
            )
            merged = pd.DataFrame(
                {
                    "row_id": np.r_[tr, te],
                    "partition": ["train"] * len(tr) + ["test"] * len(te),
                }
            )
            for prefix, x, z in zip(("semantic", "grain"), xs, zs, strict=True):
                v = np.r_[x, z]
                for j in range(v.shape[1]):
                    merged[f"{prefix}_{j}"] = v[:, j]
            for j in range(5):
                merged[f"y_{j}"] = y[np.r_[tr, te], j]
            merged.to_csv(out / f"private_fold{fold}.csv", index=False)
    command = [
        args.rscript,
        str(ROOT / "scripts/run_grain_predecessor_joint.R"),
        str(out),
        str(args.upstream),
    ]
    with (out / "r_execution.log").open("w") as log:
        result = subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, timeout=900, check=False
        )
    if result.returncode:
        raise RuntimeError("R reference failed; inspect archived log")
    p = pd.read_csv(out / "private_sjive_predictions.csv")
    status = pd.read_csv(out / "sjive_status.csv")
    joint = np.full_like(y, np.nan)
    parts = {key: np.full_like(y, np.nan) for key in ("shared", "semantic", "grain")}
    for row in p.itertuples():
        joint[row.row_id, row.trait] = row.prediction
        for key, values in parts.items():
            values[row.row_id, row.trait] = getattr(row, key)
    complete = bool(np.isfinite(joint).all())
    rows = [{"method": "matched_pca_concat", **r} for r in metrics(y, concat, "big5")]
    if complete:
        rows.extend(
            {"method": "upstream_sjive_fixed", **r} for r in metrics(y, joint, "big5")
        )
        pd.DataFrame([paired_delta(y, joint, concat)]).to_csv(
            out / "paired_comparison.csv", index=False
        )
        components = [
            {
                "component": k,
                "trait": trait,
                "r_with_y": pearsonr(y[:, j], v[:, j]).statistic,
            }
            for k, v in parts.items()
            for j, trait in enumerate(TRAITS)
        ]
        pd.DataFrame(components).to_csv(out / "component_associations.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "metrics_by_trait.csv", index=False)
    summary = (
        pd.DataFrame(rows)
        .groupby("method")[["pearson", "spearman", "mae", "rmse"]]
        .agg(lambda c: c.mean(skipna=False))
    )
    summary.to_csv(out / "comparison.csv")
    np.savez_compressed(
        out / "private_oof.npz",
        author_ids=a["author_ids"],
        folds=folds,
        y=y,
        concat=concat,
        sjive=joint,
        **parts,
    )
    write(
        out / "status.json",
        {
            "complete_predictions": complete,
            "successful_fits": int(status.success.sum()),
            "iteration_limit_hits": int(status.limit_hit.fillna(False).sum()),
            "scope": "source_estimator_with_fixed_rank_local_adapter",
        },
    )
    print(summary.to_string(), flush=True)


def main():
    """CLI for a bounded source-estimator comparison, no new neural training."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--material", type=Path, default=DEFAULT)
    p.add_argument(
        "--output",
        type=Path,
        default=BASE / "predecessor_integration_v1/integration_r1",
    )
    p.add_argument(
        "--upstream", type=Path, default=BASE / "predecessor_integration_v1/upstream"
    )
    p.add_argument("--rscript", default="/usr/local/bin/Rscript")
    run(p.parse_args())


if __name__ == "__main__":
    main()
