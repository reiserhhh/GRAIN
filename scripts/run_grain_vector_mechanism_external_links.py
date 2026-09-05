#!/usr/bin/env python3
"""VM05 source-only nested OOF heads; bridge evaluation is deliberately separate."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.external_heads import (
    AXES,
    TRAITS,
    load_source_labels,
    metrics,
    pooled_moments,
    predict_head,
    select_head,
    task_score,
)
from grain_research.vector_mechanisms.external_material import (
    raw_author_features,
    unpack_features,
)
from grain_research.vector_mechanisms.inventory import archive_source, code_fingerprint
from scripts.run_grain_vector_mechanism_controls import write_json
from scripts.run_grain_vector_mechanism_external_features import bundle_arms
from scripts.run_grain_vector_mechanism_external_queue import verify_completed


def dense_features(bundles, common_source, common, a):
    """Read only complete, checksum-verified observer caches; no score interpolation."""
    if a.feature == "grain_intermediate":
        if a.arm != "raw":
            pre = json.loads((a.preflight_dir / "manifest.json").read_text())
            path = (
                ROOT
                / "results/v7_multiview_projection/e1_v72_full_20260715/artifacts/common_source_comment_representation.joblib"
            )
            if hashlib.sha256(path.read_bytes()).hexdigest() != pre["runtime_sha256"]:
                raise ValueError("Frozen intermediate operator changed")
            runtime = joblib.load(path)
            rows = [
                raw_author_features(
                    runtime.transform(bundle_arms(b)[a.arm])[None, :, :],
                    pre["feature_names"],
                )[0]
                for b in bundles
            ]
            return np.stack(rows)
        with np.load(
            a.preflight_dir / "grain_frozen_features.npz", allow_pickle=False
        ) as z:
            ix = {u: i for i, u in enumerate(z["author_ids"])}
            return z["all36_raw_features"][[ix[b["author_id"]] for b in bundles]]
    if a.vector_dir is None:
        raise ValueError("Dense observer features require the completed vector queue")
    queue = json.loads((a.vector_dir / "manifest.json").read_text())
    stage = "qwen" if a.feature.startswith("qwen_") else a.feature
    if queue["material_sha256"] != common["material_sha256"]:
        raise ValueError("Vector cohort differs from common evidence")
    paths = {}
    for i in range(queue["shards"]):
        folder = a.vector_dir / stage / f"shard_{i:02d}"
        verify_completed(
            folder,
            common_source,
            common,
            stage,
            queue["shards"],
            i,
            queue["code_sha256"],
        )
        m = json.loads((folder / "manifest.json").read_text())
        paths.update({u: folder / "private_cache" / u for u in m["authors"]})
    if set(paths) != set(common["author_ids"]):
        raise ValueError("Full observer author set is incomplete")
    rows = []
    for b in bundles:
        path = paths[b["author_id"]]
        meta = json.loads(path.with_suffix(".json").read_text())
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as z:
            h = unpack_features(
                {k: z[k] for k in z.files}, meta.get("feature_aliases", {})
            )
        if stage != "qwen":
            vectors = h[a.arm + "_comment_vectors"].reshape(36, -1).astype(float)
            rows.append(np.concatenate([vectors.mean(0), vectors.std(0)]))
        else:
            site = a.feature.removeprefix("qwen_")
            counts = [
                next(
                    r["body_tokens"]
                    for r in meta["readouts"]
                    if r["arm"] == a.arm and r["view"] == v and r["task"] == "behavior"
                )
                for v in range(3)
            ]
            rows.append(
                pooled_moments(
                    h[f"{a.arm}_{site}_body_mean"],
                    h[f"{a.arm}_{site}_body_std"],
                    counts,
                )
            )
    return np.stack(rows)


def bootstrap_score(y, pred, task, draws=1000):
    """Author-resampled score uncertainty conditional on already fitted OOF heads."""
    rng = np.random.default_rng(731)
    values = []
    for _ in range(draws):
        ix = rng.integers(len(y), size=len(y))
        values.append(task_score(y[ix], pred[ix], task))
    finite = np.asarray(values)[np.isfinite(values)]
    return {
        "lower": float(np.quantile(finite, 0.025)),
        "upper": float(np.quantile(finite, 0.975)),
        "draws": draws,
        "undefined_draws": draws - len(finite),
        "scope": "author-resampled, conditional on fitted OOF predictions; no refit",
    }


def main():
    """Finish source OOF and seal a full-source head without reading bridge outcomes."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--views", type=Path, required=True)
    p.add_argument("--preflight-dir", type=Path, required=True)
    p.add_argument("--parent-dir", type=Path, required=True)
    p.add_argument("--prepared-root", type=Path, required=True)
    p.add_argument("--vector-dir", type=Path)
    p.add_argument("--task", choices=["big5", *AXES], required=True)
    p.add_argument(
        "--feature",
        choices=[
            "word_char",
            "grain_intermediate",
            "bge_large",
            "e5_large",
            "qwen_block_04",
            "qwen_block_10",
            "qwen_block_16",
            "qwen_final_norm",
        ],
        required=True,
    )
    p.add_argument("--arm", choices=["raw", "cue_mask", "placebo_mask"], default="raw")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--permutation-seed", type=int)
    p.add_argument("--no-pca", action="store_true")
    p.add_argument("--max-seconds", type=int, default=7200)
    a = p.parse_args()
    if not 1 <= a.max_seconds <= 28800:
        raise ValueError("Invalid bounded runtime")
    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Use a new output directory")
    archive_source(ROOT, out / "source_snapshot.zip")
    source = json.loads(a.views.read_text(encoding="utf-8"))
    common = json.loads((a.preflight_dir / "common_material_manifest.json").read_text())
    parent = json.loads((a.parent_dir / "manifest.json").read_text())
    if material_hash(source) != common["material_sha256"]:
        raise ValueError("Common material changed")
    group = "big5_only" if a.task == "big5" else "mbti_only"
    bundles = [b for b in source if b["source_group"] == group]
    ids = np.asarray([b["author_id"] for b in bundles])
    if len(set(ids)) != len(ids) or set(ids) & {
        b["author_id"] for b in source if b["source_group"] == "bridge"
    }:
        raise ValueError("Source/bridge author overlap")
    if a.feature == "word_char":
        x = np.asarray(["\n\n".join(bundle_arms(b)[a.arm]) for b in bundles])
        feature_kind = "word_char"
    else:
        x = dense_features(bundles, source, common, a)
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite representation")
        feature_kind = "dense"
    # Feature rules and the execution manifest are bound before outcome conversion.
    manifest = {
        "protocol": "GRAIN_VM05_SOURCE_OOF_V1",
        "task": a.task,
        "feature": a.feature,
        "arm": a.arm,
        "source_group": group,
        "authors": len(bundles),
        "source_material_sha256": material_hash(bundles),
        "common_material_sha256": common["material_sha256"],
        "code_sha256": code_fingerprint(ROOT),
        "seed": 731,
        "pca": not a.no_pca,
        "permutation_seed": a.permutation_seed,
        "bridge_labels_used": False,
        "inner_cv": "3 author folds, seed731",
        "outer_cv": "5 source-specific official folds",
        "goal": "External readability; not proof that the LLM uses a personality direction",
        "grain_comparison_scope": "Frozen V7 lexical projection mean/std, not the entire current GRAIN 4W theory or released profile",
        "feature_input_sha256": material_hash(x.tolist())
        if feature_kind == "word_char"
        else hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest(),
    }
    manifest["identity"] = material_hash(manifest)
    write_json(out / "manifest.json", manifest)
    y, label_audit = load_source_labels(
        a.prepared_root, bundles, a.task, parent["source_tables"]
    )
    if a.permutation_seed is not None:
        y = y[np.random.default_rng(a.permutation_seed).permutation(len(y))]
    write_json(out / "source_label_audit.json", label_audit)
    folds = np.asarray([b["official_folds"][a.task] for b in bundles])
    if set(folds) != set(range(5)):
        raise ValueError("All five official folds required")
    pred = np.full(y.shape, np.nan)
    selected, fold_rows = [], []
    started = time.monotonic()
    with threadpool_limits(limits=4):
        for fold in range(5):
            train, test = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
            if set(ids[train]) & set(ids[test]):
                raise ValueError("Outer author overlap")
            model, search, parameter = select_head(
                x[train], y[train], ids[train], feature_kind, a.task, pca=not a.no_pca
            )
            pred[test] = predict_head(model, x[test], a.task)
            selected.append(
                {
                    "fold": fold,
                    "parameter": parameter,
                    "train_n": len(train),
                    "test_n": len(test),
                    "inner_search": search,
                }
            )
            fold_rows.extend(
                dict(row, fold=fold) for row in metrics(y[test], pred[test], a.task)
            )
            joblib.dump(model, out / f"private_fold_{fold}.joblib", compress=3)
            write_json(out / "selection.json", selected)
            write_json(
                out / "run_status.json",
                {
                    "status": "RUNNING",
                    "completed_folds": fold + 1,
                    "seconds": time.monotonic() - started,
                    "bridge_labels_used": False,
                },
            )
            print(
                json.dumps(
                    {
                        "fold": fold,
                        "score": task_score(y[test], pred[test], a.task),
                        "seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
            if time.monotonic() - started > a.max_seconds:
                write_json(
                    out / "run_status.json",
                    {"status": "PARTIAL_TIME_BOUND", "completed_folds": fold + 1},
                )
                return
        result = metrics(y, pred, a.task)
        score = task_score(y, pred, a.task)
        uncertainty = bootstrap_score(y, pred, a.task)
        # This full-source parameter is selected inside source data only.
        full, full_search, full_parameter = select_head(
            x, y, ids, feature_kind, a.task, pca=not a.no_pca
        )
        full_path = out / "private_full_source_head.joblib"
        joblib.dump(full, full_path, compress=3)
    pd.DataFrame(result).to_csv(out / "metrics_by_trait.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(out / "metrics_by_fold.csv", index=False)
    rows = []
    names = TRAITS if a.task == "big5" else [a.task]
    yy, pp = (y, pred) if a.task == "big5" else (y[:, None], pred[:, None])
    for i, uid in enumerate(ids):
        for j, trait in enumerate(names):
            rows.append(
                {
                    "author_id": uid,
                    "fold": int(folds[i]),
                    "trait": trait,
                    "y_true": yy[i, j],
                    "prediction": pp[i, j],
                }
            )
    pd.DataFrame(rows).to_csv(out / "private_oof_predictions.csv", index=False)
    summary = {
        "status": "COMPLETE",
        "authors": len(bundles),
        "task": a.task,
        "feature": a.feature,
        "arm": a.arm,
        "score": score,
        "score_name": "mean_pearson" if a.task == "big5" else "auc",
        "bootstrap": uncertainty,
        "seconds": time.monotonic() - started,
        "full_parameter": full_parameter,
        "full_search": full_search,
        "full_head_sha256": hashlib.sha256(full_path.read_bytes()).hexdigest(),
        "parent_identity": manifest["identity"],
        "bridge_labels_used": False,
        "permutation_seed": a.permutation_seed,
        "open_controls": [
            "paired observer increments",
            "cue/placebo contrasts",
            "shared-thread sensitivity",
            "complete 199-permutation null",
            "independent bridge",
        ],
        "not_comparable_to_full_paper_cohort": True,
    }
    write_json(out / "summary.json", summary)
    write_json(out / "run_status.json", summary)
    (out / "report.md").write_text(
        f"# VM05 source OOF: {a.task} / {a.feature} / {a.arm}\n\n"
        f"Authors: {len(ids)}. {summary['score_name']}: {score:.6f}. "
        f"Author bootstrap: [{uncertainty['lower']:.6f}, {uncertainty['upper']:.6f}].\n\n"
        "Independent source-specific official OOF, not a new lockbox or full-paper replication. "
        "No bridge labels used. No personality mechanism claim. "
        "Other observers and frozen negative controls remain necessary.\n",
        encoding="utf-8",
    )
    print(
        json.dumps({k: v for k, v in summary.items() if k != "full_search"}), flush=True
    )


if __name__ == "__main__":
    main()
