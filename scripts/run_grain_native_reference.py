#!/usr/bin/env python3
"""Replay the frozen native GRAIN author operator on predecessor source material.

This is the existing V7 relative-geometry branch, not a claim that every GRAIN
theoretical or V8 semantic/process component has a single fitted predictor.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.vector_mechanisms.external_heads import metrics
from scripts.prepare_grain_predecessor_reference import BASE, DEFAULT, digest, write
from scripts.run_grain_paraphrase_reference import E5_REVISION, encode_batches
from scripts.run_grain_predecessor_reference import fit_fold, paired_delta
from suica_core.v7_geometry import GeometryBundle, score_geometry_bundle
from suica_core.v7_observations import (
    ObservationSpec,
    build_observations,
    canonicalize_comments,
)
from suica_core.v7_psychometric import author_features_from_embeddings

E1 = ROOT / "results/v7_multiview_projection/e1_v72_full_20260715"
GEO = ROOT / "results/v7_geometry/g1_corrected_v2_full_20260715/geometry_bundle.json"
READY = "GEOMETRY_PROFILE_READY"


def native_observations(bundles, config):
    """Apply the actual legacy observation operator, retaining source-row links."""
    rows = []
    for b in bundles:
        if b["source_group"] != "big5_only":
            raise ValueError("Only the authorized Big5-only material is permitted")
        for v in b["views"]:
            for c in v:
                if not np.isfinite(c["timestamp"]):
                    raise ValueError("Native chronological sampling needs timestamps")
                rows.append(
                    {
                        "author": b["author_id"],
                        "body": c["body"],
                        "created_utc": c["timestamp"],
                        "subreddit": c["subreddit"],
                    }
                )
    frame = canonicalize_comments(
        pd.DataFrame(rows), min_tokens=config["min_tokens_per_unit"]
    )
    frame["split"] = "predecessor_development"
    raw = next(o for o in config["operators"] if o["name"] == "native")
    spec = ObservationSpec(
        **raw,
        min_tokens=config["min_tokens_per_unit"],
        max_units_per_user=config["max_units_per_user"],
        max_source_comments_per_user=config["max_source_comments_per_user"],
        max_source_tokens_per_user=config["max_source_tokens_per_user"],
    )
    return build_observations(frame, spec), spec


def align(frame, author_ids):
    """Require one complete feature row per author in the original order."""
    if frame.user_id.duplicated().any() or set(frame.user_id) != set(author_ids):
        raise ValueError("Author-feature mismatch")
    return frame.set_index("user_id").loc[author_ids].reset_index()


def run(args):
    """Compare source-fitted heads only where the frozen geometry permits scoring."""
    out = args.output
    if out.exists():
        raise ValueError("Use a fresh output directory")
    out.mkdir(parents=True)
    manifest = json.loads((args.material / "material_manifest.json").read_text())
    for name, sha in manifest["files"].items():
        if digest(args.material / name) != sha:
            raise ValueError("Changed material")
    data = json.loads((args.material / "private_material.json").read_text())
    with np.load(args.material / "private_arrays.npz", allow_pickle=False) as z:
        ids, folds, raw36 = z["author_ids"], z["folds"], z["grain"]
    geometry = GeometryBundle.from_dict(json.loads(GEO.read_text()))
    config = json.loads((E1 / "run_manifest.json").read_text())["config"]
    runtime_path = E1 / "artifacts/common_source_comment_representation.joblib"
    preflight = BASE / "vm05_preflight_r1"
    previous = json.loads((preflight / "manifest.json").read_text())
    if (
        digest(GEO) != previous["geometry_sha256"]
        or digest(runtime_path) != previous["runtime_sha256"]
        or digest(E1 / "author_features_native.csv")
        != geometry.reference_population["cohort_commitment"]
    ):
        raise ValueError("Frozen scorer provenance changed")
    exposure = pd.read_csv(preflight / "budget_by_author.csv")
    exposure = exposure[exposure.author_id.isin(ids)]
    if (
        exposure.author_id.nunique() != len(ids)
        or exposure.old_reference_member.any()
        or exposure.old_e1_member.any()
    ):
        raise ValueError("Current authors overlap the old GRAIN reference/evaluation")
    runtime = joblib.load(runtime_path)
    observations, spec = native_observations(data["bundles"], config)
    h = runtime.transform(observations.text.tolist())
    native = align(author_features_from_embeddings(observations, h), ids)
    x = native[geometry.feature_names].to_numpy(float)
    native_score = score_geometry_bundle(
        geometry, x, unit_counts=native.n_units.to_numpy()
    )
    raw_score = score_geometry_bundle(
        geometry, raw36, unit_counts=np.full(len(ids), 36)
    )
    support = native[["user_id", "n_units", "n_tokens"]].copy()
    support["raw36_status"] = raw_score["status"]
    support["native_status"] = native_score["status"]
    support["raw36_radius"] = raw_score["reference_radius"]
    support["native_radius"] = native_score["reference_radius"]
    support.to_csv(out / "private_support.csv", index=False)
    observations.to_csv(out / "private_native_observations.csv", index=False)
    eligible = np.asarray(native_score["status"]) == READY
    write(
        out / "contract.json",
        {
            "material_manifest": digest(args.material / "material_manifest.json"),
            "geometry": digest(GEO),
            "runtime": digest(runtime_path),
            "legacy_config": digest(E1 / "run_manifest.json"),
            "script": digest(Path(__file__)),
            "observation_code": digest(ROOT / "suica_core/v7_observations.py"),
            "geometry_code": digest(ROOT / "suica_core/v7_geometry.py"),
            "reference_code": digest(
                ROOT / "scripts/run_grain_predecessor_reference.py"
            ),
            "observation_spec": spec.to_dict(),
            "authors": len(ids),
            "native_eligible": int(eligible.sum()),
            "source_exposure": "zero overlap with original 240 E1 authors, including 141 discovery authors",
            "matching": "All three heads receive the same native texts, authors and outer/inner folds; frozen GRAIN fit is external to current authors.",
            "scope": "Frozen native geometry branch, not a universal full-GRAIN predictor; previously explored development cohort.",
            "personality_labels_read_for_support": False,
            "old_queue_restarted": False,
        },
    )
    summary = {
        "authors": len(ids),
        "raw36_support": pd.Series(raw_score["status"]).value_counts().to_dict(),
        "native_support": pd.Series(native_score["status"]).value_counts().to_dict(),
        "native_units": native.n_units.describe().to_dict(),
        "native_tokens": native.n_tokens.describe().to_dict(),
        "median_feature_change_norm": float(
            np.median(np.linalg.norm(x - raw36, axis=1))
        ),
        "support_threshold_unchanged": native_score["support_radius_threshold"],
    }
    print(json.dumps(summary, indent=2), flush=True)
    # Refused profiles remain diagnostics. Never turn them into deployable scores.
    if args.support_only or min(np.bincount(folds[eligible], minlength=5)) < 10:
        write(out / "summary.json", summary | {"status": "SUPPORT_ONLY"})
        return
    selected_ids = ids[eligible]
    selected = observations[observations.user_id.isin(selected_ids)].copy()
    vectors, encoding = encode_batches(
        selected.text.tolist(),
        out / "e5_batches",
        "intfloat/e5-large-v2",
        E5_REVISION,
        prefix="query: ",
        cap=512,
    )
    if encoding["truncated_texts"]:
        raise ValueError("Matched evidence comparison forbids hidden truncation")
    semantic = np.stack(
        [vectors[selected.user_id.to_numpy() == uid].mean(0) for uid in selected_ids]
    )
    with np.load(args.material / "private_arrays.npz", allow_pickle=False) as z:
        y = z["y"][eligible]
    f = folds[eligible]
    variants = {
        "native_raw48": ("grain_ridge", x[eligible]),
        "native_geometry16": (
            "grain_ridge",
            native_score["landmark_distance_profile"][eligible],
        ),
        "native_e5mean": ("semantic_pca_ridge", semantic),
    }
    predictions, rows, fits = {}, [], []
    with threadpool_limits(limits=2):
        for name, (method, values) in variants.items():
            sources = {"grain": values, "semantic": values}
            prediction = np.full_like(y, np.nan)
            for fold in range(5):
                train, test = np.flatnonzero(f != fold), np.flatnonzero(f == fold)
                feature, head, pred, tuning = fit_fold(sources, y, train, test, method)
                prediction[test] = pred
                joblib.dump((feature, head), out / f"{name}_fold{fold}.joblib")
                fits.append({"method": name, "fold": fold, **tuning})
            predictions[name] = prediction
            rows.extend(
                {"method": name, **row} for row in metrics(y, prediction, "big5")
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "metrics_by_trait.csv", index=False)
    contrasts = [
        {"method": a, "reference": b, **paired_delta(y, predictions[a], predictions[b])}
        for a, b in (
            ("native_geometry16", "native_raw48"),
            ("native_geometry16", "native_e5mean"),
        )
    ]
    pd.DataFrame(contrasts).to_csv(out / "paired_comparisons.csv", index=False)
    write(out / "fit_log.json", fits)
    np.savez_compressed(
        out / "private_oof.npz", y=y, author_ids=selected_ids, folds=f, **predictions
    )
    summary.update(
        status="COMPLETE",
        eligible_fold_counts=np.bincount(f, minlength=5).tolist(),
        matched_mean_r=frame.groupby("method").pearson.mean().to_dict(),
        comparison_population="intersection of frozen native radial support and current source cohort",
        full_cohort_prediction_claim=False,
        encoding=encoding,
    )
    write(out / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    """Run the compatibility replay, optionally stopping before any encoder call."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--material", type=Path, default=DEFAULT)
    p.add_argument(
        "--output",
        type=Path,
        default=BASE / "predecessor_integration_v1/native_grain_r1",
    )
    p.add_argument("--support-only", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
