#!/usr/bin/env python3
"""Transport saved source heads exactly and compare naive coordinate refits."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.external_heads import predict_head
from grain_research.vector_mechanisms.external_readout_transport import (
    collapse_dense_pipeline,
    landmark_radial_lift,
    pullback_readout,
    transport_readout,
)
from grain_research.vector_mechanisms.inventory import archive_source
from scripts.run_grain_vector_mechanism_controls import write_json
from scripts.summarize_grain_vector_external_links import load_oof, paired_delta


def sha(path):
    """Hash a local artifact used by this read-only audit."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    """Replay frozen models without refits, new labels or bridge outcomes."""
    p = argparse.ArgumentParser(description=__doc__)
    for key in (
        "profile-dir",
        "profile-batch",
        "geometry-batch",
        "views",
        "output-dir",
    ):
        p.add_argument("--" + key, type=Path, required=True)
    a = p.parse_args()
    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if any(f for f in out.iterdir() if not f.name.startswith("._")):
        raise ValueError("Use a fresh transport audit directory")
    material = json.loads(a.views.read_text())
    feature_path = a.profile_dir / "private_profile_features.npz"
    profile = json.loads((a.profile_dir / "manifest.json").read_text())
    profile_summary = json.loads((a.profile_dir / "summary.json").read_text())
    geometry_path = (
        ROOT / "results/v7_geometry/g1_corrected_v2_full_20260715/geometry_bundle.json"
    )
    if sha(geometry_path) != profile["geometry_sha256"]:
        raise ValueError("Changed frozen geometry")
    geometry = json.loads(geometry_path.read_text())
    with np.load(feature_path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    if arrays["author_ids"].tolist() != [b["author_id"] for b in material]:
        raise ValueError("Changed feature row order")
    x, q = arrays["raw_intermediate"], arrays["raw_whitened"]
    w, center = (
        np.asarray(geometry["metric_whitener"]),
        np.asarray(geometry["feature_center"]),
    )
    lifted, lm, lo = landmark_radial_lift(
        q,
        geometry["reference_landmarks"],
        relative_tolerance=profile["relative_tolerance"],
    )
    squared = arrays["raw_canonical_distances"] ** 2
    white_error = float(np.max(abs((x - center) @ w - q)))
    lift_error = float(np.max(abs(lifted @ lm + lo - squared)))
    if white_error > 1e-10 or lift_error > 1e-9:
        raise ValueError("Coordinate identity does not replay")
    archive_source(ROOT, out / "source_snapshot.zip")
    records, objects, sealed_maps, identities = [], {}, {}, []
    with threadpool_limits(limits=4):
        for folder, family, target in (
            (a.profile_batch, "profiles", "intermediate"),
            (a.geometry_batch, "geometry_information", "squared_distances"),
        ):
            batch = json.loads((folder / "batch_manifest.json").read_text())
            summary = json.loads((folder / "summary.json").read_text())
            if (
                summary["status"] != "COMPLETE"
                or batch.get("family", "profiles") != family
                or batch["cohort"] != "raw_ready"
                or batch["material_sha256"] != material_hash(material)
                or batch["profile_identity"] != profile_summary["parent_identity"]
                or batch["profile_npz_sha256"] != sha(feature_path)
                or material_hash({k: v for k, v in batch.items() if k != "identity"})
                != batch["identity"]
            ):
                raise ValueError("Changed or unsupported source batch")
            identities.append(batch["identity"])
            for case in batch["cases"]:
                if case["channel"] not in {target, "whitened"}:
                    continue
                run = folder / case["id"]
                result = load_oof(run)
                m = result["manifest"]
                group = "big5_only" if case["task"] == "big5" else "mbti_only"
                ix = [
                    i
                    for i, b in enumerate(material)
                    if b["source_group"] == group and arrays["raw_ready"][i]
                ]
                bundles = [material[i] for i in ix]
                old_all = {
                    "intermediate": x,
                    "whitened": q,
                    "squared_distances": squared,
                }[case["channel"]]
                old = old_all[ix]
                if (
                    m["batch_identity"] != batch["identity"]
                    or m["arm"] != "raw"
                    or case["arm"] != "raw"
                    or m["task"] != case["task"]
                    or m["pca"] != case["pca"]
                    or m["feature"] != case["channel"]
                    or m["source_material_sha256"] != material_hash(bundles)
                    or m["feature_input_sha256"]
                    != hashlib.sha256(np.ascontiguousarray(old).tobytes()).hexdigest()
                    or sha(run / "private_full_source_head.joblib")
                    != result["summary"]["full_head_sha256"]
                ):
                    raise ValueError("Different source membership, features or readout")
                lookup = {b["author_id"]: i for i, b in enumerate(bundles)}
                if set(result["ids"]) != set(lookup):
                    raise ValueError("Different ready author set")
                order = [lookup[u] for u in result["ids"]]
                if not np.array_equal(
                    result["folds"],
                    [bundles[i]["official_folds"][case["task"]] for i in order],
                ):
                    raise ValueError("Different official folds")
                old = old[order]
                new = (q if target == "intermediate" else lifted)[ix][order]
                objects[(case["task"], case["channel"], case["pca"])] = result
                if case["channel"] == "whitened":
                    continue
                for fold in [*range(5), "full_source"]:
                    path = run / (
                        f"private_fold_{fold}.joblib"
                        if isinstance(fold, int)
                        else "private_full_source_head.joblib"
                    )
                    model = joblib.load(path)
                    if ("pca" in model.named_steps) != case["pca"]:
                        raise ValueError("Changed PCA condition")
                    test = (
                        result["folds"] == fold
                        if isinstance(fold, int)
                        else np.ones(len(old), bool)
                    )
                    collapsed = collapse_dense_pipeline(model)
                    moved = (
                        transport_readout(collapsed, w, -center @ w)
                        if target == "intermediate"
                        else pullback_readout(collapsed, lm, lo)
                    )
                    predicted = predict_head(model, old[test], case["task"])
                    collapse_error = float(
                        np.max(abs(collapsed.predict(old[test]) - predicted))
                    )
                    transport_error = float(
                        np.max(abs(moved.predict(new[test]) - predicted))
                    )
                    replay_error = (
                        float(np.max(abs(predicted - result["p"][test])))
                        if isinstance(fold, int)
                        else 0.0
                    )
                    if max(collapse_error, transport_error, replay_error) > 1e-9:
                        raise ValueError("Failed exact readout replay or transport")
                    key = case["id"] + "_" + str(fold)
                    sealed_maps[key] = moved
                    records.append(
                        {
                            "case": case["id"],
                            "task": case["task"],
                            "pca": case["pca"],
                            "fold": fold,
                            "map": "intermediate_to_whitened"
                            if target == "intermediate"
                            else "squared_distances_to_total_radial_lift",
                            "authors": int(test.sum()),
                            "model_sha256": sha(path),
                            "collapse_max_error": collapse_error,
                            "transport_max_error": transport_error,
                            "oof_replay_max_error": replay_error,
                            "full_source_is_in_sample_parity_only": not isinstance(
                                fold, int
                            ),
                        }
                    )
    contrasts = []
    for (task, channel, pca), original in objects.items():
        if channel != "intermediate":
            continue
        naive = objects[(task, "whitened", pca)]
        contrasts.append(
            {
                "task": task,
                "pca": pca,
                "original_and_transported_score": original["summary"]["score"],
                "naively_refitted_whitened_score": naive["summary"]["score"],
                **paired_delta(naive, original),
            }
        )
    joblib.dump(sealed_maps, out / "private_transported_readouts.joblib")
    pd.DataFrame(records).to_csv(out / "transport_replay.csv", index=False)
    pd.DataFrame(contrasts).to_csv(out / "naive_refit_comparison.csv", index=False)
    result = {
        "status": "COMPLETE",
        "batch_identities": identities,
        "frozen_geometry_sha256": sha(geometry_path),
        "profile_npz_sha256": sha(feature_path),
        "verified_fold_models": sum(isinstance(r["fold"], int) for r in records),
        "verified_full_source_heads": sum(
            not isinstance(r["fold"], int) for r in records
        ),
        "max_transport_error": max(r["transport_max_error"] for r in records),
        "max_oof_replay_error": max(r["oof_replay_max_error"] for r in records),
        "all_material_coordinate_checks_authors": len(material),
        "whitening_coordinate_max_error": white_error,
        "total_radial_lift_distance_squared_max_error": lift_error,
        "whitener_condition_number": float(np.linalg.cond(w)),
        "lift_distance_map_rank": int(np.linalg.matrix_rank(lm)),
        "lift_distance_map_condition_number": float(np.linalg.cond(lm)),
        "new_label_values_read": False,
        "bridge_labels_read": False,
        "refits": 0,
        "support_envelope_changed": False,
        "contrasts": len(contrasts),
        "interpretation": "Coordinate identities and fitted-readout parity, not a new predictive result or personality mechanism",
        "artifacts": {
            p.name: sha(p)
            for p in out.iterdir()
            if p.is_file() and not p.name.startswith("._")
        },
    }
    write_json(out / "summary.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
