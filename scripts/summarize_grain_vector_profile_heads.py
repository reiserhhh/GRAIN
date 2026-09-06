#!/usr/bin/env python3
"""Replay every ready-profile fold model and same-author channel/arm contrast."""

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
from grain_research.vector_mechanisms.inventory import archive_source
from scripts.run_grain_vector_mechanism_controls import write_json
from scripts.summarize_grain_vector_external_links import load_oof, paired_delta


def main():
    """Validate stored model/data correspondence without refits or new labels."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-dir", type=Path, required=True)
    p.add_argument("--profile-dir", type=Path, required=True)
    p.add_argument("--views", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    batch = json.loads((a.batch_dir / "batch_manifest.json").read_text())
    bs = json.loads((a.batch_dir / "summary.json").read_text())
    ps = json.loads((a.profile_dir / "summary.json").read_text())
    material = json.loads(a.views.read_text())
    feature_path = a.profile_dir / "private_profile_features.npz"
    if (
        bs["status"] != "COMPLETE"
        or material_hash(material) != batch["material_sha256"]
        or ps["parent_identity"] != batch["profile_identity"]
        or hashlib.sha256(feature_path.read_bytes()).hexdigest()
        != batch["profile_npz_sha256"]
        or material_hash({k: v for k, v in batch.items() if k != "identity"})
        != batch["identity"]
    ):
        raise ValueError("Unsealed or changed profile source batch")
    with np.load(feature_path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    if arrays["author_ids"].tolist() != [b["author_id"] for b in material]:
        raise ValueError("Changed profile row order")
    family = batch.get("family", "profiles")
    if family == "geometry_information":
        from grain_research.vector_mechanisms.external_geometry_coordinates import (
            coordinate_channels,
        )

        profile_manifest = json.loads((a.profile_dir / "manifest.json").read_text())
        geometry_path = (
            ROOT
            / "results/v7_geometry/g1_corrected_v2_full_20260715/geometry_bundle.json"
        )
        if (
            hashlib.sha256(geometry_path.read_bytes()).hexdigest()
            != profile_manifest["geometry_sha256"]
        ):
            raise ValueError("Different frozen geometry at replay")
        landmarks = json.loads(geometry_path.read_text())["reference_landmarks"]
        for arm in sorted({case["arm"] for case in batch["cases"]}):
            values, _ = coordinate_channels(
                arrays[arm + "_whitened"],
                arrays[arm + "_canonical_distances"],
                landmarks,
                relative_tolerance=profile_manifest["relative_tolerance"],
            )
            arrays.update({arm + "_" + key: value for key, value in values.items()})
    objects, records, rows = {}, [], []
    with threadpool_limits(limits=4):
        for case in batch["cases"]:
            folder = a.batch_dir / case["id"]
            result = load_oof(folder)
            m = result["manifest"]
            group = "big5_only" if case["task"] == "big5" else "mbti_only"
            ix = [
                i
                for i, b in enumerate(material)
                if b["source_group"] == group and arrays[batch["cohort"]][i]
            ]
            bundles = [material[i] for i in ix]
            x = arrays[case["arm"] + "_" + case["channel"]][ix]
            if (
                m["batch_identity"] != batch["identity"]
                or m["task"] != case["task"]
                or m["feature"] != case["channel"]
                or m["pca"] != case["pca"]
                or m["arm"] != case["arm"]
                or m["source_material_sha256"] != material_hash(bundles)
                or m["feature_input_sha256"]
                != hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
                or hashlib.sha256(
                    (folder / "private_full_source_head.joblib").read_bytes()
                ).hexdigest()
                != result["summary"]["full_head_sha256"]
            ):
                raise ValueError("Different source channel, input or model artifact")
            lookup = {b["author_id"]: i for i, b in enumerate(bundles)}
            if set(result["ids"]) != set(lookup):
                raise ValueError("OOF author set differs from frozen ready source")
            order = [lookup[uid] for uid in result["ids"]]
            if not np.array_equal(
                result["folds"],
                [bundles[i]["official_folds"][case["task"]] for i in order],
            ):
                raise ValueError("Incorrect official source folds")
            x = x[order]
            selections = json.loads((folder / "selection.json").read_text())
            if len(selections) != 5:
                raise ValueError("Missing source selection records")
            for fold in range(5):
                test = result["folds"] == fold
                selection = selections[fold]
                model = joblib.load(folder / f"private_fold_{fold}.joblib")
                if (
                    selection["fold"] != fold
                    or selection["test_n"] != int(test.sum())
                    or selection["train_n"] != int((~test).sum())
                    or ("pca" in model.named_steps) != case["pca"]
                ):
                    raise ValueError("Different fold support or model complexity")
                pred = predict_head(model, x[test], case["task"])
                error = float(np.max(abs(pred - result["p"][test])))
                if not np.isfinite(error) or error > 1e-10:
                    raise ValueError("Held-out prediction fails saved-model replay")
                records.append(
                    {
                        "case": case["id"],
                        "fold": fold,
                        "max_absolute_error": error,
                        "train_n": selection["train_n"],
                        "test_n": selection["test_n"],
                        "projected_dimensions": model.named_steps["pca"].n_components_
                        if case["pca"]
                        else x.shape[1],
                    }
                )
            objects[(case["task"], case["arm"], case["channel"], case["pca"])] = result
            rows.append(
                {
                    **case,
                    "authors": len(result["ids"]),
                    "pooled_score": result["summary"]["score"],
                    "mean_fold_score": result["mean_fold_score"],
                    "intercept_only_pooled_score": result[
                        "intercept_only_pooled_score"
                    ],
                }
            )
    contrasts = []
    baseline_name = "intermediate" if family == "profiles" else "whitened"
    for (task, arm, channel, pca), result in objects.items():
        refs = (
            [("versus_" + baseline_name, (task, arm, baseline_name, pca))]
            if channel != baseline_name
            else []
        )
        if arm == "cue_mask":
            refs += [
                ("cue_minus_placebo", (task, "placebo_mask", channel, pca)),
                ("cue_minus_raw", (task, "raw", channel, pca)),
            ]
        for name, key in refs:
            contrasts.append(
                {
                    "contrast": name,
                    "task": task,
                    "arm": arm,
                    "channel": channel,
                    "pca": pca,
                    **paired_delta(result, objects[key]),
                }
            )
    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if any(q for q in out.iterdir() if not q.name.startswith("._")):
        raise ValueError("Use a fresh profile audit directory")
    archive_source(ROOT, out / "source_snapshot.zip")
    pd.DataFrame(rows).to_csv(out / "verified_comparison.csv", index=False)
    pd.DataFrame(records).to_csv(out / "replay_by_fold.csv", index=False)
    pd.DataFrame(contrasts).to_csv(out / "paired_comparison.csv", index=False)
    summary = {
        "status": "COMPLETE",
        "batch_identity": batch["identity"],
        "verified_conditions": len(rows),
        "verified_fold_models": len(records),
        "max_absolute_error": max(r["max_absolute_error"] for r in records),
        "contrasts": len(contrasts),
        "new_label_values_read": False,
        "model_refits": 0,
        "bridge_labels_read": False,
    }
    write_json(out / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
