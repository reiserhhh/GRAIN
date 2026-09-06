#!/usr/bin/env python3
"""Independently rebuild random bases and replay all completed held-out heads."""

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
from grain_research.vector_mechanisms.external_heads import predict_head, task_score
from grain_research.vector_mechanisms.external_random_subspaces import (
    orthogonal_partition,
)
from grain_research.vector_mechanisms.inventory import archive_source
from scripts.run_grain_vector_mechanism_controls import write_json
from scripts.run_grain_vector_random_subspaces import sha, validate_result
from scripts.summarize_grain_vector_external_links import load_oof


def main():
    """Verify saved evidence, including orientation and rows, without refitting."""
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("batch-dir", "geometry-batch", "profile-dir", "views", "output-dir"):
        p.add_argument("--" + key, type=Path, required=True)
    a = p.parse_args()
    batch = json.loads((a.batch_dir / "batch_manifest.json").read_text())
    summary = json.loads((a.batch_dir / "summary.json").read_text())
    material = json.loads(a.views.read_text())
    feature_path = a.profile_dir / "private_profile_features.npz"
    geometry_path = (
        ROOT / "results/v7_geometry/g1_corrected_v2_full_20260715/geometry_bundle.json"
    )
    geometry_batch = json.loads((a.geometry_batch / "batch_manifest.json").read_text())
    if (
        summary["status"] != "COMPLETE"
        or summary["conditions"] != len(batch["cases"])
        or summary["batch_identity"] != batch["identity"]
        or batch["identity"]
        != material_hash({k: v for k, v in batch.items() if k != "identity"})
        or batch["source_batch_identity"] != geometry_batch["identity"]
        or batch["material_sha256"] != material_hash(material)
        or batch["profile_npz_sha256"] != sha(feature_path)
        or batch["geometry_sha256"] != sha(geometry_path)
    ):
        raise ValueError("Changed random-control source evidence")
    for name, digest in summary["outputs_sha256"].items():
        if sha(a.batch_dir / name) != digest:
            raise ValueError("Changed aggregate random-control artifact")
    with np.load(feature_path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    if arrays["author_ids"].tolist() != [b["author_id"] for b in material]:
        raise ValueError("Feature row order mismatch")
    geometry = json.loads(geometry_path.read_text())
    q = arrays["raw_whitened"] - np.asarray(geometry["reference_landmarks"])[0]
    targets, coordinates = {}, {}
    for task in batch["tasks"]:
        result = load_oof(a.geometry_batch / f"{task}_raw_whitened_no_pca")
        if result["manifest"]["identity"] != batch["source_reference_manifests"][task]:
            raise ValueError("Different source reference labels")
        group = "big5_only" if task == "big5" else "mbti_only"
        ix = {
            b["author_id"]: i
            for i, b in enumerate(material)
            if b["source_group"] == group and arrays["raw_ready"][i]
        }
        if set(ix) != set(result["ids"]):
            raise ValueError("Different source-exclusive ready author set")
        order = [ix[u] for u in result["ids"]]
        if not np.array_equal(
            result["folds"], [material[i]["official_folds"][task] for i in order]
        ):
            raise ValueError("Different official fold assignment")
        targets[task], coordinates[task] = result, q[order]
    records, metrics = [], []
    with threadpool_limits(limits=4):
        for case in batch["cases"]:
            folder = a.batch_dir / case["id"]
            s = validate_result(folder, batch["identity"])
            if any(s[k] != case[k] for k in case):
                raise ValueError("Mismatched case identity")
            target = targets[case["task"]]
            first, second = orthogonal_partition(48, 15, case["seed"])
            basis = first if case["rank"] == 15 else second
            x = coordinates[case["task"]] @ basis.T
            with np.load(folder / "private_predictions.npz", allow_pickle=False) as z:
                pred = z["predictions"]
                if (
                    not np.array_equal(z["basis"], basis)
                    or not np.array_equal(z["author_ids"], target["ids"].astype(str))
                    or not np.array_equal(z["folds"], target["folds"])
                ):
                    raise ValueError("Changed random basis or target alignment")
            if (
                s["feature_input_sha256"]
                != hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
            ):
                raise ValueError("Random projected features do not reproduce")
            score = task_score(target["y"], pred, case["task"])
            if not np.isclose(score, s["score"], rtol=0, atol=1e-12):
                raise ValueError("OOF metric does not reproduce")
            for fold in range(5):
                test = target["folds"] == fold
                model = joblib.load(folder / f"private_fold_{fold}.joblib")
                if "pca" in model.named_steps or model.n_features_in_ != case["rank"]:
                    raise ValueError("Changed random-control complexity")
                replay = predict_head(model, x[test], case["task"])
                error = float(np.max(abs(replay - pred[test])))
                if not np.isfinite(error) or error > 1e-10:
                    raise ValueError("Independent saved-model replay failed")
                records.append({**case, "fold": fold, "max_absolute_error": error})
            metrics.append({**case, "score": score, "authors": len(pred)})
    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if any(f for f in out.iterdir() if not f.name.startswith("._")):
        raise ValueError("Use a fresh independent audit directory")
    archive_source(ROOT, out / "source_snapshot.zip")
    pd.DataFrame(records).to_csv(out / "independent_replay.csv", index=False)
    pd.DataFrame(metrics).to_csv(out / "verified_scores.csv", index=False)
    result = {
        "status": "COMPLETE",
        "batch_identity": batch["identity"],
        "verified_conditions": len(metrics),
        "verified_fold_models": len(records),
        "max_absolute_error": max(r["max_absolute_error"] for r in records),
        "random_bases_rebuilt": True,
        "new_label_values_read": False,
        "bridge_labels_read": False,
        "refits": 0,
        "outputs_sha256": {
            f.name: sha(f)
            for f in out.iterdir()
            if f.is_file() and not f.name.startswith("._")
        },
    }
    write_json(out / "summary.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
