#!/usr/bin/env python3
"""Bounded nested source readouts with fixed dimension-matched random partitions."""

from __future__ import annotations

import argparse
import fcntl
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
from grain_research.vector_mechanisms.external_geometry_coordinates import (
    coordinate_channels,
)
from grain_research.vector_mechanisms.external_heads import (
    AXES,
    predict_head,
    task_score,
)
from grain_research.vector_mechanisms.external_random_subspaces import (
    orthogonal_partition,
    random_subspace_grid,
)
from grain_research.vector_mechanisms.inventory import archive_source
from scripts.run_grain_vector_mechanism_controls import write_json
from scripts.run_grain_vector_profile_heads import nested_fit
from scripts.summarize_grain_vector_external_links import load_oof


def sha(path):
    """Hash local provenance without exporting source records."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_result(folder, identity):
    """Accept only complete cases with intact saved predictions and models."""
    s = json.loads((folder / "summary.json").read_text())
    if s["status"] != "COMPLETE" or s["batch_identity"] != identity:
        raise ValueError("Incomplete or incompatible random control")
    for name, digest in s["artifacts"].items():
        if sha(folder / name) != digest:
            raise ValueError("Changed random-control artifact")
    return s


def main():
    """Fit only existing source outcomes; keep bridge labels and scorer unchanged."""
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("profile-dir", "geometry-batch", "views", "output-dir"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--max-seconds", type=int, default=1200)
    p.add_argument("--resume-completed", action="store_true")
    a = p.parse_args()
    if not 60 <= a.max_seconds <= 3600:
        raise ValueError("Require a bounded condition-boundary runtime")
    start = time.monotonic()
    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / ".execution.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    profile = json.loads((a.profile_dir / "manifest.json").read_text())
    ps = json.loads((a.profile_dir / "summary.json").read_text())
    batch = json.loads((a.geometry_batch / "batch_manifest.json").read_text())
    bs = json.loads((a.geometry_batch / "summary.json").read_text())
    source = json.loads(a.views.read_text())
    geometry_path = (
        ROOT / "results/v7_geometry/g1_corrected_v2_full_20260715/geometry_bundle.json"
    )
    npz_path = a.profile_dir / "private_profile_features.npz"
    if (
        ps["status"] != "COMPLETE"
        or bs["status"] != "COMPLETE"
        or material_hash({k: v for k, v in batch.items() if k != "identity"})
        != batch["identity"]
        or material_hash({k: v for k, v in profile.items() if k != "identity"})
        != profile["identity"]
        or ps["parent_identity"] != profile["identity"]
        or batch["profile_identity"] != profile["identity"]
        or batch["cohort"] != "raw_ready"
        or batch["family"] != "geometry_information"
        or batch["material_sha256"] != material_hash(source)
        or batch["profile_npz_sha256"] != sha(npz_path)
        or profile["geometry_sha256"] != sha(geometry_path)
    ):
        raise ValueError("Changed source geometry or profile evidence")
    with np.load(npz_path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    if arrays["author_ids"].tolist() != [b["author_id"] for b in source]:
        raise ValueError("Changed source row order")
    geometry = json.loads(geometry_path.read_text())
    values, _ = coordinate_channels(
        arrays["raw_whitened"],
        arrays["raw_canonical_distances"],
        geometry["reference_landmarks"],
    )
    q = arrays["raw_whitened"] - np.asarray(geometry["reference_landmarks"])[0]
    targets, actual, reference_records = {}, {}, []
    with threadpool_limits(limits=4):
        for task in ("big5", *AXES):
            group = "big5_only" if task == "big5" else "mbti_only"
            ix = [
                i
                for i, b in enumerate(source)
                if b["source_group"] == group and arrays["raw_ready"][i]
            ]
            bundles = [source[i] for i in ix]
            for channel in ("parallel", "perpendicular", "whitened"):
                folder = a.geometry_batch / f"{task}_raw_{channel}_no_pca"
                result = load_oof(folder)
                m = result["manifest"]
                original_x = values[channel][ix]
                if (
                    m["batch_identity"] != batch["identity"]
                    or m["task"] != task
                    or m["pca"]
                    or m["arm"] != "raw"
                    or m["feature"] != channel
                    or m["source_material_sha256"] != material_hash(bundles)
                    or m["feature_input_sha256"]
                    != hashlib.sha256(
                        np.ascontiguousarray(original_x).tobytes()
                    ).hexdigest()
                ):
                    raise ValueError("Changed source reference features")
                lookup = {b["author_id"]: i for i, b in enumerate(bundles)}
                if set(result["ids"]) != set(lookup):
                    raise ValueError("Different source support")
                order = [lookup[u] for u in result["ids"]]
                if not np.array_equal(
                    result["folds"], [bundles[i]["official_folds"][task] for i in order]
                ):
                    raise ValueError("Different official folds")
                previous = targets.get(task)
                if previous is not None and any(
                    not np.array_equal(previous[k], result[k])
                    for k in ("ids", "y", "folds")
                ):
                    raise ValueError("Unpaired reference outcomes")
                targets[task] = result | {"q": q[ix][order]}
                for fold in range(5):
                    model_path = folder / f"private_fold_{fold}.joblib"
                    model = joblib.load(model_path)
                    test = result["folds"] == fold
                    error = float(
                        np.max(
                            abs(
                                predict_head(model, original_x[order][test], task)
                                - result["p"][test]
                            )
                        )
                    )
                    if not np.isfinite(error) or error > 1e-10:
                        raise ValueError("Reference model replay failed")
                    reference_records.append(
                        {
                            "task": task,
                            "channel": channel,
                            "fold": fold,
                            "max_error": error,
                            "model_sha256": sha(model_path),
                        }
                    )
                actual[(task, channel)] = result["summary"]["score"]
    code_paths = [
        Path(__file__),
        ROOT / "grain_research/vector_mechanisms/external_random_subspaces.py",
        ROOT / "grain_research/vector_mechanisms/external_heads.py",
        ROOT / "grain_research/vector_mechanisms/external_geometry_coordinates.py",
        ROOT / "scripts/run_grain_vector_profile_heads.py",
        ROOT / "scripts/summarize_grain_vector_external_links.py",
    ]
    grid = random_subspace_grid(("big5", *AXES))
    manifest = {
        "cases": grid,
        "source_batch_identity": batch["identity"],
        "profile_npz_sha256": sha(npz_path),
        "geometry_sha256": sha(geometry_path),
        "material_sha256": material_hash(source),
        "seeds": list(range(94100, 94120)),
        "tasks": list(targets),
        "pca": False,
        "arm": "raw",
        "bridge_labels_read": False,
        "source_reference_manifests": {
            task: result["manifest"]["identity"] for task, result in targets.items()
        },
        "code": {str(path.relative_to(ROOT)): sha(path) for path in code_paths},
    }
    manifest["identity"] = material_hash(manifest)
    if a.resume_completed:
        if json.loads((out / "batch_manifest.json").read_text()) != manifest:
            raise ValueError("Changed random-control resume identity")
    else:
        if any(
            f
            for f in out.iterdir()
            if not f.name.startswith("._") and f.name != ".execution.lock"
        ):
            raise ValueError("Use a fresh output or explicitly resume complete cases")
        write_json(out / "batch_manifest.json", manifest)
        archive_source(ROOT, out / "source_snapshot.zip")
        pd.DataFrame(reference_records).to_csv(
            out / "reference_replay.csv", index=False
        )
    rows, all_predictions = [], {}
    with threadpool_limits(limits=4):
        for case in grid:
            folder = out / case["id"]
            if folder.exists():
                s = validate_result(folder, manifest["identity"])
                with np.load(
                    folder / "private_predictions.npz", allow_pickle=False
                ) as z:
                    pred = z["predictions"]
            else:
                if time.monotonic() - start >= a.max_seconds:
                    write_json(
                        out / "batch_status.json",
                        {
                            "status": "PAUSED_BUDGET",
                            "completed": len(rows),
                            "total": len(grid),
                        },
                    )
                    print("PAUSED_BUDGET", flush=True)
                    return
                write_json(
                    out / "batch_status.json",
                    {
                        "status": "RUNNING",
                        "current": case["id"],
                        "completed": len(rows),
                        "total": len(grid),
                    },
                )
                folder.mkdir()
                first, second = orthogonal_partition(48, 15, case["seed"])
                basis = first if case["rank"] == 15 else second
                target = targets[case["task"]]
                x = target["q"] @ basis.T

                def checkpoint(fold, model, selection, folder=folder):
                    """Preserve each completed source fold before starting the next."""
                    joblib.dump(model, folder / f"private_fold_{fold}.joblib")
                    write_json(folder / "selection.json", selection)

                tick = time.monotonic()
                pred, _selections, fold_metrics, full, search, parameter = nested_fit(
                    x,
                    target["y"],
                    target["ids"],
                    target["folds"],
                    case["task"],
                    False,
                    checkpoint,
                )
                error = 0.0
                for fold in range(5):
                    test = target["folds"] == fold
                    saved = joblib.load(folder / f"private_fold_{fold}.joblib")
                    error = max(
                        error,
                        float(
                            np.max(
                                abs(
                                    predict_head(saved, x[test], case["task"])
                                    - pred[test]
                                )
                            )
                        ),
                    )
                if not np.isfinite(pred).all() or error > 1e-10:
                    raise ValueError("Random-control saved model replay failed")
                joblib.dump(full, folder / "private_full_source_head.joblib")
                np.savez_compressed(
                    folder / "private_predictions.npz",
                    predictions=pred,
                    author_ids=target["ids"].astype(str),
                    folds=target["folds"],
                    basis=basis,
                )
                pd.DataFrame(fold_metrics).to_csv(
                    folder / "metrics_by_fold.csv", index=False
                )
                write_json(
                    folder / "full_source_selection.json",
                    {"parameter": parameter, "search": search},
                )
                s = {
                    "status": "COMPLETE",
                    "batch_identity": manifest["identity"],
                    **case,
                    "authors": len(pred),
                    "score": task_score(target["y"], pred, case["task"]),
                    "seconds": time.monotonic() - tick,
                    "saved_model_replay_error": error,
                    "feature_input_sha256": hashlib.sha256(
                        np.ascontiguousarray(x).tobytes()
                    ).hexdigest(),
                    "artifacts": {
                        f.name: sha(f)
                        for f in folder.iterdir()
                        if f.is_file() and not f.name.startswith("._")
                    },
                }
                write_json(folder / "summary.json", s)
            rows.append({k: v for k, v in s.items() if k != "artifacts"})
            all_predictions[(case["task"], case["seed"], case["rank"])] = pred
    frame = pd.DataFrame(rows)
    comparisons = []
    for task, target in targets.items():
        for rank, channel in ((15, "parallel"), (33, "perpendicular")):
            scores = frame.loc[
                (frame.task == task) & (frame["rank"] == rank), "score"
            ].to_numpy()
            ensemble = np.mean(
                [all_predictions[(task, seed, rank)] for seed in manifest["seeds"]],
                axis=0,
            )
            value = actual[(task, channel)]
            comparisons.append(
                {
                    "task": task,
                    "rank": rank,
                    "actual_channel": channel,
                    "authors": len(target["ids"]),
                    "actual_score": value,
                    "full_whitened_score": actual[(task, "whitened")],
                    "random_mean": float(scores.mean()),
                    "random_min": float(scores.min()),
                    "random_max": float(scores.max()),
                    "random_q025": float(np.quantile(scores, 0.025)),
                    "random_q975": float(np.quantile(scores, 0.975)),
                    "actual_minus_random_mean": value - float(scores.mean()),
                    "random_scores_below_actual": int(np.sum(scores < value)),
                    "random_directions": len(scores),
                    "random_prediction_ensemble_score": task_score(
                        target["y"], ensemble, task
                    ),
                    "scope": "orientation sensitivity, not a permutation p value; ensemble is not a single rank-matched representation",
                }
            )
    frame.to_csv(out / "scores_by_condition.csv", index=False)
    pd.DataFrame(comparisons).to_csv(out / "matched_rank_comparison.csv", index=False)
    paired = frame.pivot(index=["task", "seed"], columns="rank", values="score")
    paired["rank33_minus_rank15"] = paired[33] - paired[15]
    paired.to_csv(out / "paired_rank_contrasts.csv")
    summary = {
        "status": "COMPLETE",
        "batch_identity": manifest["identity"],
        "conditions": len(rows),
        "new_fold_models": 5 * len(rows),
        "new_full_source_heads": len(rows),
        "reference_models_replayed": len(reference_records),
        "max_saved_model_replay_error": max(
            r["saved_model_replay_error"] for r in rows
        ),
        "new_source_label_files_read": False,
        "bridge_labels_read": False,
        "scorer_changed": False,
        "seconds_this_invocation": time.monotonic() - start,
        "outputs_sha256": {
            name: sha(out / name)
            for name in (
                "batch_manifest.json",
                "source_snapshot.zip",
                "reference_replay.csv",
                "scores_by_condition.csv",
                "matched_rank_comparison.csv",
                "paired_rank_contrasts.csv",
            )
        },
    }
    write_json(out / "summary.json", summary)
    write_json(
        out / "batch_status.json",
        {"status": "COMPLETE", "completed": len(rows), "total": len(grid)},
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
