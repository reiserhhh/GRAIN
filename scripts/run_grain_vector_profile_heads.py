#!/usr/bin/env python3
"""Source-only nested readouts of frozen profile channels on identical ready authors."""

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
    predict_head,
    select_head,
    task_score,
)
from grain_research.vector_mechanisms.inventory import archive_source
from scripts.run_grain_vector_mechanism_controls import write_json
from scripts.run_grain_vector_mechanism_external_links import bootstrap_score
from scripts.summarize_grain_vector_external_links import load_oof, paired_delta

CHANNELS = (
    "intermediate",
    "canonical_distances",
    "sorted_distances",
    "canonical_shape",
)


def condition_grid(cohort):
    """Freeze all transformations, tasks and complexity sensitivities before fitting."""
    arms = ("raw",) if cohort == "raw_ready" else ("raw", "cue_mask", "placebo_mask")
    return [
        {
            "id": f"{task}_{arm}_{channel}_{'pca' if pca else 'no_pca'}",
            "task": task,
            "arm": arm,
            "channel": channel,
            "pca": pca,
        }
        for task in ("big5", *AXES)
        for arm in arms
        for pca in (True, False)
        for channel in CHANNELS
    ]


def nested_fit(x, y, ids, folds, task, pca, checkpoint):
    """Use unchanged source heads; no outer outcome affects parameter selection."""
    if (
        len(ids) != len(set(ids))
        or len(ids) != len(folds)
        or set(folds) != set(range(5))
    ):
        raise ValueError(
            "Invalid unique source authors or official five-fold partition"
        )
    predictions = np.full(y.shape, np.nan)
    selection, fold_metrics = [], []
    for fold in range(5):
        train, test = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        model, search, parameter = select_head(
            x[train], y[train], ids[train], "dense", task, pca=pca
        )
        predictions[test] = predict_head(model, x[test], task)
        item = {
            "fold": fold,
            "train_n": len(train),
            "test_n": len(test),
            "parameter": parameter,
            "inner_search": search,
        }
        selection.append(item)
        fold_metrics.extend(
            dict(r, fold=fold) for r in metrics(y[test], predictions[test], task)
        )
        checkpoint(fold, model, selection)
    full, search, parameter = select_head(x, y, ids, "dense", task, pca=pca)
    return predictions, selection, fold_metrics, full, search, parameter


def main():
    """Complete bounded channel conditions and seal source heads, never bridge outcomes."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile-dir", type=Path, required=True)
    p.add_argument("--views", type=Path, required=True)
    p.add_argument("--parent-dir", type=Path, required=True)
    p.add_argument("--prepared-root", type=Path, required=True)
    p.add_argument(
        "--cohort", choices=("raw_ready", "all_arms_ready"), default="raw_ready"
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--resume-completed", action="store_true")
    p.add_argument("--max-seconds", type=int, default=3600)
    a = p.parse_args()
    if not 60 <= a.max_seconds <= 7200:
        raise ValueError("Require bounded profile readout runtime")
    import fcntl

    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / ".execution.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pm = json.loads((a.profile_dir / "manifest.json").read_text())
    ps = json.loads((a.profile_dir / "summary.json").read_text())
    if (
        ps["status"] != "COMPLETE"
        or ps["parent_identity"] != pm["identity"]
        or pm["labels_read"]
        or material_hash({k: v for k, v in pm.items() if k != "identity"})
        != pm["identity"]
    ):
        raise ValueError("Profile material is not sealed label-free")
    for name, digest in ps["outputs_sha256"].items():
        if hashlib.sha256((a.profile_dir / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Frozen profile output changed")
    source = json.loads(a.views.read_text())
    if material_hash(source) != pm["material_sha256"]:
        raise ValueError("Different source evidence")
    parent = json.loads((a.parent_dir / "manifest.json").read_text())
    with np.load(
        a.profile_dir / "private_profile_features.npz", allow_pickle=False
    ) as z:
        arrays = {k: z[k] for k in z.files}
    if arrays["author_ids"].tolist() != [b["author_id"] for b in source]:
        raise ValueError("Different profile author order")
    fields = {
        "protocol": "VM05_READY_PROFILE_SOURCE_HEADS_V1",
        "cohort": a.cohort,
        "cases": condition_grid(a.cohort),
        "profile_identity": pm["identity"],
        "profile_npz_sha256": ps["outputs_sha256"]["private_profile_features.npz"],
        "material_sha256": pm["material_sha256"],
        "parent_identity": parent["identity"],
        "scoring_code": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "scripts/run_grain_vector_profile_heads.py",
                "grain_research/vector_mechanisms/external_heads.py",
            )
        },
        "bridge_labels_used": False,
        "thread_control": "retain",
        "reference_membership": "legacy_reference_reuse_declared_not_new_lockbox",
    }
    fields["identity"] = material_hash(fields)
    path = out / "batch_manifest.json"
    if path.exists():
        if not a.resume_completed or json.loads(path.read_text()) != fields:
            raise ValueError("Explicit identical completed-only resume required")
    elif any(
        q.name != ".execution.lock" and not q.name.startswith("._")
        for q in out.iterdir()
    ):
        raise ValueError("Use a new profile-head output directory")
    else:
        write_json(path, fields)
        archive_source(ROOT, out / "source_snapshot.zip")
    start, completed, objects, label_cache = time.monotonic(), [], {}, {}
    with threadpool_limits(limits=4):
        for case in fields["cases"]:
            folder, task = out / case["id"], case["task"]
            group = "big5_only" if task == "big5" else "mbti_only"
            index = np.asarray(
                [
                    i
                    for i, b in enumerate(source)
                    if b["source_group"] == group and arrays[a.cohort][i]
                ]
            )
            bundles = [source[i] for i in index]
            ids = np.asarray([b["author_id"] for b in bundles])
            x = arrays[case["arm"] + "_" + case["channel"]][index]
            folds = np.asarray([b["official_folds"][task] for b in bundles])
            if not np.isfinite(x).all() or any(
                arrays[case["arm"] + "_status"][index] != "GEOMETRY_PROFILE_READY"
            ):
                raise ValueError("Cannot use refused or nonfinite profile rows")
            if folder.exists():
                result = load_oof(folder)
            else:
                if time.monotonic() - start > a.max_seconds - 30:
                    write_json(
                        out / "batch_status.json",
                        {
                            "status": "PAUSED_BUDGET",
                            "completed": completed,
                            "current": case["id"],
                            "planned": len(fields["cases"]),
                        },
                    )
                    return
                folder.mkdir()
                manifest = {
                    "protocol": "VM05_READY_PROFILE_CONDITION_V1",
                    "task": task,
                    "feature": case["channel"],
                    "arm": case["arm"],
                    "pca": case["pca"],
                    "authors": len(ids),
                    "source_group": group,
                    "cohort": a.cohort,
                    "source_material_sha256": material_hash(bundles),
                    "batch_identity": fields["identity"],
                    "feature_input_sha256": hashlib.sha256(
                        np.ascontiguousarray(x).tobytes()
                    ).hexdigest(),
                    "permutation_seed": None,
                    "bridge_labels_used": False,
                }
                manifest["identity"] = material_hash(manifest)
                write_json(folder / "manifest.json", manifest)
                if task not in label_cache:
                    label_cache[task] = load_source_labels(
                        a.prepared_root, bundles, task, parent["source_tables"]
                    )
                y, label_audit = label_cache[task]
                write_json(folder / "source_label_audit.json", label_audit)

                def checkpoint(
                    fold, model, selection, folder=folder, case_id=case["id"]
                ):
                    joblib.dump(
                        model, folder / f"private_fold_{fold}.joblib", compress=3
                    )
                    write_json(folder / "selection.json", selection)
                    write_json(
                        out / "batch_status.json",
                        {
                            "status": "RUNNING",
                            "current": case_id,
                            "completed": completed,
                            "completed_folds": fold + 1,
                            "planned": len(fields["cases"]),
                            "bridge_labels_used": False,
                        },
                    )
                    if time.monotonic() - start > a.max_seconds:
                        raise RuntimeError(
                            "Profile fold-boundary time limit; preserve partial condition"
                        )

                pred, _selection, rows, model, search, parameter = nested_fit(
                    x, y, ids, folds, task, case["pca"], checkpoint
                )
                full_path = folder / "private_full_source_head.joblib"
                joblib.dump(model, full_path, compress=3)
                names = TRAITS if task == "big5" else [task]
                yy, pp = (y, pred) if task == "big5" else (y[:, None], pred[:, None])
                pd.DataFrame(
                    [
                        {
                            "author_id": u,
                            "fold": int(folds[i]),
                            "trait": t,
                            "y_true": yy[i, j],
                            "prediction": pp[i, j],
                        }
                        for i, u in enumerate(ids)
                        for j, t in enumerate(names)
                    ]
                ).to_csv(folder / "private_oof_predictions.csv", index=False)
                pd.DataFrame(rows).to_csv(folder / "metrics_by_fold.csv", index=False)
                pd.DataFrame(metrics(y, pred, task)).to_csv(
                    folder / "metrics_by_trait.csv", index=False
                )
                summary = {
                    "status": "COMPLETE",
                    "parent_identity": manifest["identity"],
                    "authors": len(ids),
                    "task": task,
                    "feature": case["channel"],
                    "arm": case["arm"],
                    "score": task_score(y, pred, task),
                    "bootstrap": bootstrap_score(y, pred, task),
                    "full_parameter": parameter,
                    "full_search": search,
                    "full_head_sha256": hashlib.sha256(
                        full_path.read_bytes()
                    ).hexdigest(),
                    "bridge_labels_used": False,
                }
                write_json(folder / "summary.json", summary)
                result = load_oof(folder)
            m = result["manifest"]
            if (
                m["batch_identity"] != fields["identity"]
                or m["task"] != task
                or m["feature"] != case["channel"]
                or m["arm"] != case["arm"]
                or m["pca"] != case["pca"]
                or m["source_material_sha256"] != material_hash(bundles)
                or m["feature_input_sha256"]
                != hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
            ):
                raise ValueError("Changed completed profile condition")
            objects[(task, case["arm"], case["channel"], case["pca"])] = result
            row = {
                **case,
                "authors": len(ids),
                "score": result["summary"]["score"],
                "mean_fold_score": result["mean_fold_score"],
            }
            completed.append(row)
            pd.DataFrame(completed).to_csv(out / "comparison.csv", index=False)
            print(json.dumps(row | {"completed": len(completed)}), flush=True)
    contrasts = []
    for (task, arm, channel, pca), result in objects.items():
        if channel != "intermediate":
            baseline = objects[(task, arm, "intermediate", pca)]
            contrasts.append(
                {
                    "task": task,
                    "arm": arm,
                    "candidate": channel,
                    "baseline": "intermediate",
                    "pca": pca,
                    **paired_delta(result, baseline),
                }
            )
    pd.DataFrame(contrasts).to_csv(out / "paired_comparison.csv", index=False)
    summary = {
        "status": "COMPLETE",
        "completed_conditions": len(completed),
        "planned": len(fields["cases"]),
        "paired_contrasts": len(contrasts),
        "seconds": time.monotonic() - start,
        "bridge_labels_used": False,
        "scope": "Exploratory frozen-component source readouts, not full GRAIN acceptance; observer/sparse ready refits pending",
    }
    write_json(out / "summary.json", summary)
    write_json(out / "batch_status.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
