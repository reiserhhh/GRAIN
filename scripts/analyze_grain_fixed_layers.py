#!/usr/bin/env python3
"""Compare the three declared layer sensitivities to primary observers, without refitting."""

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.external_controls import exclude_reference
from grain_research.vector_mechanisms.external_heads import metrics
from grain_research.vector_mechanisms.inventory import archive_source
from grain_research.vector_mechanisms.ready_comparison import digest
from scripts.run_grain_neural_sensitivity_batch import (
    cases,
    validate_primary,
    validate_replay,
    validate_result,
)
from scripts.run_grain_vector_mechanism_controls import write_json
from scripts.summarize_grain_neural_source_primary import score
from scripts.summarize_grain_vector_external_links import load_oof

OBSERVERS = (
    "bge_large",
    "e5_large",
    "qwen_block_10",
    "qwen_block_04",
    "qwen_block_16",
    "qwen_final_norm",
)
PAIRS = tuple((i, j) for i in range(3, 6) for j in (2, 0, 1))
TASKS = ("big5", "EI_cont", "SN_cont", "TF_cont", "JP_cont")
ARMS = ("raw", "cue_mask", "placebo_mask")


def paired_family(y, predictions, folds, task, draws=1000):
    """Use one set of author-within-fold draws for all six observers and nine contrasts."""
    y, predictions, folds = np.asarray(y), np.asarray(predictions), np.asarray(folds)
    if (
        predictions.shape != (6, *y.shape)
        or folds.shape != (len(y),)
        or set(folds) != set(range(5))
        or draws < 20
        or task not in TASKS
        or not np.isfinite(y).all()
        or not np.isfinite(predictions).all()
        or (task == "big5" and (y.ndim != 2 or y.shape[1] != 5))
        or (task != "big5" and (y.ndim != 1 or not np.isin(y, [0, 1]).all()))
    ):
        raise ValueError("Require six finite paired observers and five official folds")
    parts = [np.flatnonzero(folds == f) for f in range(5)]

    def estimates(indices):
        joined = np.concatenate(indices)
        pooled = [score(y[joined], p[joined], task) for p in predictions]
        within = [[score(y[ix], p[ix], task) for p in predictions] for ix in indices]
        return np.asarray([pooled, np.mean(within, axis=0)])

    observed = estimates(parts)
    if not np.isfinite(observed).all():
        raise ValueError("Undefined source score")
    rng = np.random.default_rng(731)
    samples = np.asarray(
        [
            estimates([rng.choice(ix, len(ix), replace=True) for ix in parts])
            for _ in range(draws)
        ]
    )
    finite = samples[np.isfinite(samples).all(axis=(1, 2))]
    if len(finite) < 0.95 * draws:
        raise ValueError("Insufficient bootstrap target support")
    rows = []
    for a, b in PAIRS:
        for k, estimand in enumerate(("pooled", "mean_official_fold")):
            low, high = np.quantile(finite[:, k, a] - finite[:, k, b], [0.025, 0.975])
            rows.append(
                {
                    "candidate": OBSERVERS[a],
                    "baseline": OBSERVERS[b],
                    "estimand": estimand,
                    "candidate_score": observed[k, a],
                    "baseline_score": observed[k, b],
                    "delta": observed[k, a] - observed[k, b],
                    "lower": low,
                    "upper": high,
                    "draws": draws,
                    "undefined_draws": draws - len(finite),
                    "contrast_type": "within_model_layer"
                    if b == 2
                    else "observer_pipeline",
                }
            )
    return rows


def complete_arms(folder):
    """A partial report may include only whole15-head arms, never a favored task/layer."""
    result = []
    for arm in ARMS:
        selected = [c for c in cases()[9:] if c["arm"] == arm]
        if all(
            (folder / (c["id"] + "_replay") / "summary.json").exists() for c in selected
        ):
            result.append(arm)
    return result


def require_paired(packets):
    """Reject silently restricted, reordered or differently folded observer comparisons."""
    if len(packets) != 6 or any(
        not np.array_equal(packets[0][key], p[key])
        for p in packets[1:]
        for key in ("ids", "y", "folds")
    ):
        raise ValueError(
            "Layer comparison requires identical authors, targets and official folds"
        )


def main():
    """Bind all source evidence, then report layers without changing the primary scorer."""
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "batch-dir",
        "primary-dir",
        "audit-dir",
        "resource-dir",
        "views",
        "preflight-dir",
        "output-dir",
    ):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--allow-partial", action="store_true")
    a = p.parse_args()
    primary, audit = validate_primary(a.primary_dir, a.audit_dir, a.resource_dir)
    batch = json.loads((a.batch_dir / "batch_manifest.json").read_text())
    if (
        batch["protocol"] != "VM05_FROZEN_NEURAL_SENSITIVITIES_V1"
        or material_hash({k: v for k, v in batch.items() if k != "identity"})
        != batch["identity"]
        or batch["cases"] != cases()
        or batch["primary_identity"] != primary["identity"]
        or batch["primary_audit_identity"] != audit["identity"]
        or digest(a.views) != primary["views_sha256"]
        or digest(a.preflight_dir / "manifest.json")
        != primary["preflight_manifest_sha256"]
        or any(digest(ROOT / n) != sha for n, sha in batch["code_sha256"].items())
    ):
        raise ValueError("Changed layer sensitivity parent or input")
    arms = complete_arms(a.batch_dir)
    if not arms or (len(arms) != 3 and not a.allow_partial):
        raise ValueError(
            "Complete three-arm family required; explicit partial keeps only whole arms"
        )
    source = json.loads(a.views.read_text(encoding="utf-8"))
    common = json.loads((a.preflight_dir / "common_material_manifest.json").read_text())
    if material_hash(source) != common["material_sha256"]:
        raise ValueError("Common material changed")
    source = exclude_reference(
        source,
        pd.read_csv(a.preflight_dir / "budget_by_author.csv", dtype={"author_id": str}),
    )
    groups, inputs = [], {}
    for arm in arms:
        for task in TASKS:
            packets = []
            for index, feature in enumerate(OBSERVERS):
                base_name = f"{task}_{arm}_{feature}"
                case = {
                    "task": task,
                    "arm": arm,
                    "feature": feature,
                    "pca": True,
                    "id": base_name + "_pca16",
                }
                root = a.primary_dir if index < 3 else a.batch_dir
                folder = root / (base_name if index < 3 else case["id"])
                result = load_oof(folder)
                validate_result(result, case, source, common["material_sha256"])
                if (
                    digest(folder / "private_full_source_head.joblib")
                    != result["summary"]["full_head_sha256"]
                ):
                    raise ValueError("Source head changed")
                if index >= 3:
                    replay = root / (case["id"] + "_replay")
                    validate_replay(replay, case)
                    paths = [
                        folder / n
                        for n in (
                            "manifest.json",
                            "summary.json",
                            "private_oof_predictions.csv",
                            "private_full_source_head.joblib",
                        )
                    ]
                    paths += [folder / f"private_fold_{f}.joblib" for f in range(5)]
                    paths += [replay / "summary.json", replay / "replay_by_fold.csv"]
                    inputs[case["id"]] = {
                        str(path.relative_to(root)): digest(path) for path in paths
                    }
                packets.append(result)
            require_paired(packets)
            groups.append((task, arm, packets))
    rows, scores, traits = [], [], []
    for task, arm, packets in groups:
        rows.extend(
            dict(task=task, arm=arm, authors=len(packets[0]["ids"]), **r)
            for r in paired_family(
                packets[0]["y"], [r["p"] for r in packets], packets[0]["folds"], task
            )
        )
        for feature, r in zip(OBSERVERS, packets, strict=True):
            scores.append(
                {
                    "task": task,
                    "arm": arm,
                    "observer": feature,
                    "authors": len(r["ids"]),
                    "pooled_score": r["summary"]["score"],
                    "mean_fold_score": r["mean_fold_score"],
                }
            )
            traits.extend(
                dict(task=task, arm=arm, observer=feature, **m)
                for m in metrics(r["y"], r["p"], task)
            )
        print(
            json.dumps({"compared_task": task, "arm": arm, "observers": 6}), flush=True
        )
    a.output_dir.mkdir(parents=True, exist_ok=False)
    code = (
        "scripts/analyze_grain_fixed_layers.py",
        "scripts/summarize_grain_neural_source_primary.py",
        "scripts/analyze_grain_ei_source_gap.py",
    )
    m = {
        "protocol": "VM05_FIXED_LAYER_COMPARISON_V1",
        "sensitivity_identity": batch["identity"],
        "primary_audit_identity": audit["identity"],
        "inputs": inputs,
        "accepted_arms": arms,
        "planned_arms": list(ARMS),
        "observers": list(OBSERVERS),
        "contrasts": [[OBSERVERS[a], OBSERVERS[b]] for a, b in PAIRS],
        "code_sha256": {n: digest(ROOT / n) for n in code},
        "bootstrap": "1000 shared author-within-fold draws seed731; pointwise conditional on fitted OOF",
        "analysis_implementation_after_some_point_estimates": True,
    }
    m["identity"] = material_hash(m)
    write_json(a.output_dir / "manifest.json", m)
    archive_source(ROOT, a.output_dir / "source_snapshot.zip")
    with zipfile.ZipFile(
        a.output_dir / "source_snapshot.zip", "a", zipfile.ZIP_DEFLATED
    ) as z:
        for n in code:
            if n not in z.namelist():
                z.write(ROOT / n, n)
    tables = {
        "observer_scores.csv": scores,
        "metrics_by_trait.csv": traits,
        "paired_layers.csv": rows,
    }
    for name, records in tables.items():
        pd.DataFrame(records).to_csv(a.output_dir / name, index=False)
    summary = {
        "status": "COMPLETE" if len(arms) == 3 else "PARTIAL",
        "parent_identity": m["identity"],
        "accepted_arms": arms,
        "pending_arms": [arm for arm in ARMS if arm not in arms],
        "sensitivity_heads": len(inputs),
        "paired_rows": len(rows),
        "source_label_tables_read": False,
        "existing_source_oof_outcomes_reused": True,
        "bridge_labels_read": False,
        "model_refits": 0,
        "program_complete": False,
        "outputs_sha256": {n: digest(a.output_dir / n) for n in tables},
        "scope": "Declared layer sensitivity, not best-layer selection, simultaneous inference, model-size effect or causal personality mechanism",
    }
    write_json(a.output_dir / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
