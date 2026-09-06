#!/usr/bin/env python3
"""Post-hoc proper-score diagnostics for all36 frozen primary MBTI source heads."""

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
from grain_research.vector_mechanisms.inventory import archive_source
from grain_research.vector_mechanisms.ready_comparison import digest
from scripts.run_grain_neural_sensitivity_batch import validate_primary
from scripts.run_grain_vector_mechanism_controls import write_json
from scripts.summarize_grain_neural_source_primary import OBSERVERS, score
from scripts.summarize_grain_vector_external_links import load_oof

EPSILON = 1e-15


def training_prior(y, folds):
    """Predict each held-out fold using only the other four folds' class proportion."""
    y, folds = np.asarray(y), np.asarray(folds)
    if (
        y.ndim != 1
        or folds.shape != y.shape
        or set(folds) != set(range(5))
        or not np.isin(y, [0, 1]).all()
    ):
        raise ValueError("Five paired official folds and binary outcomes required")
    prior = np.empty(len(y), dtype=float)
    for fold in range(5):
        prevalence = y[folds != fold].mean()
        if not 0 < prevalence < 1:
            raise ValueError("Each training set must contain both classes")
        prior[folds == fold] = prevalence
    return prior


def proper_scores(y, predictions, folds, draws=1000):
    """Compare three fixed heads with fold-local priors; no calibration or head refit."""
    y, predictions, folds = map(np.asarray, (y, predictions, folds))
    prior = training_prior(y, folds)
    if (
        predictions.shape != (3, len(y))
        or not np.isfinite(predictions).all()
        or np.any((predictions < 0) | (predictions > 1))
        or draws < 20
    ):
        raise ValueError(
            "Three paired probability vectors and at least20 draws required"
        )
    all_p = np.vstack([prior, predictions])
    clipped = np.clip(all_p, EPSILON, 1 - EPSILON)
    losses = np.stack(
        [
            -(y * np.log(clipped) + (1 - y) * np.log1p(-clipped)),
            (all_p - y) ** 2,
        ]
    )
    parts = [np.flatnonzero(folds == fold) for fold in range(5)]

    def estimate(indices):
        return np.stack(
            [
                losses[:, :, np.concatenate(indices)].mean(axis=2),
                np.mean([losses[:, :, ix].mean(axis=2) for ix in indices], axis=0),
            ]
        )

    observed = estimate(parts)
    rng = np.random.default_rng(731)
    samples = np.asarray(
        [
            estimate([rng.choice(ix, len(ix), replace=True) for ix in parts])
            for _ in range(draws)
        ]
    )
    rows = []
    for model, name in enumerate(OBSERVERS, start=1):
        for estimand_index, estimand in enumerate(("pooled", "mean_official_fold")):
            for metric_index, metric in enumerate(("log_loss_nats", "brier")):
                improvement = (
                    samples[:, estimand_index, metric_index, 0]
                    - samples[:, estimand_index, metric_index, model]
                )
                lower, upper = np.quantile(improvement, [0.025, 0.975])
                base, candidate = observed[estimand_index, metric_index, [0, model]]
                rows.append(
                    {
                        "observer": name,
                        "estimand": estimand,
                        "metric": metric,
                        "prior_loss": base,
                        "candidate_loss": candidate,
                        "improvement": base - candidate,
                        "lower": lower,
                        "upper": upper,
                        "draws": draws,
                        "clipped_predictions": int(
                            np.count_nonzero(all_p[model] != clipped[model])
                        ),
                    }
                )
    return rows


def main():
    """Bind the accepted45-head parent and analyze all36 binary heads without selection."""
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("primary-dir", "audit-dir", "resource-dir", "output-dir"):
        p.add_argument("--" + name, required=True, type=Path)
    a = p.parse_args()
    primary, audit = validate_primary(a.primary_dir, a.audit_dir, a.resource_dir)
    rows, rankings = [], []
    for arm in ("raw", "cue_mask", "placebo_mask"):
        for task in ("EI_cont", "SN_cont", "TF_cont", "JP_cont"):
            packets = [
                load_oof(a.primary_dir / f"{task}_{arm}_{observer}")
                for observer in OBSERVERS
            ]
            if any(
                not np.array_equal(packets[0][k], other[k])
                for other in packets[1:]
                for k in ("ids", "y", "folds")
            ) or any(
                r["manifest"]["thread_control"] != "retain"
                or not r["manifest"]["exclude_reference"]
                for r in packets
            ):
                raise ValueError(
                    "Incompatible cohort or fold-prior training membership"
                )
            y, folds = packets[0]["y"], packets[0]["folds"]
            rows.extend(
                dict(task=task, arm=arm, authors=len(y), **r)
                for r in proper_scores(y, [r["p"] for r in packets], folds)
            )
            for observer, r in zip(OBSERVERS, packets, strict=True):
                rankings.append(
                    {
                        "task": task,
                        "arm": arm,
                        "observer": observer,
                        "authors": len(y),
                        "auc": score(y, r["p"], task),
                        "mean_official_fold_auc": r["mean_fold_score"],
                        "positive_rate_true": y.mean(),
                        "positive_rate_pred": (r["p"] >= 0.5).mean(),
                        "mean_predicted_probability": r["p"].mean(),
                        "sd_predicted_probability": r["p"].std(),
                    }
                )
            print(json.dumps({"task": task, "arm": arm, "heads": 3}), flush=True)
    a.output_dir.mkdir(parents=True, exist_ok=False)
    code = "scripts/analyze_grain_source_proper_scores.py"
    manifest = {
        "protocol": "VM05_PRIMARY_PROPER_SCORE_DIAGNOSTIC_V1",
        "primary_identity": primary["identity"],
        "primary_audit_identity": audit["identity"],
        "primary_audit_manifest_sha256": digest(a.audit_dir / "manifest.json"),
        "primary_audit_summary_sha256": digest(a.audit_dir / "summary.json"),
        "source_code_sha256": digest(ROOT / code),
        "bootstrap": "1000 shared author-within-fold draws seed731; conditional on fixed OOF/prior",
        "prior": "Class proportion on the other four official source folds",
        "epsilon": EPSILON,
        "diagnostic_declared_after_primary_auc_results": True,
        "not_mutual_information": True,
    }
    manifest["identity"] = material_hash(manifest)
    write_json(a.output_dir / "manifest.json", manifest)
    archive_source(ROOT, a.output_dir / "source_snapshot.zip")
    with zipfile.ZipFile(
        a.output_dir / "source_snapshot.zip", "a", zipfile.ZIP_DEFLATED
    ) as z:
        if code not in z.namelist():
            z.write(ROOT / code, code)
    tables = {"proper_scores.csv": rows, "ranking_diagnostics.csv": rankings}
    for name, values in tables.items():
        pd.DataFrame(values).to_csv(a.output_dir / name, index=False)
    summary = {
        "status": "COMPLETE",
        "parent_identity": manifest["identity"],
        "heads": len(rankings),
        "paired_rows": len(rows),
        "source_label_tables_read": False,
        "existing_source_oof_outcomes_reused": True,
        "source_heads_refit": 0,
        "unique_fold_prior_count_estimators": 20,
        "bridge_labels_read": False,
        "outputs_sha256": {name: digest(a.output_dir / name) for name in tables},
        "scope": "Post-hoc probability-loss diagnostic; not MI, native LLM use, recalibration or winner selection",
        "program_complete": False,
    }
    write_json(a.output_dir / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
