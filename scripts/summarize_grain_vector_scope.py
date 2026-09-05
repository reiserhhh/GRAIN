#!/usr/bin/env python3
"""Calibrate scoped readouts without confusing synthetic, calibration or trait validity."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.analysis import grouped_bootstrap
from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.interventions import fit_margin_threshold
from grain_research.vector_mechanisms.inventory import archive_source, code_fingerprint
from scripts.run_grain_vector_mechanism_controls import write_json


def metrics(sub, predictions, *, score=None):
    """Keep single-class source samples out of discriminative metric claims."""
    y = sub.expected_yes.astype(int).to_numpy()
    predictions = np.asarray(predictions).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    return {
        "rows": len(sub),
        "groups": sub.group_id.nunique(),
        "positive_truth": int(y.sum()),
        "positive_predictions": int(predictions.sum()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions))
        if len(set(y)) == 2
        else None,
        "auc": float(roc_auc_score(y, sub.margin if score is None else score))
        if len(set(y)) == 2
        else None,
        **{
            "accuracy_" + k: v
            for k, v in grouped_bootstrap(predictions == y, sub.group_id).items()
        },
    }


def ambiguity_bounds(sub, threshold):
    """Enumerate unknown source codes without choosing the assignment that helps."""
    unknown = sub.expected_yes.isna().to_numpy()
    if unknown.sum() > 12:
        raise ValueError("Ambiguity enumeration budget exceeded")
    base = sub.expected_yes.fillna(0).astype(int).to_numpy()
    scores = sub.margin.to_numpy()
    predictions = scores >= threshold
    values = []
    for assignment in itertools.product((0, 1), repeat=int(unknown.sum())):
        y = base.copy()
        y[unknown] = assignment
        if len(set(y)) != 2:
            continue
        values.append(
            {
                "accuracy": float(np.mean(y == predictions)),
                "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
                "raw_margin_auc": float(roc_auc_score(y, scores)),
            }
        )
    if not values:
        return {"unknown": int(unknown.sum()), "assignments": 0}
    table = pd.DataFrame(values)
    return {
        "unknown": int(unknown.sum()),
        "assignments": len(values),
        **{k + "_min": float(table[k].min()) for k in table},
        **{k + "_max": float(table[k].max()) for k in table},
    }


def natural_auc_uncertainty(sub):
    """Bootstrap raw-margin discrimination by author; no LOO calibration claim."""
    sub = sub[sub.expected_yes.notna()]
    if sub.expected_yes.nunique() != 2:
        return {"auc": None, "ci_low": None, "ci_high": None, "draws": 0}
    groups = [g for _, g in sub.groupby("group_id")]
    rng = np.random.default_rng(731)
    values = []
    for _ in range(2000):
        sample = pd.concat(
            [groups[i] for i in rng.integers(0, len(groups), len(groups))]
        )
        if sample.expected_yes.nunique() == 2:
            values.append(roc_auc_score(sample.expected_yes, sample.margin))
    return {
        "auc": float(roc_auc_score(sub.expected_yes, sub.margin)),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "draws": len(values),
    }


def main():
    """Freeze scalar thresholds; natural calibration fits are not a held-out result."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-dir", type=Path, required=True)
    p.add_argument("--materials", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))
    manifest, status = (
        read(a.parent_dir / "manifest.json"),
        read(a.parent_dir / "run_status.json"),
    )
    if status["status"] != "COMPLETE":
        raise ValueError("Incomplete scope run")
    ratings = read(a.materials / "source_ratings.json")
    sources = read(a.materials / "calibration_sources.json")
    if manifest["ratings_sha256"] != material_hash(ratings):
        raise ValueError("Source coding changed after model run")
    frame = pd.read_csv(a.parent_dir / "readouts.csv")
    if len(frame) != status["cases"] or frame.case_id.tolist() != [
        r["case_id"] for r in manifest["jobs"]
    ]:
        raise ValueError("Readout material order/count mismatch")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    if any(a.output_dir.iterdir()):
        raise ValueError("Use a new analysis directory")
    archive_source(ROOT, a.output_dir / "source_snapshot.zip")
    summary, thresholds, variants, sensitivity = [], [], [], []
    for (task, wording), part in frame.groupby(["task", "wording"]):
        train = part[part.split == "scope_development"]
        threshold = fit_margin_threshold(train.margin, train.expected_yes)
        cal = part[(part.kind == "natural_scope") & part.expected_yes.notna()]
        natural_threshold = (
            fit_margin_threshold(cal.margin, cal.expected_yes)
            if cal.expected_yes.nunique() == 2
            else None
        )
        thresholds.append(
            {
                "task": task,
                "wording": wording,
                "synthetic_development_threshold": threshold,
                "natural_calibration_threshold": natural_threshold,
                "natural_calibration_both_classes": natural_threshold is not None,
            }
        )
        natural = part[part.kind == "natural_scope"]
        if natural.expected_yes.isna().any():
            sensitivity.append(
                {
                    "task": task,
                    "wording": wording,
                    "mode": "fixed_synthetic_threshold_all_ambiguity_assignments",
                    **ambiguity_bounds(natural, threshold),
                    **{
                        "resolved_" + k: v
                        for k, v in natural_auc_uncertainty(natural).items()
                    },
                }
            )
        for split, sub in part.groupby("split"):
            usable = sub[sub.expected_yes.notna()]
            if not len(usable):
                continue
            for mode, pred in (
                ("default", usable.predicted_yes),
                ("synthetic_threshold", usable.margin >= threshold),
            ):
                summary.append(
                    {
                        "task": task,
                        "wording": wording,
                        "split": split,
                        "mode": mode,
                        "unresolved": int(sub.expected_yes.isna().sum()),
                        **metrics(usable, pred),
                    }
                )
            if split == "calibration" and natural_threshold is not None:
                predicted = []
                for i, row in usable.iterrows():
                    others = usable[usable.group_id != row.group_id]
                    cutoff = fit_margin_threshold(others.margin, others.expected_yes)
                    predicted.append(int(row.margin >= cutoff))
                summary.append(
                    {
                        "task": task,
                        "wording": wording,
                        "split": split,
                        "mode": "natural_leave_one_author_out",
                        "unresolved": int(sub.expected_yes.isna().sum()),
                        **metrics(usable, predicted),
                    }
                )
                summary.append(
                    {
                        "task": task,
                        "wording": wording,
                        "split": split,
                        "mode": "natural_resubstitution_not_test",
                        "unresolved": int(sub.expected_yes.isna().sum()),
                        **metrics(usable, usable.margin >= natural_threshold),
                    }
                )
        for (split, variant), sub in part[part.kind == "synthetic_scope"].groupby(
            ["split", "variant"]
        ):
            variants.append(
                {
                    "task": task,
                    "wording": wording,
                    "split": split,
                    "variant": variant,
                    **metrics(sub, sub.predicted_yes),
                }
            )
    pd.DataFrame(summary).to_csv(a.output_dir / "scope_metrics.csv", index=False)
    pd.DataFrame(variants).to_csv(a.output_dir / "scope_variants.csv", index=False)
    pd.DataFrame(sensitivity).to_csv(
        a.output_dir / "source_coding_sensitivity.csv", index=False
    )
    write_json(
        a.output_dir / "readout_thresholds.json",
        {
            "parent_identity": manifest["identity"],
            "ratings_sha256": manifest["ratings_sha256"],
            "primary_wording": "question",
            "secondary_wording": "definition",
            "thresholds": thresholds,
            "external_labels_used": False,
            "limitations": "Natural enacted coding is single-analyst with ambiguous cases withheld. Reported guidance has no natural positives and cannot be calibrated on this sample.",
        },
    )
    marker_counts = {
        marker: sum(marker in r["current"] for r in sources)
        for marker in ("TYPE_MENTION", "COGFUNC_MENTION", "Ã", "â€")
    }
    output = {
        "status": "COMPLETE",
        "parent_identity": manifest["identity"],
        "code_sha256": code_fingerprint(ROOT),
        "synthetic_scenes": 12,
        "development_scenes": 8,
        "heldout_scenes": 4,
        "calibration_authors": 48,
        "raw_source_marker_comments": marker_counts,
        "runtime": status,
        "external_labels_used": False,
        "auc_definition": "Raw margin AUC in every scope_metrics row; not cross-fitted calibrated AUC in LOO rows.",
        "access": "48 calibration current comments have now been coded and observed. 48 mechanism-holdout authors remain unopened.",
    }
    write_json(a.output_dir / "summary.json", output)
    print(pd.DataFrame(summary).to_string(index=False))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
