#!/usr/bin/env python3
"""Recompute VM01 evidence and lexical controls from saved synthetic records."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.interventions import fit_margin_threshold


def main():
    """Save reproducible aggregate comparisons, never exporting any raw text."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        json.loads((args.parent_dir / "contrast_manifest.json").read_text())
    )
    preds = pd.read_csv(args.parent_dir / "fact_predictions.csv")
    frame = frame.merge(
        preds[["case_id", "margin"]], on="case_id", validate="one_to_one"
    )
    rows = []
    for family, sub in frame.groupby("family"):
        tr, te = sub[sub.split == "discovery"], sub[sub.split != "discovery"]
        if set(tr.group_id) & set(te.group_id):
            raise ValueError("Scenario split leak")
        threshold = fit_margin_threshold(tr.margin, tr.expected_yes)
        row = {
            "family": family,
            "heldout_scenarios": te.group_id.nunique(),
            "heldout_prompts": len(te),
            "llm_auc": roc_auc_score(te.expected_yes, te.margin),
            "default_balanced_accuracy": balanced_accuracy_score(
                te.expected_yes, te.margin >= 0
            ),
            "discovery_threshold": threshold,
            "calibrated_balanced_accuracy": balanced_accuracy_score(
                te.expected_yes, te.margin >= threshold
            ),
        }
        for maximum in [1, 2]:
            vectorizer = TfidfVectorizer(ngram_range=(1, maximum), lowercase=True)
            x = vectorizer.fit_transform(tr.passage)
            model = LogisticRegression(
                C=0.01, solver="liblinear", random_state=731
            ).fit(x, tr.expected_yes)
            score = model.decision_function(vectorizer.transform(te.passage))
            row[f"word_1_{maximum}_gram_auc"] = roc_auc_score(te.expected_yes, score)
        rows.append(row)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(args.output_dir / "semantic_comparison.csv", index=False)
    layers = pd.read_csv(args.parent_dir / "layer_observability.csv")
    layers.to_csv(args.output_dir / "layer_observability.csv", index=False)
    input_rows = layers[layers.site.str.startswith("input_embedding")]
    validation = {
        "scope": "synthetic semantic controls; NOT human personality",
        "rows": len(frame),
        "independent_scenarios": frame.group_id.nunique(),
        "discovery_scenarios": frame[frame.split == "discovery"].group_id.nunique(),
        "heldout_scenarios": frame[frame.split != "discovery"].group_id.nunique(),
        "initial_embedding_auc_all_chance": bool(
            np.allclose(input_rows.heldout_auc, 0.5)
        ),
        "model_manifest": json.loads(
            (args.parent_dir / "model_manifest.json").read_text()
        ),
        "parent_identity": json.loads((args.parent_dir / "manifest.json").read_text())[
            "identity"
        ],
        "external_validation": "NOT_RUN",
        "templates_held_out": False,
        "caveats": [
            "Only four held-out scenes; templates shared across splits",
            "Positive result validates instruments, not human personality",
            "Calibration is discovery-only secondary analysis",
            "High decoding AUC does not prove functional specificity",
        ],
    }
    (args.output_dir / "validation.json").write_text(
        json.dumps(validation, indent=2) + "\n"
    )
    print(comparison.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
