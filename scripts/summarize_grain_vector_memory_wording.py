#!/usr/bin/env python3
"""Paired interrogative/statement audit, conditional on a fixed event family."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.analysis import grouped_bootstrap
from grain_research.vector_mechanisms.inventory import archive_source, code_fingerprint
from scripts.run_grain_vector_mechanism_controls import write_json


def main():
    """Evaluate a diagnostic wording change without fitting a classifier."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original", type=Path, required=True)
    p.add_argument("--statement", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    frames, features, manifests = [], [], []
    for path in (a.original, a.statement):
        status = json.loads((path / "run_status.json").read_text(encoding="utf-8"))
        if status["status"] != "COMPLETE":
            raise ValueError("Incomplete parent")
        manifests.append(
            json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        )
        f = pd.read_csv(path / "readouts.csv")
        f["feature_row"] = np.arange(len(f))
        frames.append(f[f.kind == "synthetic_on_natural"])
        features.append(np.load(path / "representations.npz"))
    for key in ("parent_identity", "review_sha256", "generated_sha256", "padding"):
        if manifests[0][key] != manifests[1][key]:
            raise ValueError("Different source/protocol parents")
    for key in ("revision", "dtype", "device", "template_sha256", "pad_to_length"):
        if manifests[0]["observer"][key] != manifests[1]["observer"][key]:
            raise ValueError("Different numerical observers")
    if (
        manifests[0].get("question_style", "interrogative") != "interrogative"
        or manifests[1]["question_style"] != "statement"
    ):
        raise ValueError("Unexpected wording conditions")
    keys = ["case_id", "author_id", "arm", "placement", "negative_question"]
    x = frames[0].set_index(keys).sort_index()
    y = frames[1].set_index(keys).sort_index()
    if not x.index.equals(y.index) or not np.array_equal(
        x.expected_yes, y.expected_yes
    ):
        raise ValueError("Unpaired materials/truth")
    equalities = {}
    for key in (
        "block_10_current_mean",
        "block_10_evidence_end",
        "block_16_evidence_end",
    ):
        err = float(
            np.max(
                np.abs(
                    features[0][key][x.feature_row] - features[1][key][y.feature_row]
                )
            )
        )
        equalities[key] = err
        if err != 0:
            raise ValueError("Different pre-question evidence encoding")
    paired = x.reset_index()[keys]
    for name, frame in (("interrogative", x), ("statement", y)):
        for col in ("margin", "correct", "predicted_yes", "expected_yes"):
            paired[name + "_" + col] = frame[col].to_numpy()
    groups = [("all", paired)]
    groups += [
        (f"{key}={value}", sub)
        for key in ("arm", "placement", "negative_question")
        for value, sub in paired.groupby(key)
    ]
    rows = []
    for label, sub in groups:
        row = {
            "group": label,
            "prompts_per_wording": len(sub),
            "authors": sub.author_id.nunique(),
        }
        for name in ("interrogative", "statement"):
            truth = sub[name + "_expected_yes"].astype(int)
            pred = sub[name + "_predicted_yes"]
            margin = sub[name + "_margin"]
            row.update(
                {
                    name + "_" + k: v
                    for k, v in grouped_bootstrap(
                        sub[name + "_correct"], sub.author_id
                    ).items()
                }
            )
            row[name + "_auc"] = (
                float(roc_auc_score(truth, margin)) if truth.nunique() == 2 else None
            )
            row[name + "_balanced_accuracy"] = (
                float(balanced_accuracy_score(truth, pred))
                if truth.nunique() == 2
                else None
            )
        row.update(
            {
                "delta_accuracy_" + k: v
                for k, v in grouped_bootstrap(
                    sub.statement_correct - sub.interrogative_correct, sub.author_id
                ).items()
            }
        )
        rows.append(row)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    if any(a.output_dir.iterdir()):
        raise ValueError("Use a new directory")
    archive_source(ROOT, a.output_dir / "source_snapshot.zip")
    pd.DataFrame(rows).to_csv(a.output_dir / "wording_comparison.csv", index=False)
    polarity = []
    for name, frame in (("interrogative", x), ("statement", y)):
        f = frame.reset_index()
        for placement, sub in f.groupby("placement"):
            pairs = sub.pivot(
                index=["author_id", "case_id", "arm"],
                columns="negative_question",
                values="predicted_yes",
            )
            stats = grouped_bootstrap(
                (pairs[False] != pairs[True]).astype(float),
                pairs.index.get_level_values("author_id"),
            )
            polarity.append({"wording": name, "placement": placement, **stats})
    pd.DataFrame(polarity).to_csv(a.output_dir / "polarity_comparison.csv", index=False)
    output = {
        "status": "COMPLETE",
        "original_identity": manifests[0]["identity"],
        "analysis_code_sha256": code_fingerprint(ROOT),
        "statement_identity": manifests[1]["identity"],
        "prefix_max_errors": equalities,
        "overall": rows[0],
        "scope": "Post-observation diagnostic; one fixed event family, six natural background authors. Not confirmation or personality accuracy.",
    }
    write_json(a.output_dir / "summary.json", output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
