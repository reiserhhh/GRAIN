#!/usr/bin/env python3
"""Freeze revised materials, fit source-only readers, and expose real readout errors."""

import argparse
import csv
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.gptpro_foundation_v3.f08 import fit_reader
from grain_research.gptpro_foundation_v3.f08_materials import material_bank, read_scores


def table(path, rows):
    """Serialize inspectable generated data with fixed column order."""
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    """Train only after bank freeze, then score each material without refitting."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    bank = material_bank()
    bank_path = args.output / "grammar_bank.csv"
    table(bank_path, bank)
    bank_hash = hashlib.sha256(bank_path.read_bytes()).hexdigest()
    frozen = {
        "bank_sha256": bank_hash,
        "fit_role": "source",
        "source_rows": 32,
        "role_split": "disjoint framing families; ownership/pole core phrases shared",
        "model": "word1-2/char2-4 TFIDF max5000 each,Ridge alpha1,qualification logistic C1",
        "raw_private_data": False,
        "qualification": "actual/corrected=1;third-party/hypothetical=0",
    }
    (args.output / "bank_frozen_before_fit.json").write_text(
        json.dumps(frozen, indent=2) + "\n"
    )
    source = [r for r in bank if r["role"] == "source"]
    reader = fit_reader(
        [r["text"] for r in source],
        [r["pole"] for r in source],
        [r["qualification"] for r in source],
    )
    with (args.output / "reader.pkl").open("wb") as stream:
        pickle.dump(reader, stream)
    if hashlib.sha256(bank_path.read_bytes()).hexdigest() != bank_hash:
        raise RuntimeError("bank changed during fit")
    scores = read_scores(reader, [r["text"] for r in bank])
    scored = [
        {
            **row,
            "pole_raw": float(scores["pole_raw"][i]),
            "pole_score": float(scores["pole"][i]),
            "qualification_score": float(scores["qualification"][i]),
        }
        for i, row in enumerate(bank)
    ]
    table(args.output / "reader_scores.csv", scored)
    metrics, calibration, confusions = [], [], []
    for role in ("source", "validation", "application"):
        subset = [r for r in scored if r["role"] == role]
        for target, truth_key, score_key in (
            ("pole", "pole", "pole_score"),
            ("qualification", "qualification", "qualification_score"),
        ):
            y = np.array([r[truth_key] for r in subset])
            p = np.array([r[score_key] for r in subset])
            metrics.append(
                {
                    "role": role,
                    "target": target,
                    "n": len(y),
                    "brier": float(np.mean((p - y) ** 2)),
                    "accuracy": float(np.mean((p >= 0.5) == y)),
                }
            )
            matrix = confusion_matrix(y, p >= 0.5, labels=[0, 1])
            for actual in (0, 1):
                for predicted in (0, 1):
                    confusions.append(
                        {
                            "role": role,
                            "target": target,
                            "actual": actual,
                            "predicted": predicted,
                            "count": int(matrix[actual, predicted]),
                        }
                    )
            for bin_index, indices in enumerate(
                np.array_split(np.argsort(p, kind="stable"), 10)
            ):
                calibration.append(
                    {
                        "role": role,
                        "target": target,
                        "bin": bin_index,
                        "n": len(indices),
                        "mean_score": float(p[indices].mean()),
                        "observed_rate": float(y[indices].mean()),
                    }
                )
    table(args.output / "reader_metrics.csv", metrics)
    table(args.output / "confusion.csv", confusions)
    table(args.output / "calibration_10_equal_count.csv", calibration)
    controls = []
    for seed in (20260915, 20260916, 20260917):
        order = np.random.default_rng(seed).permutation(len(bank))
        perm = read_scores(reader, [bank[i]["text"] for i in order])
        maximum = max(
            float(np.max(np.abs(perm[k][np.argsort(order)] - scores[k])))
            for k in scores
        )
        controls.append(
            {
                "seed": seed,
                "control": "ID_and_row_order_permutation",
                "maximum_difference": maximum,
            }
        )
        if maximum > 1e-12:
            raise RuntimeError("ID/order dependence")
    # Word-order reversal is a diagnostic, not a semantics-preserving invariance claim.
    reversed_scores = read_scores(
        reader, [" ".join(r["text"].split()[::-1]) for r in bank]
    )
    controls.append(
        {
            "seed": "",
            "control": "word_reversal_pole_change",
            "maximum_difference": float(
                np.max(np.abs(reversed_scores["pole"] - scores["pole"]))
            ),
        }
    )
    table(args.output / "controls.csv", controls)
    contrasts = []
    for family in ("plain", "contextual", "retrospective"):
        for pole in (0, 1):
            for variant in range(4):
                pair = [
                    r
                    for r in scored
                    if r["family"] == family
                    and r["pole"] == pole
                    and r["variant"] == variant
                ]
                actual = next(r for r in pair if r["subject_status"] == "actual")
                for other in pair:
                    if other["subject_status"] != "actual":
                        contrasts.append(
                            {
                                "actual_id": actual["material_id"],
                                "other_id": other["material_id"],
                                "status": other["subject_status"],
                                "pole_delta": other["pole_score"]
                                - actual["pole_score"],
                                "qualification_delta": other["qualification_score"]
                                - actual["qualification_score"],
                            }
                        )
    table(args.output / "ownership_contrasts.csv", contrasts)
    summary = {
        "status": "REVISED_READER_COMPUTED_DOWNSTREAM_OPEN",
        "materials": 96,
        "fit_rows": 32,
        "scope": "synthetic transparent templates; wrapper shift only,not natural ownership generalization",
        "next": "source-transition oracle/soft/hard filtering and actual B1/B2/M target losses;F07 measured confusion interface",
        "metrics": metrics,
        "bank_sha256": bank_hash,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in args.output.iterdir()
        if p.is_file()
    }
    (args.output / "output_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
