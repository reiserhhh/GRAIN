#!/usr/bin/env python3
"""Fixed offline NLI comparator on the exact frozen F08 public synthetic bank."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_grain_entailment_reference import MODEL, REVISION
from scripts.run_grain_evidence_calibration import infer_pairs

QUERIES = {
    "positive": "The writer supported the plan.",
    "negative": "The writer rejected the plan.",
    "qualification": "The writer describes a decision that they actually made about the plan.",
}


def main():
    """Freeze hypotheses first, preserve every class probability, and never retune."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--reader", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    bank_path = args.reader / "grammar_bank.csv"
    bank = pd.read_csv(bank_path)
    if len(bank) != 96 or bank.text.nunique() != 96:
        raise ValueError("expected frozen96 materials")
    plan = {
        "model": MODEL,
        "revision": REVISION,
        "local_files_only": True,
        "queries": QUERIES,
        "unique_inputs": 288,
        "bank_sha256": hashlib.sha256(bank_path.read_bytes()).hexdigest(),
        "pole_score": "positive entailment/(positive+negative entailment);not calibrated posterior",
        "qualification_score": "raw entailment of fixed own-actual-decision hypothesis",
        "threshold": 0.5,
        "no_prompt_rescue": True,
        "private_data": False,
    }
    (args.output / "protocol_before_inference.json").write_text(
        json.dumps(plan, indent=2) + "\n"
    )
    pairs = pd.DataFrame(
        [
            {
                "audit_id": f"{row.material_id}:{name}",
                "sentence": row.text,
                "item_text": query,
            }
            for row in bank.itertuples()
            for name, query in QUERIES.items()
        ]
    )
    pairs.to_csv(args.output / "nli_inputs.csv", index=False)
    start = time.monotonic()
    try:
        readings = infer_pairs(pairs)
    except (OSError, ImportError) as error:
        (args.output / "summary.json").write_text(
            json.dumps(
                {
                    "status": "BLOCKED_LOCAL_MODEL_OR_DEPENDENCY",
                    "reason": str(error),
                    "elapsed": time.monotonic() - start,
                },
                indent=2,
            )
            + "\n"
        )
        raise
    readings.to_csv(args.output / "nli_probabilities.csv", index=False)
    indexed = readings.set_index("audit_id")
    scored = []
    for row in bank.to_dict("records"):
        name = row["material_id"]
        positive = float(indexed.loc[f"{name}:positive", "p_entailment"])
        negative = float(indexed.loc[f"{name}:negative", "p_entailment"])
        if positive + negative <= 0:
            raise ValueError("undefined pole ratio;do not impute")
        pole = positive / (positive + negative)
        scored.append(
            {
                **row,
                "pole_raw": pole,
                "pole_score": pole,
                "qualification_score": float(
                    indexed.loc[f"{name}:qualification", "p_entailment"]
                ),
                "pole_entailment_sum": positive + negative,
            }
        )
    frame = pd.DataFrame(scored)
    frame.to_csv(args.output / "reader_scores.csv", index=False)
    bank.to_csv(args.output / "grammar_bank.csv", index=False)
    metrics = []
    for role in ("source", "validation", "application"):
        for status in ("all", "actual", "quoted", "hypothetical", "negated"):
            subset = frame.loc[frame.role.eq(role)]
            if status != "all":
                subset = subset.loc[subset.subject_status.eq(status)]
            for target in ("pole", "qualification"):
                p = subset[f"{target}_score"].to_numpy()
                y = subset[target].to_numpy()
                metrics.append(
                    {
                        "role": role,
                        "subject_status": status,
                        "target": target,
                        "n": len(y),
                        "brier": float(np.mean((p - y) ** 2)),
                        "accuracy": float(np.mean((p >= 0.5) == y)),
                        "mean_score": float(p.mean()),
                    }
                )
    pd.DataFrame(metrics).to_csv(args.output / "metrics.csv", index=False)
    result = {
        "status": "FIXED_OFFLINE_NLI_READINGS_COMPLETE",
        "actual_pairs": len(readings),
        "elapsed_seconds": time.monotonic() - start,
        "application": [
            r
            for r in metrics
            if r["role"] == "application" and r["subject_status"] == "all"
        ],
        "scope": "polarity on unqualified material is semantic reference,not personal behavior;qualification must gate downstream",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
