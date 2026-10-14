#!/usr/bin/env python3
"""Post-packet mechanism contrast: condition clause vs whole-utterance coordinates.

This is one declared follow-up after condition-accumulation r1, not a preregistered
arm of that first packet. Original events and NLI readings stay unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_grain_predecessor_reference import digest, write
from scripts.run_grain_condition_accumulation import (
    aggregate_metrics,
    read_conditions,
    simulation,
    verify,
)
from scripts.run_grain_paraphrase_reference import MODEL, REVISION, encode_batches


def condition_spans(bank):
    """Extract an exact leading clause in the explicitly declared fixture grammar."""
    rows = []
    for row in bank.itertuples():
        end = row.text.index(",")
        rows.append(
            {
                "case_id": row.case_id,
                "start": 0,
                "end": end,
                "condition_text": row.text[:end],
                "full_text": row.text,
            }
        )
    return pd.DataFrame(rows)


def coupling(scores):
    """Measure induced changes in inferred condition at fixed condition wording."""
    rows = []
    for (q, part), f in scores.groupby(["query", "partition"]):
        paired = f.pivot(
            index=["condition", "context_variant", "response_variant"],
            columns="sign",
            values="condition_hat",
        )
        effect = paired[1] - paired[-1]
        # Descriptive decomposition on the fixed factorial, not a scoring correction.
        x = np.column_stack(
            [np.ones(len(f)), f.condition, f.sign, f.condition * f.sign]
        )
        beta = np.linalg.lstsq(x, f.condition_hat, rcond=None)[0]
        rows.append(
            {
                "query": q,
                "partition": part,
                "condition_accuracy": float(
                    np.mean(np.sign(f.condition_hat) == f.condition)
                ),
                "mean_abs_behavior_effect": float(effect.abs().mean()),
                "offset": beta[0],
                "condition_gain": beta[1],
                "behavior_gain": beta[2],
                "interaction_gain": beta[3],
                "factorial_remainder_rmse": float(
                    np.sqrt(np.mean((f.condition_hat - x @ beta) ** 2))
                ),
            }
        )
    return pd.DataFrame(rows)


def main():
    """Freeze a scope-only follow-up, collect local vectors and compare all authors."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    config, bank = verify(args.source)
    if args.output.exists():
        raise FileExistsError("Fresh output required")
    args.output.mkdir(parents=True)
    write(
        args.output / "contract.json",
        {
            "source": str(args.source),
            "source_summary": digest(args.source / "summary.json"),
            "source_manifest": digest(args.source / "reader_manifest.json"),
            "code": digest(Path(__file__)),
            "comparison": "Post-r1 scope-only; same model/alpha/source labels/event scores; exact leading clause extraction is fixture-specific.",
            "model": MODEL,
            "revision": REVISION,
        },
    )
    manifest = json.loads((args.source / "reader_manifest.json").read_text())
    for name, sha in manifest["files"].items():
        if digest(args.source / name) != sha:
            raise ValueError("Source reader cache changed")
    spans = condition_spans(bank)
    spans.to_csv(args.output / "public_condition_spans.csv", index=False)
    unique = list(dict.fromkeys(spans.condition_text))
    v, _ = encode_batches(unique, args.output / "encoder", MODEL, REVISION, cap=128)
    if json.loads((args.output / "encoder/contract.json").read_text())[
        "truncated_texts"
    ]:
        raise ValueError("Unexpected truncation")
    vectors = np.vstack([v[unique.index(t)] for t in spans.condition_text])
    np.save(args.output / "vectors.npy", vectors)
    scores, fitted = read_conditions(
        config, bank, vectors, pd.read_csv(args.source / "public_nli.csv")
    )
    old = pd.read_csv(args.source / "public_scored_bank.csv")
    np.testing.assert_allclose(
        scores.evidence_hat, old.evidence_hat, rtol=1e-7, atol=1e-7
    )
    scores.to_csv(args.output / "public_scored_bank.csv", index=False)
    write(args.output / "fitted_source.json", fitted)
    comparisons = pd.concat(
        [
            coupling(old).assign(scope="whole"),
            coupling(scores).assign(scope="condition_clause"),
        ]
    )
    comparisons.to_csv(args.output / "condition_coupling.csv", index=False)
    pred = simulation(config, scores)
    pred.to_csv(args.output / "profile_predictions.csv", index=False)
    aggregate_metrics(pred).to_csv(args.output / "metrics_by_seed.csv", index=False)
    aggregate_metrics(pred, False).to_csv(args.output / "metrics.csv", index=False)
    write(
        args.output / "summary.json",
        {
            "status": "complete",
            "unique_encoded_clauses": len(unique),
            "new_nli_calls": 0,
            "same_evidence_scores": True,
            "prediction_rows": len(pred),
            "files": {
                name: digest(args.output / name)
                for name in (
                    "public_condition_spans.csv",
                    "vectors.npy",
                    "public_scored_bank.csv",
                    "fitted_source.json",
                    "condition_coupling.csv",
                    "profile_predictions.csv",
                    "metrics_by_seed.csv",
                    "metrics.csv",
                )
            },
        },
    )
    print(comparisons.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
