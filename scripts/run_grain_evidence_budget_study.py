#!/usr/bin/env python3
"""Forecast detector coverage from source authors before evaluating held authors."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.evidence_budget import (
    beta_coverage,
    finite_coverage,
    fit_beta_binomial,
)
from grain_research.evidence_contracts import verify_frozen_top1
from scripts.prepare_grain_predecessor_reference import BASE, DEFAULT, digest, write


def run(args):
    """Use original fixed IPIP detector, not newly unreviewed expanded anchors."""
    if args.output.exists():
        raise ValueError("Use a new output directory")
    args.output.mkdir(parents=True)
    manifest = json.loads((args.material / "material_manifest.json").read_text())
    path = args.material / "private_material.json"
    if digest(path) != manifest["files"][path.name]:
        raise ValueError("Changed material")
    data = json.loads(path.read_text())
    evidence = pd.read_csv(args.evidence)
    verification = verify_frozen_top1(
        args.material, args.evidence.parent, evidence, data
    )
    bundles = data["bundles"]
    if any(b["source_group"] != "big5_only" for b in bundles):
        raise ValueError("Only authorized Big5-only material")
    n_comments = np.array([sum(len(v) for v in b["views"]) for b in bundles])
    if not np.all(n_comments == 36):
        raise ValueError("Fixed 36-comment protocol changed")
    source = np.array([b["official_folds"]["big5"] != 0 for b in bundles])
    hits = evidence.loc[
        evidence.accepted, ["author_index", "comment_index"]
    ].drop_duplicates()
    count = np.zeros(len(bundles), dtype=int)
    for i, group in hits.groupby("author_index"):
        if not 0 <= i < len(count) or not group.comment_index.between(0, 35).all():
            raise ValueError("Detector address outside author/comment range")
        count[int(i)] = len(group)
    fitted = fit_beta_binomial(count[source], 36)
    pooled_p = float(count[source].mean() / 36)
    budgets = [1, 4, 8, 16, 24, 32, 36]
    forecasts = [
        {
            "budget": b,
            "iid_coverage": 1 - (1 - pooled_p) ** b,
            "heterogeneous_coverage": beta_coverage(b, fitted["alpha"], fitted["beta"]),
        }
        for b in budgets
    ]
    # Save the forecast before any held-author coverage or bootstrap is computed.
    write(
        args.output / "frozen_forecast.json",
        {
            "fit": fitted,
            "source_authors": int(source.sum()),
            "pooled_comment_p": pooled_p,
            "forecasts": forecasts,
        },
    )
    write(
        args.output / "contract.json",
        {
            "response": "at least one original IPIP/paraphrase detector match, not validated psychological evidence",
            "fit": "504 non-fold0 authors; binomial and beta-binomial specified before held evaluation",
            "evaluation": "120 fold0 authors; exact finite-pool expected coverage, author bootstrap conditional on fitted source model",
            "history": "existing development material, not new untouched lockbox",
            "sequential_stopping": False,
            "personality_outcomes_accessed": False,
            "source_verification": verification,
            "hashes": {
                "material": digest(path),
                "evidence": digest(args.evidence),
                "runner": digest(Path(__file__)),
                "model": digest(ROOT / "grain_research/evidence_budget.py"),
            },
        },
    )
    rng = np.random.default_rng(731)
    held = count[~source]
    indices = rng.integers(len(held), size=(2000, len(held)))
    rows = []
    paired_errors = []
    for forecast in forecasts:
        coverage = finite_coverage(held, 36, forecast["budget"])
        bootstrap = coverage[indices].mean(axis=1)
        paired_errors.append(
            np.abs(forecast["heterogeneous_coverage"] - bootstrap)
            - np.abs(forecast["iid_coverage"] - bootstrap)
        )
        for label in ("iid", "heterogeneous"):
            prediction = forecast[label + "_coverage"]
            rows.append(
                {
                    "budget": forecast["budget"],
                    "method": label,
                    "predicted_coverage": prediction,
                    "observed_expected_coverage": float(coverage.mean()),
                    "observed_ci_low": float(np.quantile(bootstrap, 0.025)),
                    "observed_ci_high": float(np.quantile(bootstrap, 0.975)),
                    "signed_calibration_error": float(prediction - coverage.mean()),
                    "absolute_calibration_error": float(
                        abs(prediction - coverage.mean())
                    ),
                    "expected_brier": float(
                        np.mean(prediction**2 + (1 - 2 * prediction) * coverage)
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "coverage_predictions.csv", index=False)
    pd.DataFrame(
        {
            "author_index": np.arange(len(count)),
            "is_source": source,
            "matched_comments": count,
        }
    ).to_csv(args.output / "private_author_counts.csv", index=False)
    summary = {
        "status": "COMPLETE",
        "source_authors": int(source.sum()),
        "held_authors": len(held),
        "source_matched_authors": int((count[source] > 0).sum()),
        "held_matched_authors": int((held > 0).sum()),
        "fit": fitted,
        "count_diagnostics": {
            name: {
                "mean": float(values.mean()),
                "variance": float(values.var(ddof=1)),
                "zero_share": float(np.mean(values == 0)),
            }
            for name, values in (("source", count[source]), ("held", held))
        },
        "paired_mae_delta_heterogeneous_minus_iid_ci": np.quantile(
            np.mean(paired_errors, axis=0), [0.025, 0.975]
        ).tolist(),
        "diagnostic_scope": "held distribution inspected only after frozen forecasts; no refitting or model selection",
        "mean_absolute_coverage_error": frame.groupby("method")
        .absolute_calibration_error.mean()
        .to_dict(),
        "scope": "availability forecasting, not reader precision or personality prediction; a mixture is not a causal author trait",
    }
    write(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--material", type=Path, default=DEFAULT)
    p.add_argument(
        "--evidence",
        type=Path,
        default=BASE
        / "predecessor_integration_v1/paraphrase_r1/private_sentence_evidence.csv",
    )
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args())
