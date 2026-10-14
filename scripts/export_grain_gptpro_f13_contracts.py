#!/usr/bin/env python3
"""Cold-export actual nested scores and scoped calibration without promotion."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from grain_research.gptpro_foundation_v3.f13_contract import (
    calibrate_intervals,
    measurement_result,
)


def main():
    """Replay saved predictions through the same downstream result contract."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = [
        args.input / "predictions.csv",
        args.input / "metrics.csv",
        args.input / "protocol_before_run.json",
        Path(__file__),
        Path("grain_research/gptpro_foundation_v3/f13_contract.py"),
    ]
    (args.output / "input_manifest.json").write_text(
        json.dumps(
            [
                {
                    "path": str(p.resolve()),
                    "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                }
                for p in inputs
            ],
            indent=2,
        )
        + "\n"
    )
    frame = pd.read_csv(args.input / "predictions.csv")
    keys = ["world", "missingness", "seed", "rho_assumed", "method", "target"]
    old = pd.read_csv(args.input / "metrics.csv").set_index(keys)
    checks = []
    statuses = {}
    count = 0
    with gzip.open(args.output / "scores.jsonl.gz", "wt") as stream:
        for key, group in frame.groupby(keys):
            scope = dict(zip(keys, key, strict=True)) | {
                "population": "synthetic_application",
                "source_observation": "complete",
                "residual_kernel": "declared_not_learned",
            }
            half = 1.96 * np.sqrt(group.variance)
            check = calibrate_intervals(
                group.truth,
                group.score - half,
                group.score + half,
                group.author_id.tolist(),
                scope,
            )
            previous = old.loc[key]
            np.testing.assert_allclose(
                [check["coverage"], *check["wilson_95"], check["mean_width"]],
                [
                    previous.coverage,
                    previous.wilson_low,
                    previous.wilson_high,
                    previous.mean_width,
                ],
                atol=1e-12,
            )
            assert check["decision"] == previous.decision
            checks.append(check)
            for row in group.itertuples():
                result = measurement_result(
                    row.score,
                    row.variance,
                    scope,
                    "observed" if row.observed else "model_borrowed",
                    check,
                )
                result.update(
                    author_id=row.author_id,
                    observed_events=row.observed,
                    calibration_use="evaluation_annotation_not_model_selection",
                )
                stream.write(json.dumps(result) + "\n")
                statuses[result["calibration_check"]] = (
                    statuses.get(result["calibration_check"], 0) + 1
                )
                count += 1
    (args.output / "calibration_checks.json").write_text(
        json.dumps(checks, indent=2) + "\n"
    )
    summary = {
        "status": "COLD_UNCERTAINTY_CONTRACT_EXPORT_COMPLETE",
        "scores": count,
        "calibration_cells": len(checks),
        "score_annotation_counts": statuses,
        "scope": "export and finite same-scope coverage assessment; not F16 end-to-end integration or independent human validity",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
