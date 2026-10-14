#!/usr/bin/env python3
"""Diagnose inner-draw contamination without changing frozen primary intervals."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from grain_research.gptpro_foundation_v3.f13_monte_carlo import (
    inner_sampling_correction,
)


def main():
    """Read complete aggregate covariance files; keep all signed corrections."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads((args.input / "protocol_before_run.json").read_text())
    rows = []
    negative = 0
    for world in config["worlds"]:
        for seed in config["seeds"]:
            path = args.input / f"covariances_{world}_{seed}.json"
            if not path.exists():
                continue
            for r in json.loads(path.read_text()):
                result = inner_sampling_correction(
                    r["personal_parameter"], r["source"], config["outer"], 200
                )
                negative += int(
                    np.linalg.eigvalsh(result["source_corrected"]).min() < 0
                )
                for i, target in enumerate(
                    (
                        "current_state",
                        "future_observation_k4",
                        "future_observation_average4",
                    )
                ):
                    rows.append(
                        {
                            "world": world,
                            "seed": seed,
                            "author_id": r["author_id"],
                            "budget": r["budget"],
                            "target": target,
                            "source_raw": r["source"][i][i],
                            "personal_raw": r["personal_parameter"][i][i],
                            **{k: float(v[i, i]) for k, v in result.items()},
                        }
                    )
    with (args.output / "components.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "DIAGNOSTIC_ONLY_PRIMARY_INTERVALS_UNCHANGED",
        "rows": len(rows),
        "indefinite_corrected_source_matrices": negative,
        "assumptions": "B=200 fixed source models,J=200 independent identically distributed Laplace draws per model; not a source-population bias correction",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
