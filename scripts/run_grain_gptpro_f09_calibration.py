#!/usr/bin/env python3
"""Execute frozen composite-null calibration and independent validation."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.gptpro_foundation_v3.f09_detection import calibrate, score_oof
from grain_research.gptpro_foundation_v3.f09_rivals import (
    BETAS,
    EVEN,
    ODD,
    candidates,
    probability,
)


def main():
    """Keep all calibration arrays and independent validation/application counts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "seeds": [20260915, 20260916, 20260917],
        "calibration_per_atom": 2000,
        "threshold": 0.05,
        "validation": 100,
        "application": 200,
        "application_M4_fraction": 0.25,
        "design": "three independent trials at each even corner,then three at each odd corner",
        "scope": "detection subpacket;full process/class and odd-prefix predictions remain open",
    }
    (args.output / "protocol_before_run.json").write_text(
        json.dumps(protocol, indent=2) + "\n"
    )
    results = []
    atoms = candidates()["atoms"]
    for seed in protocol["seeds"]:
        calibration = calibrate(seed)
        np.savez_compressed(args.output / f"calibration_{seed}.npz", **calibration)
        for role_number, (role, n) in enumerate(
            (("validation", 100), ("application", 200))
        ):
            rng = np.random.default_rng(
                np.random.SeedSequence([seed, 9, 902, role_number])
            )
            # Cross atom labels independently of material draws; fixed25% M4 only application.
            labels = np.resize(np.arange(9), n)
            rng.shuffle(labels)
            outside = np.zeros(n, dtype=bool)
            if role == "application":
                outside[rng.choice(n, n // 4, replace=False)] = True
            beta_out = np.resize(np.array(BETAS), n)
            rng.shuffle(beta_out)
            for i in range(n):
                family, beta = (
                    ("M4", float(beta_out[i])) if outside[i] else atoms[labels[i]]
                )
                d = rng.binomial(3, probability(EVEN, family, beta))
                f = rng.binomial(3, probability(ODD, family, beta))
                score = score_oof(d, f, calibration["statistics"])
                results.append(
                    {
                        "seed": seed,
                        "role": role,
                        "index": i,
                        "family": family,
                        "beta": beta,
                        "outside": bool(outside[i]),
                        "development": d.tolist(),
                        "future": f.tolist(),
                        **{
                            k: v.tolist() if isinstance(v, np.ndarray) else v
                            for k, v in score.items()
                        },
                    }
                )
    (args.output / "individual_checks.json").write_text(json.dumps(results) + "\n")
    summaries = []
    for role, outsider in (
        ("validation", False),
        ("application", False),
        ("application", True),
    ):
        rows = [r for r in results if r["role"] == role and r["outside"] == outsider]
        n = len(rows)
        rate = sum(r["rejected"] for r in rows) / n
        den = 1 + 1.96**2 / n
        center = (rate + 1.96**2 / (2 * n)) / den
        radius = 1.96 * np.sqrt(rate * (1 - rate) / n + 1.96**2 / (4 * n * n)) / den
        summaries.append(
            {
                "role": role,
                "outside": outsider,
                "n": n,
                "rejection_rate": rate,
                "wilson_low": center - radius,
                "wilson_high": center + radius,
            }
        )
    with (args.output / "rejection_rates.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    summary = {
        "status": "CALIBRATION_AND_INDEPENDENT_CHECKS_COMPUTED",
        "calibration_authors": 54000,
        "validation_application_authors": 900,
        "results": summaries,
        "threshold_changed": False,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
