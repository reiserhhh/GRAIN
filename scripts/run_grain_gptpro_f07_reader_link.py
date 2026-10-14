#!/usr/bin/env python3
"""Connect actual frozen F08 hard-reader emissions to F07 information laws."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.gptpro_foundation_v3.f07_channels import (
    categorical_information,
    efficient_slope_information,
)
from grain_research.gptpro_foundation_v3.f07_reader_channel import (
    fit_emission,
    reader_law,
)


def main():
    """Freeze source emissions; report held-out channel discrepancies separately."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--reader", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    input_path = args.reader / "reader_scores.csv"
    rows = list(csv.DictReader(input_path.open()))
    source = [r for r in rows if r["role"] == "source"]
    emissions = {panel: fit_emission(source, panel) for panel in ("complete", "mixed")}
    (args.output / "source_channel.json").write_text(
        json.dumps(
            {
                "source_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                "symbols": ["missing", "positive", "negative"],
                "emissions": {k: v.tolist() for k, v in emissions.items()},
                "scope": "TEXT_SYNTHETIC;source grammar design weights,not human observation frequencies",
            },
            indent=2,
        )
        + "\n"
    )
    info = []
    checks = []
    held = []
    for panel, emission in emissions.items():
        for theta in ([0, 0.5], [-1, 0.2], [1, 1]):
            total = np.zeros((2, 2))
            for z in (-1, 1):
                p, d = reader_law(theta, z, emission)
                information = categorical_information(p, d)
                total += information / 2
                # Independent scalar mixture identity: dP/dtheta=(E1-E0)*dp/dtheta.
                latent = 1 / (1 + np.exp(-(theta[0] + theta[1] * z)))
                active = p > 0
                scalar = (latent * (1 - latent)) ** 2 * np.sum(
                    (emission[1, active] - emission[0, active]) ** 2 / p[active]
                )
                reference = scalar * np.outer([1, z], [1, z])
                np.testing.assert_allclose(information, reference, atol=1e-12)
                for h in (1e-4, 1e-5, 1e-6):
                    numerical = np.column_stack(
                        [
                            (
                                reader_law(np.array(theta) + h * e, z, emission)[0]
                                - reader_law(np.array(theta) - h * e, z, emission)[0]
                            )
                            / (2 * h)
                            for e in np.eye(2)
                        ]
                    )
                    delta = float(np.max(np.abs(d - numerical)))
                    if delta > 1e-8:
                        raise ValueError("derivative replay")
                    checks.append(
                        {
                            "panel": panel,
                            "a": theta[0],
                            "b": theta[1],
                            "z": z,
                            "step": h,
                            "max_abs_difference": delta,
                        }
                    )
            info.append(
                {
                    "panel": panel,
                    "a": theta[0],
                    "b": theta[1],
                    "Iaa": total[0, 0],
                    "Iab": total[0, 1],
                    "Ibb": total[1, 1],
                    "efficient_slope": efficient_slope_information(total),
                    "unit": "per_declared_opportunity",
                }
            )
        for role in ("validation", "application"):
            # Direct enumeration of held-out design, not a new fitted source model.
            observed = np.zeros((2, 3))
            bank = {
                (int(r["pole"]), r["subject_status"], int(r["variant"])): r
                for r in rows
                if r["role"] == role
            }
            statuses = (
                ("actual", "negated")
                if panel == "complete"
                else ("actual", "quoted", "hypothetical", "negated")
            )
            for truth in (0, 1):
                for status in statuses:
                    choices = [truth] if status in ("actual", "negated") else [0, 1]
                    for pole in choices:
                        for v in range(4):
                            r = bank[(pole, status, v)]
                            symbol = (
                                0
                                if panel == "mixed"
                                and float(r["qualification_score"]) < 0.5
                                else 1
                                if float(r["pole_score"]) >= 0.5
                                else 2
                            )
                            observed[truth, symbol] += 1 / (
                                len(statuses) * len(choices) * 4
                            )
            held.append(
                {
                    "panel": panel,
                    "role": role,
                    "maximum_emission_difference": float(
                        np.max(np.abs(observed - emission))
                    ),
                    "frozen_source_replaced": False,
                }
            )
    for name, data in (
        ("information.csv", info),
        ("derivative_checks.csv", checks),
        ("held_out_channel_checks.csv", held),
    ):
        with (args.output / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    result = {
        "status": "MEASURED_HARD_CHANNEL_LINK_COMPUTED",
        "information_rows": len(info),
        "derivative_checks": len(checks),
        "held_out_checks": held,
        "scope": "measured hard emissions on fixed synthetic grammar;soft score calibration remains distinct;five text levels not observed",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
