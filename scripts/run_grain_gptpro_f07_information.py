#!/usr/bin/env python3
"""Compute corrected F07 categorical information, derivatives and MC checks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.gptpro_foundation_v3.f00 import sha256
from grain_research.gptpro_foundation_v3.f07_channels import (
    COSTS,
    OPPORTUNITIES,
    binary_law,
    categorical_information,
    channel_information,
    efficient_slope_information,
    ordinal_law,
    world_parameters,
)


def classify(values):
    """Describe exact computed shapes without forcing the world's expected name."""
    change = np.diff(values)
    if np.all(np.abs(change) < 1e-12):
        return "flat"
    if np.all(change >= -1e-12):
        return "increasing"
    if np.all(change <= 1e-12):
        return "decreasing"
    maximum = int(np.argmax(values))
    if (
        0 < maximum < len(values) - 1
        and np.all(change[:maximum] >= -1e-12)
        and np.all(change[maximum:] <= 1e-12)
    ):
        return "interior_peak"
    return "mixed"


def main():
    """Build mathematical evidence only; population MAP experiments remain open."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve earlier results")
    args.output.mkdir(parents=True)
    theta = np.array([0.0, 0.5])
    rows, derivatives, monte_carlo, trends = [], [], [], []
    for world_index, world in enumerate(("U", "D", "P", "flat")):
        presence, flip = world_parameters(world)
        for variant_index, variant in enumerate(("base", "offtarget", "mnar")):
            curve = []
            for level in range(5):
                information = channel_information(theta, level, world, variant)
                token = efficient_slope_information(information["per_token"])
                curve.append(token)
                for budget_kind, n in (
                    ("tokens400", int(400 // COSTS[level] * OPPORTUNITIES[level])),
                    ("responses32", 32),
                ):
                    matrix = np.zeros((2, 2))
                    for t in range(n):
                        matrix += categorical_information(
                            *binary_law(
                                theta,
                                -1.0 if t % 2 == 0 else 1.0,
                                presence[level],
                                flip[level],
                                variant,
                            )
                        )
                    rows.append(
                        {
                            "world": world,
                            "variant": variant,
                            "level": level + 1,
                            "presence": presence[level],
                            "flip": flip[level],
                            "budget": budget_kind,
                            "opportunities": n,
                            "I_aa": matrix[0, 0],
                            "I_ab": matrix[0, 1],
                            "I_bb": matrix[1, 1],
                            "I_b_given_a": efficient_slope_information(matrix),
                            "per_token_balanced": token,
                            "per_opportunity_balanced": efficient_slope_information(
                                information["per_opportunity"]
                            ),
                        }
                    )
                for z in (-1.0, 1.0):
                    p, derivative = binary_law(
                        theta, z, presence[level], flip[level], variant
                    )
                    for delta in (1e-4, 1e-5, 1e-6):
                        numeric = np.column_stack(
                            [
                                (
                                    binary_law(
                                        theta + np.eye(2)[j] * delta,
                                        z,
                                        presence[level],
                                        flip[level],
                                        variant,
                                    )[0]
                                    - binary_law(
                                        theta - np.eye(2)[j] * delta,
                                        z,
                                        presence[level],
                                        flip[level],
                                        variant,
                                    )[0]
                                )
                                / (2 * delta)
                                for j in range(2)
                            ]
                        )
                        error = float(np.max(np.abs(numeric - derivative)))
                        relative = error / max(float(np.max(np.abs(derivative))), 1e-12)
                        derivatives.append(
                            {
                                "world": world,
                                "variant": variant,
                                "level": level + 1,
                                "z": z,
                                "step": delta,
                                "max_abs_error": error,
                                "relative_error": relative,
                            }
                        )
                    score = np.divide(
                        derivative,
                        p[:, None],
                        out=np.zeros_like(derivative),
                        where=p[:, None] > 0,
                    )
                    expected = categorical_information(p, derivative)
                    for seed in (20260915, 20260916, 20260917):
                        rng = np.random.default_rng(
                            np.random.SeedSequence(
                                [seed, 7, world_index, variant_index, level, int(z > 0)]
                            )
                        )
                        counts = rng.multinomial(100000, p)
                        empirical = np.einsum(
                            "i,ij,ik->jk", counts / 100000, score, score
                        )
                        values = score[:, 1] ** 2
                        se = np.sqrt(
                            max(float(p @ values**2 - (p @ values) ** 2), 0) / 100000
                        )
                        monte_carlo.append(
                            {
                                "world": world,
                                "variant": variant,
                                "level": level + 1,
                                "z": z,
                                "seed": seed,
                                "n": 100000,
                                "missing": int(counts[0]),
                                "positive": int(counts[1]),
                                "negative": int(counts[2]),
                                "Ibb_exact": expected[1, 1],
                                "Ibb_mc": empirical[1, 1],
                                "Ibb_mc_se": se,
                                "matrix_max_abs_error": float(
                                    np.max(np.abs(empirical - expected))
                                ),
                            }
                        )
            trends.append(
                {
                    "world": world,
                    "variant": variant,
                    "per_token_shape": classify(curve),
                    "levels": json.dumps(curve),
                }
            )
    ordinal = []
    for z in (-1.0, 1.0):
        p, d = ordinal_law(theta, z)
        matrix = categorical_information(p, d)
        for k in range(5):
            ordinal.append(
                {
                    "z": z,
                    "category": k,
                    "probability": p[k],
                    "d_a": d[k, 0],
                    "d_b": d[k, 1],
                    "information_b_contribution": d[k, 1] ** 2 / p[k],
                }
            )
        assert (
            abs(
                matrix[1, 1]
                - sum(x["information_b_contribution"] for x in ordinal if x["z"] == z)
            )
            < 1e-12
        )
    checks = pd.DataFrame(derivatives)
    assert (
        checks.groupby(["world", "variant", "level", "z"]).relative_error.apply(
            lambda x: (x < 1e-4).sum()
        )
        >= 2
    ).all()
    for filename, values in (
        ("information_by_budget.csv", rows),
        ("derivative_checks.csv", derivatives),
        ("monte_carlo_checks.csv", monte_carlo),
        ("curve_shapes.csv", trends),
        ("ordinal_reference.csv", ordinal),
    ):
        pd.DataFrame(values).to_csv(args.output / filename, index=False)
    result = {
        "status": "CORRECTED_CHANNEL_INFORMATION_COMPUTED_INFERENCE_PENDING",
        "information_rows": len(rows),
        "derivative_rows": len(derivatives),
        "mc_rows": len(monte_carlo),
        "max_derivative_relative_error": float(checks.relative_error.max()),
        "flat_definition": "per-token balanced information calibrated at a=0,b=.5; not discovered invariance",
        "remaining": [
            "exact MAP source/validation/application experiments",
            "reader-derived confusion matrices from F08",
            "independent audit and original task contracts",
        ],
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output / "source_archive").mkdir()
    for path in (
        Path(__file__),
        ROOT / "grain_research/gptpro_foundation_v3/f07_channels.py",
    ):
        (args.output / "source_archive" / path.name).write_bytes(path.read_bytes())
    manifest = [
        {"path": str(p.relative_to(args.output)), "sha256": sha256(p)}
        for p in args.output.rglob("*")
        if p.is_file() and not p.name.startswith("._")
    ]
    (args.output / "output_manifest.json").write_text(
        json.dumps({"files": manifest}, indent=2) + "\n"
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
