#!/usr/bin/env python3
"""Replay F07 information through independent presence/content decomposition."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.gptpro_foundation_v3.f00 import sha256


def separate_information(z, presence, epsilon, variant):
    """Decompose the full observed likelihood into selection and conditional content."""
    p = expit(0.5 * z)
    gradient = p * (1 - p) * np.array([1.0, z])
    s = epsilon + (1 - 2 * epsilon) * p
    ds = (1 - 2 * epsilon) * gradient
    if variant != "base":
        s, ds = 0.8 * s + 0.1, 0.8 * ds
    c, dc = presence, np.zeros(2)
    if variant == "mnar":
        c = presence * (0.5 + 0.5 * p)
        dc = 0.5 * presence * gradient
    selection = np.outer(dc, dc) / (c * (1 - c)) if 0 < c < 1 else np.zeros((2, 2))
    content = c * np.outer(ds, ds) / (s * (1 - s))
    return selection + content, selection, content


def main():
    """Verify every budget matrix and the data-dependent missingness contribution."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve previous audit")
    args.output.mkdir(parents=True)
    manifest = json.loads((args.input / "output_manifest.json").read_text())["files"]
    for item in manifest:
        assert sha256(args.input / item["path"]) == item["sha256"]
    data = pd.read_csv(args.input / "information_by_budget.csv")
    output, maximum = [], 0.0
    for row in data.itertuples(index=False):
        total, selection = np.zeros((2, 2)), np.zeros((2, 2))
        for t in range(row.opportunities):
            matrix, part, _ = separate_information(
                -1.0 if t % 2 == 0 else 1.0, row.presence, row.flip, row.variant
            )
            total += matrix
            selection += part
        effective = total[1, 1] - total[0, 1] ** 2 / total[0, 0]
        expected = np.array([total[0, 0], total[0, 1], total[1, 1], effective])
        stored = np.array([row.I_aa, row.I_ab, row.I_bb, row.I_b_given_a])
        maximum = max(maximum, float(np.max(np.abs(expected - stored))))
        np.testing.assert_allclose(expected, stored, atol=1e-10, rtol=1e-8)
        output.append(
            {
                "world": row.world,
                "variant": row.variant,
                "level": row.level,
                "budget": row.budget,
                "Ibb_total": total[1, 1],
                "Ibb_selection": selection[1, 1],
                "selection_fraction_Ibb": selection[1, 1] / total[1, 1],
            }
        )
    mc = pd.read_csv(args.input / "monte_carlo_checks.csv")
    assert ((mc.missing + mc.positive + mc.negative) == mc.n).all()
    for row in mc.itertuples(index=False):
        matching = data[
            (data.world == row.world)
            & (data.variant == row.variant)
            & (data.level == row.level)
        ].iloc[0]
        matrix, _, _ = separate_information(
            row.z, matching.presence, matching.flip, row.variant
        )
        np.testing.assert_allclose(matrix[1, 1], row.Ibb_exact, atol=1e-12, rtol=0)
    pd.DataFrame(output).to_csv(
        args.output / "selection_content_decomposition.csv", index=False
    )
    summary = {
        "status": "PRESENCE_CONTENT_FISHER_REPLAY_PASS",
        "budget_matrices": len(data),
        "mc_reference_rows": len(mc),
        "verified_hashes": len(manifest),
        "maximum_numeric_difference": maximum,
        "scope": "information identity audit, not fitted MAP performance or measured psychological openness",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "runner_snapshot.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
