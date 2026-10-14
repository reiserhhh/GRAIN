#!/usr/bin/env python3
"""Correct Gaussian reference integration without changing the primary text scores."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from grain_research.gptpro_foundation_v3.f14_metadata import sha256_file
from grain_research.gptpro_foundation_v3.f16_dynamic_gaussian_reference import (
    score_reference,
)
from scripts.run_grain_gptpro_f16_dynamic_text import discrepancy
from scripts.run_grain_gptpro_f16_t1_text import metrics, save


def score(inputs, bundle, arm, ev, nodes):
    """Bound memory; each posterior remains an independent author calculation."""
    return [
        r
        for i in range(0, len(inputs), 48)
        for r in score_reference(inputs[i : i + 48], bundle, arm, ev, nodes)
    ]


def main():
    """Fix 64-node eta quadrature before evaluation and check with 128 nodes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    code = [
        Path(__file__),
        root / "grain_research/gptpro_foundation_v3/f16_dynamic_gaussian_reference.py",
        root / "tests/test_grain_gptpro_f16_gaussian_reference.py",
    ]
    protocol = json.loads((args.run / "protocol_before_run.json").read_text())
    save(
        args.output / "protocol.json",
        {
            "code_hashes": {str(p): sha256_file(p) for p in code},
            "original_manifest_hash": sha256_file(args.run / "output_manifest.json"),
            "changed": "only integration of two Gaussian references;no new model/observation/target fitting",
            "reason": "priorQMC256 misses concentrated posterior tails;not changing primary four categorical arms",
            "method": "analytic conditional(a,b,state)|eta;64GaussHermite eta nodes;200originalsource models",
            "numerical_panel": "first12IDs plus per-arm lowest oldESS,64vs128;no target-based selection",
        },
    )
    rows, numerical = [], []
    for seed in protocol["seeds"]:
        folder = args.output / str(seed)
        folder.mkdir()
        source = args.run / str(seed)
        bundle = json.loads((source / "bundle.json").read_text())
        inputs = [
            json.loads(x)
            for x in (source / "application_inputs.jsonl").read_text().splitlines()
        ]
        evaluator = json.loads(
            (source / "application_rendering_evaluator.json").read_text()
        )
        ev = {(x["author_id"], x["event_id"]): x for x in evaluator}
        for arm in ("naive_gaussian", "eligible_continuous"):
            scores = score(inputs, bundle, arm, ev, 64)
            save(folder / f"{arm}_scores.json", scores)
            old = json.loads((source / f"{arm}_scores.json").read_text())
            indices = sorted(
                set(range(12)) | {int(np.argmin([x["qmc_ess_min"] for x in old]))}
            )
            fine = score([inputs[i] for i in indices], bundle, arm, ev, 128)
            coarse = [scores[i] for i in indices]
            diff = discrepancy(coarse, fine)
            save(
                folder / f"{arm}_quadrature_check.json",
                {"indices": indices, "nodes128": fine, "discrepancy": diff},
            )
            numerical.append(
                {"seed": seed, "arm": arm, "authors": len(indices), **diff}
            )
            for x, s in zip(inputs, scores, strict=True):
                for target, item in s["targets"].items():
                    rows.append(
                        {
                            "seed": seed,
                            "author_id": x["author_id"],
                            "arm": arm,
                            "target": target,
                            **{
                                k: item[k]
                                for k in ("score", "variance", "lower", "upper")
                            },
                        }
                    )
            print(
                json.dumps(
                    {
                        "seed": seed,
                        "arm": arm,
                        "64vs128": diff,
                        "seconds": time.monotonic() - start,
                    }
                ),
                flush=True,
            )
    # Evaluation targets come from the old frozen table, after corrected scoring.
    old_frame = pd.read_csv(args.run / "scores.csv")
    keys = ["seed", "author_id", "arm", "target"]
    corrected = pd.DataFrame(rows).merge(
        old_frame[keys + ["truth"]], on=keys, validate="one_to_one"
    )
    frame = pd.concat(
        [
            old_frame[~old_frame.arm.isin(["naive_gaussian", "eligible_continuous"])],
            corrected,
        ],
        ignore_index=True,
    )
    frame.to_csv(args.output / "scores.csv", index=False)
    tables, contrasts = [], []
    for target, subset in frame.groupby("target"):
        table = metrics(subset)
        table["target"] = target
        tables.append(table)
        for arm in sorted(set(subset.arm) - {"hybrid_joint"}):
            pair = subset[subset.arm == "hybrid_joint"].merge(
                subset[subset.arm == arm],
                on=["seed", "author_id"],
                suffixes=("_p", "_r"),
                validate="one_to_one",
            )
            d = (pair.score_p - pair.truth_p) ** 2 - (pair.score_r - pair.truth_r) ** 2
            rng, boot = np.random.default_rng(165), np.zeros(2000)
            for seed in protocol["seeds"]:
                x = d[pair.seed == seed].to_numpy()
                boot += x[rng.integers(0, len(x), (2000, len(x)))].mean(1) / len(
                    protocol["seeds"]
                )
            lo, hi = np.quantile(boot, [0.025, 0.975])
            contrasts.append(
                {
                    "target": target,
                    "reference": arm,
                    "delta_mse": float(d.mean()),
                    "ci_low": float(lo),
                    "ci_high": float(hi),
                }
            )
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(args.output / "metrics_by_seed.csv", index=False)
    table.groupby(["target", "arm"])[
        ["mse", "bias", "r", "coverage", "width"]
    ].mean().to_csv(args.output / "metrics.csv")
    pd.DataFrame(contrasts).to_csv(args.output / "paired_contrasts.csv", index=False)
    save(
        args.output / "summary.json",
        {
            "status": "GAUSSIAN_REFERENCES_ANALYTICALLY_MARGINALIZED",
            "authors": 600,
            "corrected_score_objects": len(corrected) // 2,
            "primary_four_arms_unchanged": True,
            "numerical_checks": numerical,
            "seconds": time.monotonic() - start,
            "scope": "numerical correction, not new psychological effect;underlying Gaussian model unchanged",
        },
    )
    artifacts = [
        p for p in args.output.rglob("*") if p.is_file() and not p.name.startswith("._")
    ]
    save(
        args.output / "manifest.json",
        {str(p.relative_to(args.output)): sha256_file(p) for p in artifacts},
    )


if __name__ == "__main__":
    main()
