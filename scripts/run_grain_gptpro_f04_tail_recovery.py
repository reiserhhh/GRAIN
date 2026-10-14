#!/usr/bin/env python3
"""Run frozen F04 variance/tail targets on observed application prefixes."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "2")

import numpy as np
from scipy.stats import norm
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.gptpro_foundation_v3.f00 import sha256
from grain_research.gptpro_foundation_v3.f04 import CONDITIONS, grouped
from grain_research.gptpro_foundation_v3.f04_recovery import observed_arrays
from grain_research.gptpro_foundation_v3.f04_tail_recovery import predict_tail
from scripts.run_grain_gptpro_f04_application_recovery import dump, table


def main():
    """Estimate before loading latent targets, preserving all failed coverages."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve old runs")
    args.output.mkdir(parents=True)
    archive = args.output / "source_archive"
    archive.mkdir()
    source = json.loads((args.source / "summary.json").read_text())
    if bool(source["smoke"]) != args.smoke:
        raise ValueError("source mode mismatch")
    manifest = json.loads((args.source / "output_manifest.json").read_text())
    for entry in manifest["files"]:
        if sha256(args.source / entry["path"]) != entry["sha256"]:
            raise ValueError("changed source artifact")
    files = [
        Path(__file__),
        ROOT / "grain_research/gptpro_foundation_v3/f04_tail_recovery.py",
        ROOT / "grain_research/gptpro_foundation_v3/f04_recovery.py",
        ROOT / "scripts/run_grain_gptpro_f04_application_recovery.py",
    ]
    for path in files:
        (archive / (sha256(path) + "_" + path.name)).write_bytes(path.read_bytes())
    dump(
        args.output / "input_manifest.json",
        {
            "source_manifest_sha256": sha256(args.source / "output_manifest.json"),
            "files": [{"path": str(path), "sha256": sha256(path)} for path in files],
        },
    )
    rows = []
    for outcome in source["outcomes"]:
        world, seed = outcome["world"], outcome["seed"]
        root = args.source / f"{world}_{seed}"
        observed = grouped(
            [
                json.loads(line)
                for line in (root / "application.jsonl").read_text().splitlines()
            ]
        )
        pending = []
        for index, (author, events) in enumerate(observed.items()):
            for n in [8] if args.smoke else [8, 16, 24, 32]:
                for mode in (
                    ["linear", "four_cell"] if world in {"T", "Q"} else ["linear"]
                ):
                    prefix, future = events[:n], events[n:]
                    result = predict_tail(
                        prefix,
                        mode=mode,
                        interval="bootstrap" if world == "H" else "gaussian",
                        draws=80 if args.smoke else 2000,
                        seed=seed + index * 100 + n,
                    )
                    x, y = observed_arrays(prefix, mode)
                    residual = y - x @ result["beta"]
                    future_residual = np.array([])
                    if future:
                        xf, yf = observed_arrays(future, mode)
                        future_residual = yf - xf @ result["beta"]
                    pending.append((author, n, mode, result, residual, future_residual))
        truths = {
            row["author_id"]: row
            for row in json.loads((root / "truth.json").read_text())
        }
        for author, n, mode, result, residual, future_residual in pending:
            truth = truths[author]
            sigma = np.sqrt(truth["sigma2"])
            probability = (
                float(2 * student_t.sf(2 / np.sqrt(3 / 5), 5))
                if world == "H"
                else float(2 * norm.sf(2))
            )
            operational_true = (
                float(
                    2
                    * student_t.sf(
                        result["operational_threshold"] / (sigma * np.sqrt(3 / 5)), 5
                    )
                )
                if world == "H"
                else float(2 * norm.sf(result["operational_threshold"] / sigma))
            )
            # Nonlinear worlds have misspecified linear means. Keep their errors,
            # but do not label the chi-square interval an exact noise interval.
            exact_mean = world not in {"T", "Q"} or mode == "four_cell"
            events = observed[author]
            prefix_noise = np.array(
                [
                    row["y"] - truth["means"][list(CONDITIONS).index(row["z"])]
                    for row in events[:n]
                ]
            )
            row = {
                "world": world,
                "seed": seed,
                "author_id": author,
                "n": n,
                "mode": mode,
                **{key: value for key, value in result.items() if key != "beta"},
                "mean_correctly_specified": exact_mean,
                "true_variance": truth["sigma2"],
                "variance_squared_error": (result["variance"] - truth["sigma2"]) ** 2,
                "variance_covered": result["variance_ci_low"]
                <= truth["sigma2"]
                <= result["variance_ci_high"],
                "true_tail_probability": probability,
                "true_threshold": 2 * sigma,
                "prefix_true_threshold_tail": float(
                    np.mean(np.abs(residual) > 2 * sigma)
                ),
                "oracle_noise_tail_reference": float(
                    np.mean(np.abs(prefix_noise) > 2 * sigma)
                ),
                "operational_noise_probability": operational_true,
                "future_n": len(future_residual),
                "future_true_threshold_tail": float(
                    np.mean(np.abs(future_residual) > 2 * sigma)
                )
                if len(future_residual)
                else None,
                "future_operational_tail": float(
                    np.mean(np.abs(future_residual) > result["operational_threshold"])
                )
                if len(future_residual)
                else None,
            }
            rows.append(row)
        print(
            json.dumps({"world": world, "seed": seed, "tail_rows": len(pending)}),
            flush=True,
        )
    table(args.output / "variance_tail_by_author.csv", rows)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["world"], row["n"], row["mode"])].append(row)
    aggregate = []
    for key, values in groups.items():
        row = dict(zip(("world", "n", "mode"), key))
        row["authors"] = len(values)
        for metric in (
            "variance",
            "true_variance",
            "variance_squared_error",
            "variance_covered",
            "prefix_true_threshold_tail",
            "true_tail_probability",
            "oracle_noise_tail_reference",
            "prefix_operational_tail",
            "operational_noise_probability",
            "future_true_threshold_tail",
            "future_operational_tail",
        ):
            valid = [item[metric] for item in values if item[metric] is not None]
            row[metric] = float(np.mean(valid)) if valid else None
        aggregate.append(row)
    table(args.output / "variance_tail_summary.csv", aggregate)
    dump(
        args.output / "summary.json",
        {
            "status": "VARIANCE_TAIL_COMPUTATION_COMPLETE_AUDIT_PENDING",
            "smoke": args.smoke,
            "rows": len(rows),
            "H_bootstrap_draws": 80 if args.smoke else 2000,
            "source_refitted": False,
            "interval_note": "H fixed-design residual bootstrap; misspecified linear T/Q chi-square diagnostic only",
            "oracle_note": "latent sigma and noise only in terminal evaluation, never predictor",
        },
    )
    dump(
        args.output / "output_manifest.json",
        {
            "files": [
                {"path": str(path.relative_to(args.output)), "sha256": sha256(path)}
                for path in sorted(args.output.rglob("*"))
                if path.is_file() and not path.name.startswith("._")
            ]
        },
    )


if __name__ == "__main__":
    main()
