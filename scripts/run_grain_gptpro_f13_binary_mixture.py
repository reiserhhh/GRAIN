#!/usr/bin/env python3
"""Apply all source bootstraps to cached F08 histories without new reader calls."""

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np

from grain_research.gptpro_foundation_v3.f02 import c2_next_probability
from grain_research.gptpro_foundation_v3.f13_binary_mixture import BinarySourceEnsemble


def main():
    """Score before reading the next event; hidden observations only evaluate."""
    parser = argparse.ArgumentParser()
    for name in ("input", "source", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    files = [
        Path(__file__),
        Path("grain_research/gptpro_foundation_v3/f13_binary_mixture.py"),
        args.input / "observations.jsonl.gz",
        args.source / "protocol_before_run.json",
    ]
    fits = {}
    for world in ("c2_xor", "c2_iid"):
        for seed in (20260915, 20260916, 20260917):
            paths = [args.source / f"{world}_{seed}_b{b:03d}.json" for b in range(200)]
            bundles = [json.loads(p.read_text()) for p in paths]
            for b, bundle in enumerate(bundles):
                assert (bundle["world"], bundle["seed"], bundle["bootstrap"]) == (
                    world,
                    seed,
                    b,
                )
            fits[world, seed] = [b["fit"] for b in bundles]
            files.extend(paths)
    protocol = {
        "outer": 200,
        "inner": 200,
        "application_only": True,
        "smoke": args.smoke,
        "rng": "SeedSequence([seed,13,8,world_index,application_group_index,2])",
        "interpretation": "B1 finite-state working law, not B2/M posterior or calibrated psychological CI",
        "files": [
            {
                "path": str(p.resolve()),
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            }
            for p in files
        ],
    }
    (args.output / "protocol_before_run.json").write_text(
        json.dumps(protocol, indent=2) + "\n"
    )
    started = time.monotonic()
    rows_count = groups_count = 0
    max_identity = max_mc_error = 0.0
    writer = None
    with (
        gzip.open(args.input / "observations.jsonl.gz", "rt") as stream,
        gzip.open(args.output / "predictions.csv.gz", "wt", newline="") as output,
    ):
        rows = (json.loads(line) for line in stream)
        for _, group in itertools.groupby(
            rows, key=lambda r: (r["author_id"], r["panel"])
        ):
            records = list(group)
            if records[0]["role"] != "application":
                continue
            if args.smoke and groups_count == 2:
                break
            parts = records[0]["event_id"].split("_")
            world, seed = "_".join(parts[:2]), int(parts[2])
            panel = records[0]["panel"]
            histories = {
                c: BinarySourceEnsemble(fits[world, seed], panel == "complete")
                for c in ("oracle", "soft", "hard")
            }
            rng = np.random.default_rng(
                np.random.SeedSequence(
                    [seed, 13, 8, ("c2_xor", "c2_iid").index(world), groups_count, 2]
                )
            )
            hidden = []
            for row in records:
                for channel, history in histories.items():
                    if row["t"] >= 12:
                        result = history.components(rng, 200)
                        truth = c2_next_probability(
                            hidden[-2],
                            hidden[-1],
                            1,
                            "xor" if world == "c2_xor" else "iid",
                        )
                        error = result["mean"] - truth
                        result.update(
                            {
                                "true_next_p": truth,
                                "probability_squared_error": error**2,
                                "brier": (result["mean"] - row["observation"]) ** 2,
                                "working_probability_covered": int(
                                    abs(error)
                                    <= 1.96 * np.sqrt(result["probability_variance"])
                                ),
                            }
                        )
                        max_identity = max(
                            max_identity,
                            abs(
                                result["outcome_variance"]
                                - result["mean"] * (1 - result["mean"])
                            ),
                        )
                        max_mc_error = max(
                            max_mc_error, abs(result["mc_mean"] - result["mean"])
                        )
                        record = {
                            "world": world,
                            "seed": seed,
                            "author_id": row["author_id"],
                            "panel": panel,
                            "channel": channel,
                            "t": row["t"],
                            **result,
                        }
                        if writer is None:
                            writer = csv.DictWriter(output, fieldnames=list(record))
                            writer.writeheader()
                        writer.writerow(record)
                        rows_count += 1
                    if channel == "oracle":
                        p, q = row["material_pole"], row["qualification"]
                    else:
                        p, q = row["pole_score"], row["qualification_score"]
                        if channel == "hard":
                            p, q = float(p >= 0.5), float(q >= 0.5)
                    history.observe(p, 1.0 if panel == "complete" else q)
                hidden.append(row["observation"])
            groups_count += 1
            if groups_count % 100 == 0:
                checkpoint = {
                    "groups": groups_count,
                    "rows": rows_count,
                    "elapsed_seconds": time.monotonic() - started,
                }
                (args.output / "checkpoint.json").write_text(
                    json.dumps(checkpoint) + "\n"
                )
                print(json.dumps(checkpoint), flush=True)
    summary = {
        "status": "B1_SOURCE_HISTORY_MIXTURE_COMPLETE",
        "groups": groups_count,
        "rows": rows_count,
        "equivalent_inner_state_draws": rows_count * 200 * 200,
        "max_total_variance_error": max_identity,
        "max_mc_mean_error": max_mc_error,
        "elapsed_seconds": time.monotonic() - started,
        "open": "B2/M personal uncertainty; author-cluster calibration analysis; missingness/shift/history contracts",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if max_identity > 1e-12:
        raise ValueError("total binary variance identity failed")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
