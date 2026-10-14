#!/usr/bin/env python3
"""Replay cached F08 B1 predictions with explicit probability/outcome variance."""

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import pickle
from pathlib import Path

import numpy as np

from grain_research.gptpro_foundation_v3.f02 import generate_world
from grain_research.gptpro_foundation_v3.f08_estimators import QualifiedHistory
from grain_research.gptpro_foundation_v3.f13_binary import next_probability_components
from scripts.audit_grain_gptpro_f08_states import source_replay


def main():
    """No new readers/fits or outcomes used to update predictions before scoring."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads((args.input / "protocol.json").read_text())
    models = {}
    for world in ("c2_xor", "c2_iid"):
        for seed in config["seeds"]:
            with (args.input / f"source_{world}_{seed}.pkl").open("rb") as f:
                model = pickle.load(f)
            records, _ = generate_world(world, seed, counts=config["counts"])
            initial, mean = source_replay(
                [r for r in records if r["role"] == "source"], model
            )
            models[(world, seed)] = model, initial, mean
    count, max_prediction, max_identity = 0, 0.0, 0.0
    with (
        gzip.open(args.input / "predictions.jsonl.gz", "rt") as old,
        gzip.open(args.input / "observations.jsonl.gz", "rt") as stream,
        gzip.open(args.output / "components.csv.gz", "wt", newline="") as output,
    ):
        expected = (r for line in old if (r := json.loads(line))["candidate"] == "b1")
        rows = (json.loads(line) for line in stream)
        writer = None
        def group_key(row):
            # Existing simulator event address routes the frozen source bundle;
            # neither ID nor seed is a feature supplied to the history scorer.
            parts = row["event_id"].split("_")
            world, seed = "_".join(parts[:2]), int(parts[2])
            if (world, seed) not in models:
                raise ValueError("unknown source bundle address")
            return world, seed, row["author_id"], row["panel"]

        for key, group in itertools.groupby(rows, key=group_key):
            world, seed, author, panel = key
            model, initial, mean = models[(world, seed)]
            histories = {
                c: QualifiedHistory(
                    model,
                    initial,
                    mean,
                    mode="independent" if panel == "complete" else "filtered",
                )
                for c in ("oracle", "soft", "hard")
            }
            for row in group:
                for channel, history in histories.items():
                    if row["t"] >= 12:
                        result = next_probability_components(
                            history.pair, history.transition
                        )
                        reference = next(expected)
                        assert (
                            reference["world"],
                            reference["seed"],
                            reference["author_id"],
                            reference["panel"],
                            reference["t"],
                            reference["channel"],
                        ) == (world, seed, author, panel, row["t"], channel)
                        max_prediction = max(
                            max_prediction,
                            abs(
                                float(np.clip(result["mean"], 1e-6, 1 - 1e-6))
                                - reference["p"]
                            ),
                        )
                        max_identity = max(
                            max_identity,
                            abs(
                                result["total_outcome_variance"]
                                - result["mean"] * (1 - result["mean"])
                            ),
                        )
                        record = {
                            "world": world,
                            "seed": seed,
                            "author_id": author,
                            "role": row["role"],
                            "panel": panel,
                            "t": row["t"],
                            "channel": channel,
                            **result,
                        }
                        if writer is None:
                            writer = csv.DictWriter(output, fieldnames=list(record))
                            writer.writeheader()
                        writer.writerow(record)
                        count += 1
                    if channel == "oracle":
                        p, q = float(row["material_pole"]), float(row["qualification"])
                    else:
                        p, q = row["pole_score"], row["qualification_score"]
                        if channel == "hard":
                            p, q = float(p >= 0.5), float(q >= 0.5)
                    history.observe(p, 1.0 if panel == "complete" else q)
        assert next(expected, None) is None
    summary = {
        "status": "FIXED_SOURCE_B1_PROBABILITY_OUTCOME_DECOMPOSITION",
        "rows": count,
        "max_prediction_error": max_prediction,
        "max_total_variance_identity_error": max_identity,
        "scope": "working filtered state law; no new calibration evidence or source/personal parameter uncertainty; B2/M remain open",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "input_manifest.json").write_text(
        json.dumps(
            [
                {
                    "path": str(p.resolve()),
                    "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                }
                for p in (
                    args.input / "protocol.json",
                    args.input / "observations.jsonl.gz",
                    args.input / "predictions.jsonl.gz",
                )
            ]
        )
        + "\n"
    )
    if max(max_prediction, max_identity) > 1e-10:
        raise ValueError("binary replay mismatch")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
