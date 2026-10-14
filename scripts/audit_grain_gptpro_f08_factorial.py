#!/usr/bin/env python3
"""Alternate state replay for the two missing cells of F08's factorial design."""

import argparse
import csv
import gzip
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from audit_grain_gptpro_f08_states import Reference, generate_world, source_replay


def main():
    """Use enumerated-state reference and replay author-level decompositions."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((args.base / "protocol.json").read_text())
    bundles = {}
    for world in ("c2_xor", "c2_iid"):
        for seed in protocol["seeds"]:
            with (args.base / f"source_{world}_{seed}.pkl").open("rb") as stream:
                fitted = pickle.load(stream)
            records, _ = generate_world(world, seed, counts=protocol["counts"])
            initial, mean = source_replay(
                [r for r in records if r["role"] == "source"], fitted
            )
            bundles[(world, seed)] = (fitted, initial, mean)
    maximum = 0.0
    n = 0
    previous = None
    losses = defaultdict(list)
    with (
        gzip.open(args.base / "observations.jsonl.gz", "rt") as inputs,
        gzip.open(args.input / "predictions.jsonl.gz", "rt") as stream,
    ):
        predictions = iter(stream)
        for line in inputs:
            row = json.loads(line)
            if row["panel"] != "mixed":
                continue
            if row["author_id"] != previous:
                parts = row["event_id"].split("_")
                world = "_".join(parts[:2])
                seed = int(parts[2])
                fitted, initial, mean = bundles[(world, seed)]
                refs = {
                    c: Reference(fitted, initial, mean, "mixed")
                    for c in (
                        "reader_pole_oracle_qualification",
                        "oracle_pole_reader_qualification",
                    )
                }
                previous = row["author_id"]
            for channel, ref in refs.items():
                if row["t"] >= 12:
                    for candidate, value in ref.scores().items():
                        saved = json.loads(next(predictions))
                        assert (
                            saved["event_id"],
                            saved["channel"],
                            saved["candidate"],
                        ) == (row["event_id"], channel, candidate)
                        maximum = max(maximum, abs(value - saved["p"]))
                        if abs(value - saved["p"]) > 1e-10:
                            raise ValueError("independent score mismatch")
                        y = row["observation"]
                        b = (value - y) ** 2
                        l = -np.log(value if y else 1 - value)
                        losses[
                            (
                                world,
                                seed,
                                row["author_id"],
                                row["role"],
                                channel,
                                candidate,
                            )
                        ].append([b, l])
                        n += 1
                p = (
                    row["pole_score"]
                    if channel == "reader_pole_oracle_qualification"
                    else row["material_pole"]
                )
                q = (
                    row["qualification"]
                    if channel == "reader_pole_oracle_qualification"
                    else row["qualification_score"]
                )
                ref.update(p, q)
        assert next(predictions, None) is None
    metric_count = 0
    for row in csv.DictReader((args.input / "metrics_by_author.csv").open()):
        key = (
            row["world"],
            int(row["seed"]),
            row["author_id"],
            row["role"],
            row["channel"],
            row["candidate"],
        )
        np.testing.assert_allclose(
            np.mean(losses[key], axis=0),
            [float(row["brier"]), float(row["log_loss"])],
            atol=1e-10,
        )
        metric_count += 1
    groups = defaultdict(lambda: defaultdict(list))
    decomposition_count = 0
    for row in csv.DictReader((args.input / "author_decomposition.csv").open()):
        a, b, c, d = [float(row[k]) for k in ("L00", "L10", "L01", "L11")]
        for name, expected in (
            ("polarity", b - a),
            ("qualification", c - a),
            ("interaction", (d - c) - (b - a)),
            ("total", d - a),
        ):
            np.testing.assert_allclose(float(row[name]), expected, atol=1e-12)
            if row["role"] == "application":
                groups[(row["world"], row["candidate"], row["loss"], name)][
                    int(row["seed"])
                ].append(expected)
        decomposition_count += 1
    max_interval = 0.0
    intervals = 0
    for row in csv.DictReader((args.input / "paired_decomposition.csv").open()):
        values = groups[
            tuple(row[k] for k in ("world", "candidate", "loss", "component"))
        ]
        rng = np.random.default_rng(20260915)
        replays = []
        for _, group in sorted(values.items()):
            group = np.asarray(group)
            indices = rng.integers(0, len(group), size=(2000, len(group)))
            weights = np.array(
                [np.bincount(x, minlength=len(group)) for x in indices]
            ) / len(group)
            replays.append(weights @ group)
        low, high = np.quantile(np.mean(replays, axis=0), [0.025, 0.975])
        maximum_delta = max(
            abs(low - float(row["ci_low"])), abs(high - float(row["ci_high"]))
        )
        max_interval = max(max_interval, maximum_delta)
        if maximum_delta > 1e-12:
            raise ValueError("interval replay")
        intervals += 1
    args.output.mkdir(parents=True, exist_ok=False)
    result = {
        "status": "FACTORIAL_ENUMERATED_REPLAY_PASS",
        "predictions": n,
        "max_score_difference": maximum,
        "author_metrics": metric_count,
        "decomposition_rows": decomposition_count,
        "intervals": intervals,
        "max_interval_difference": max_interval,
        "scope": "same-agent alternate equations;no human causal claim",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
