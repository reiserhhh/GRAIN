#!/usr/bin/env python3
"""Frozen-record 2x2 attribution of polarity and eligibility reading errors."""

import argparse
import csv
import gzip
import hashlib
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.gptpro_foundation_v3.f02 import generate_world
from grain_research.gptpro_foundation_v3.f02_recovery import paired_bootstrap
from grain_research.gptpro_foundation_v3.f08_estimators import QualifiedHistory
from grain_research.measurement_discovery.estimators import ordered_rows


def write_table(path, rows):
    """Write a generated evidence table."""
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    """Run only the missing two factorial cells; preserve the original endpoints."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((args.input / "protocol.json").read_text())
    plan = {
        "input": str(args.input),
        "source_only_refit": False,
        "new_neural_calls": 0,
        "design": "mixed panel only;observed_pole x observed_eligibility;two existing endpoints reused",
        "oracle_pole": "material_pole,not hidden personal response on ineligible text",
        "contrast": "L10-L00, L01-L00, L11-L10-L01+L00;paired application authors",
        "input_hashes": {
            name: hashlib.sha256((args.input / name).read_bytes()).hexdigest()
            for name in (
                "protocol.json",
                "metrics_by_author.csv",
                "observations.jsonl.gz",
            )
        },
    }
    (args.output / "protocol_before_run.json").write_text(
        json.dumps(plan, indent=2) + "\n"
    )
    bundles = {}
    for world in ("c2_xor", "c2_iid"):
        for seed in protocol["seeds"]:
            with (args.input / f"source_{world}_{seed}.pkl").open("rb") as stream:
                fitted = pickle.load(stream)
            records, _ = generate_world(world, seed, counts=protocol["counts"])
            source = [r for r in records if r["role"] == "source"]
            initial = np.ones((2, 2))
            for rows in ordered_rows(source).values():
                initial[rows[0]["observation"], rows[1]["observation"]] += 1
            initial /= initial.sum()
            mean = float(np.mean([r["observation"] for r in source]))
            bundles[(world, seed)] = (fitted, initial, mean)
    totals = defaultdict(lambda: np.zeros(3))
    previous = None
    steps = 0
    with (
        gzip.open(args.input / "observations.jsonl.gz", "rt") as stream,
        gzip.open(args.output / "predictions.jsonl.gz", "wt") as output,
    ):
        for line in stream:
            row = json.loads(line)
            if row["panel"] != "mixed":
                continue
            if row["author_id"] != previous:
                parts = row["event_id"].split("_")
                world = "_".join(parts[:2])
                seed = int(parts[2])
                fitted, initial, mean = bundles[(world, seed)]
                histories = {
                    name: QualifiedHistory(fitted, initial, mean)
                    for name in (
                        "reader_pole_oracle_qualification",
                        "oracle_pole_reader_qualification",
                    )
                }
                previous = row["author_id"]
            readings = {
                "reader_pole_oracle_qualification": (
                    row["pole_score"],
                    row["qualification"],
                ),
                "oracle_pole_reader_qualification": (
                    row["material_pole"],
                    row["qualification_score"],
                ),
            }
            for channel, history in histories.items():
                if row["t"] >= 12:
                    for candidate, p in history.predict()["probabilities"].items():
                        y = row["observation"]
                        brier = (p - y) ** 2
                        log_loss = -np.log(p if y else 1 - p)
                        key = (
                            world,
                            seed,
                            row["author_id"],
                            row["role"],
                            channel,
                            candidate,
                        )
                        totals[key] += np.array([brier, log_loss, 1])
                        output.write(
                            json.dumps(
                                {
                                    "world": world,
                                    "seed": seed,
                                    "author_id": row["author_id"],
                                    "role": row["role"],
                                    "event_id": row["event_id"],
                                    "t": row["t"],
                                    "channel": channel,
                                    "candidate": candidate,
                                    "p": p,
                                    "y": y,
                                    "brier": brier,
                                    "log_loss": log_loss,
                                }
                            )
                            + "\n"
                        )
                history.observe(*readings[channel])
                steps += 1
    metrics = [
        dict(
            zip(
                ("world", "seed", "author_id", "role", "channel", "candidate"),
                key,
                strict=True,
            ),
            brier=float(v[0] / v[2]),
            log_loss=float(v[1] / v[2]),
            events=int(v[2]),
        )
        for key, v in totals.items()
    ]
    write_table(args.output / "metrics_by_author.csv", metrics)
    cells = {}
    for row in csv.DictReader((args.input / "metrics_by_author.csv").open()):
        if row["panel"] == "mixed" and row["channel"] in ("oracle", "soft"):
            key = (
                row["world"],
                int(row["seed"]),
                row["author_id"],
                row["role"],
                row["candidate"],
            )
            cells.setdefault(key, {})["00" if row["channel"] == "oracle" else "11"] = [
                float(row["brier"]),
                float(row["log_loss"]),
            ]
    for row in metrics:
        key = (
            row["world"],
            row["seed"],
            row["author_id"],
            row["role"],
            row["candidate"],
        )
        cells[key][
            "10" if row["channel"] == "reader_pole_oracle_qualification" else "01"
        ] = [row["brier"], row["log_loss"]]
    decomposed = []
    groups = defaultdict(lambda: defaultdict(list))
    maximum = 0.0
    for key, cell in cells.items():
        if set(cell) != {"00", "10", "01", "11"}:
            raise ValueError("missing factorial cell")
        for index, loss in enumerate(("brier", "log_loss")):
            l00, l10, l01, l11 = [cell[k][index] for k in ("00", "10", "01", "11")]
            polarity = l10 - l00
            qualification = l01 - l00
            interaction = l11 - l10 - l01 + l00
            total = l11 - l00
            maximum = max(maximum, abs(total - polarity - qualification - interaction))
            decomposed.append(
                {
                    "world": key[0],
                    "seed": key[1],
                    "author_id": key[2],
                    "role": key[3],
                    "candidate": key[4],
                    "loss": loss,
                    "L00": l00,
                    "L10": l10,
                    "L01": l01,
                    "L11": l11,
                    "polarity": polarity,
                    "qualification": qualification,
                    "interaction": interaction,
                    "total": total,
                }
            )
            if key[3] == "application":
                for name, value in (
                    ("polarity", polarity),
                    ("qualification", qualification),
                    ("interaction", interaction),
                    ("total", total),
                ):
                    groups[(key[0], key[4], loss, name)][key[1]].append(value)
    write_table(args.output / "author_decomposition.csv", decomposed)
    contrasts = [
        {
            "world": key[0],
            "candidate": key[1],
            "loss": key[2],
            "component": key[3],
            **paired_bootstrap(values),
        }
        for key, values in groups.items()
    ]
    write_table(args.output / "paired_decomposition.csv", contrasts)
    result = {
        "status": "FACTORIAL_COMPUTED_AUDIT_PENDING",
        "updated_steps": steps,
        "new_author_metric_rows": len(metrics),
        "decomposition_rows": len(decomposed),
        "contrasts": len(contrasts),
        "max_decomposition_identity_error": maximum,
        "scope": "pipeline input interventions on fixed synthetic histories,not human causal mediation",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
