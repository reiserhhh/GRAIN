#!/usr/bin/env python3
"""Replay proper losses, histories' evidence mass and paired channel contrasts."""

import argparse
import csv
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.gptpro_foundation_v3.f02_recovery import paired_bootstrap


def main():
    """Check every prediction and aggregate by authors rather than events."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    state = json.loads((args.input / "summary.json").read_text())
    if state["status"] != "COMPUTED_AUDIT_PENDING":
        raise ValueError("completed run required")
    masses = {}
    running = defaultdict(float)
    with gzip.open(args.input / "observations.jsonl.gz", "rt") as stream:
        for line in stream:
            r = json.loads(line)
            for channel in ("oracle", "soft", "hard"):
                key = (r["author_id"], r["panel"], channel)
                masses[(r["event_id"], r["panel"], channel)] = running[key]
                q = (
                    1
                    if r["panel"] == "complete"
                    else r["qualification"]
                    if channel == "oracle"
                    else r["qualification_score"]
                    if channel == "soft"
                    else int(r["qualification_score"] >= 0.5)
                )
                running[key] += q
    sums = defaultdict(lambda: np.zeros(3))
    oracle = {}
    channel_max = 0.0
    max_loss = 0.0
    count = 0
    with gzip.open(args.input / "predictions.jsonl.gz", "rt") as stream:
        for line in stream:
            r = json.loads(line)
            p = r["p"]
            y = r["y"]
            if not 1e-6 <= p <= 1 - 1e-6:
                raise ValueError("invalid predictive probability")
            np.testing.assert_allclose(
                r["evidence_mass"],
                masses[(r["event_id"], r["panel"], r["channel"])],
                atol=1e-12,
            )
            loss = (p - y) ** 2
            logloss = -np.log(p if y else 1 - p)
            max_loss = max(
                max_loss, abs(loss - r["brier"]), abs(logloss - r["log_loss"])
            )
            key = tuple(
                r[k]
                for k in (
                    "world",
                    "seed",
                    "author_id",
                    "role",
                    "panel",
                    "channel",
                    "candidate",
                )
            )
            sums[key] += np.array([loss, logloss, 1])
            event = (r["event_id"], r["panel"], r["candidate"])
            if r["channel"] == "oracle":
                oracle[event] = p
            if r["channel"] == "hard":
                channel_max = max(channel_max, abs(p - oracle.pop(event)))
            count += 1
    table = list(csv.DictReader((args.input / "metrics_by_author.csv").open()))
    metrics = {}
    for row in table:
        row["seed"] = int(row["seed"])
        key = tuple(
            row[k]
            for k in (
                "world",
                "seed",
                "author_id",
                "role",
                "panel",
                "channel",
                "candidate",
            )
        )
        values = sums[key]
        np.testing.assert_allclose(
            values[:2] / values[2],
            [float(row["brier"]), float(row["log_loss"])],
            atol=1e-12,
        )
        if values[2] != int(row["events"]):
            raise ValueError("event denominator")
        metrics[key] = values[:2] / values[2]
    contrasts = []
    for world in ("c2_xor", "c2_iid"):
        for panel in ("complete", "mixed"):
            for candidate in ("b1", "b2", "m", "source_mean", "personal_mean"):
                for channel in ("soft", "hard"):
                    for index, loss_name in enumerate(("brier", "log_loss")):
                        by_seed = defaultdict(list)
                        for key, value in metrics.items():
                            w, seed, author, role, p, c, k = key
                            if (w, role, p, c, k) == (
                                world,
                                "application",
                                panel,
                                channel,
                                candidate,
                            ):
                                reference = metrics[
                                    (w, seed, author, role, p, "oracle", k)
                                ]
                                by_seed[seed].append(value[index] - reference[index])
                        contrasts.append(
                            {
                                "world": world,
                                "panel": panel,
                                "candidate": candidate,
                                "channel": channel,
                                "reference": "oracle",
                                "loss": loss_name,
                                **paired_bootstrap(by_seed),
                            }
                        )
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "paired_channel_contrasts.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(contrasts[0]))
        writer.writeheader()
        writer.writerows(contrasts)
    result = {
        "status": "LOSSES_EVIDENCE_MASS_AUTHOR_AGGREGATES_REPLAY_PASS",
        "prediction_rows": count,
        "author_metric_rows": len(metrics),
        "paired_contrasts": len(contrasts),
        "max_loss_difference": max_loss,
        "max_hard_oracle_difference": channel_max,
        "scope": "all proper losses and causal evidence masses; full independent soft state/refit audit remains separate",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
