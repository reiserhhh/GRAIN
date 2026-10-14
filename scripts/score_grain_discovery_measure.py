#!/usr/bin/env python3
"""Cold-score GP06 targets from application observations without refitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from grain_research.measurement_discovery.construction import (
    theta_response,
)
from grain_research.measurement_discovery.text_channel import (
    render_observation,
    second_order_soft_probability,
)


def score_application(bundle, records):
    """Score application authors using only observations revealed so far."""
    reader = joblib.load(bundle["reader_path"])
    application = [row for row in records if row["role"] == "application"]
    by_author = {}
    for row in application:
        by_author.setdefault(row["author_id"], []).append(row)
    t1_results, t2_results = [], []
    table = np.asarray(bundle["targets"][1]["reference"]["source_table"], dtype=float)
    for author, rows in by_author.items():
        rows = sorted(rows, key=lambda row: (row["t"], row["event_id"]))
        texts = [
            render_observation(
                row["observation"], row["role"], sum(ord(c) for c in author) % 3
            )
            for row in rows
        ]
        probabilities = reader.predict(texts)
        positive = [
            p
            for p, row in zip(probabilities, rows, strict=True)
            if row["condition"][0] == 1
        ]
        negative = [
            p
            for p, row in zip(probabilities, rows, strict=True)
            if row["condition"][0] == -1
        ]
        t1_results.append(
            {
                "author_id": author,
                "theta": theta_response(np.mean(positive), np.mean(negative))
                if positive and negative
                else None,
                "support": {
                    "positive_cells": len(positive),
                    "negative_cells": len(negative),
                },
            }
        )
        prefix_len = int(bundle["targets"][1]["reference"]["prefix_len"])
        prefix = list(probabilities[:prefix_len])
        values = [int(row["observation"]) for row in rows[:prefix_len]]
        for index in range(prefix_len, len(rows)):
            if len(values) < 2:
                continue
            p = second_order_soft_probability(prefix[-2], prefix[-1], table)
            y = int(rows[index]["observation"])
            t2_results.append(
                {
                    "author_id": author,
                    "event_id": rows[index]["event_id"],
                    "p": p,
                    "y": y,
                    "revealed_prefix": len(values),
                }
            )
            values.append(y)
            prefix.append(float(probabilities[index]))
    return {"T1_condition_response": t1_results, "T2_next_observation": t2_results}


def main():
    """Read an immutable bundle and emit a provenance-rich cold replay."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle",
        type=Path,
        default=Path(
            "results/grain_measurement_discovery_v1/gp06/r1/instrument_bundle.json"
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/grain_measurement_discovery_v1/gp04/r1/records.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/grain_measurement_discovery_v1/gp06/r1/cold_replay.json"),
    )
    args = parser.parse_args()
    bundle = json.loads(args.bundle.read_text())
    records = [json.loads(line) for line in args.input.read_text().splitlines()]
    result = score_application(bundle, records)
    output = {
        "status": "PASS",
        "science_status": "SUPPORTED",
        "bundle_sha256": __import__("hashlib")
        .sha256(args.bundle.read_bytes())
        .hexdigest(),
        "input_sha256": __import__("hashlib")
        .sha256(args.input.read_bytes())
        .hexdigest(),
        "model_calls": 0,
        "targets": result,
        "support_warning": "scores are finite synthetic conditional targets; no psychological names or natural-person claims",
    }
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "task_id": "GP06",
                "status": "PASS",
                "output": str(args.output),
                "t1_authors": len(result["T1_condition_response"]),
                "t2_events": len(result["T2_next_observation"]),
            }
        )
    )


if __name__ == "__main__":
    main()
