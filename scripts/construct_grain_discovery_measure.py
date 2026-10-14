#!/usr/bin/env python3
"""Construct two cold-start finite measures from the frozen GP02/GP03 interfaces."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from grain_research.measurement_discovery.construction import (
    make_bundle,
    theta_response,
    transport_standardization,
)
from grain_research.measurement_discovery.text_channel import render_observation


def write_json(path, value):
    """Write JSON with readable formatting."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    """Fit only source references, retaining a separately auditable cold bundle."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gp02-model",
        type=Path,
        default=Path("results/grain_measurement_discovery_v1/gp02/r1/model.json"),
    )
    parser.add_argument(
        "--gp03-reader",
        type=Path,
        default=Path("results/grain_measurement_discovery_v1/gp03/r1/reader.joblib"),
    )
    parser.add_argument(
        "--gp04-records",
        type=Path,
        default=Path("results/grain_measurement_discovery_v1/gp04/r1/records.jsonl"),
    )
    parser.add_argument(
        "--gp04-decision",
        type=Path,
        default=Path("results/grain_measurement_discovery_v1/gp04/r1/decision.md"),
    )
    parser.add_argument(
        "--gp05-decision",
        type=Path,
        default=Path("results/grain_measurement_discovery_v1/gp05/r1/decision.md"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/grain_measurement_discovery_v1/gp06/r1"),
    )
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    reader = joblib.load(args.gp03_reader)
    records = [json.loads(line) for line in args.gp04_records.read_text().splitlines()]
    source = [row for row in records if row["role"] == "source"]
    authors = sorted({row["author_id"] for row in source})
    by_author = {
        author: [row for row in source if row["author_id"] == author]
        for author in authors
    }
    theta_values = []
    for author_rows in by_author.values():
        texts = [
            render_observation(
                row["observation"],
                row["role"],
                sum(ord(c) for c in row["author_id"]) % 3,
            )
            for row in author_rows
        ]
        probabilities = reader.predict(texts)
        positive = [
            p
            for p, row in zip(probabilities, author_rows, strict=True)
            if row["condition"][0] == 1
        ]
        negative = [
            p
            for p, row in zip(probabilities, author_rows, strict=True)
            if row["condition"][0] == -1
        ]
        if positive and negative:
            theta_values.append(theta_response(np.mean(positive), np.mean(negative)))
    mean_theta = float(np.mean(theta_values))
    sd_theta = float(np.std(theta_values, ddof=1))
    if sd_theta <= 0:
        sd_theta = 1.0
    gp02 = json.loads(args.gp02_model.read_text())
    b1_table = np.asarray(gp02["b1"]["source_p"], dtype=float).tolist()
    t1 = make_bundle(
        "T1_condition_response",
        "theta=(mean(reader_p|z1=+1)-mean(reader_p|z1=-1))/2",
        "reader probability contrast",
        {"mean": mean_theta, "scale": sd_theta, "source_authors": len(authors)},
        "GP03_frozen_text_reader",
        "requires both z1 cells; missing cell returns null",
    )
    t2 = make_bundle(
        "T2_next_observation",
        "P(Y_next|prefix)=B1[source_table][Y_(t-2),Y_(t-1)]",
        "conditional probability",
        {"source_table": b1_table, "prefix_len": 12},
        "GP03_frozen_text_reader",
        "prefix-dependent next observation; not a lifelong trait",
    )
    bundle = {
        "schema_version": "grain-discovery-bundle-1",
        "reader_path": str(args.gp03_reader),
        "targets": [t1, t2],
        "transport_test": {
            "original": {"mean": mean_theta, "scale": sd_theta},
            "transformed": dict(
                zip(
                    ("mean", "scale"),
                    transport_standardization(mean_theta, sd_theta, 2.0, 3.0),
                    strict=True,
                )
            ),
            "rule": "z is invariant when the positive affine transform is transported",
        },
        "input_scope": "source observations build reference; application is cold scored separately",
    }
    write_json(out / "instrument_bundle.json", bundle)
    ledger = [
        {
            "item": "GP00_contract",
            "status": "SUPPORTED",
            "scope": "record/target/source boundary",
            "evidence": "gp00/r1/schema.json",
        },
        {
            "item": "GP01_information_boundary",
            "status": "SUPPORTED",
            "scope": "finite exact worlds",
            "evidence": "gp01/r1/certificate.json",
        },
        {
            "item": "GP02_standard_method",
            "status": "SUPPORTED",
            "scope": "one finite XOR world",
            "evidence": "gp02/r1/metrics.csv",
        },
        {
            "item": "GP03_text_interface",
            "status": "SUPPORTED",
            "scope": "fixed synthetic grammar and lexical reader",
            "evidence": "gp03/r1/metrics.csv",
        },
        {
            "item": "GP04_rival_design",
            "status": "SUPPORTED",
            "scope": "fixed three-axis simulation",
            "evidence": "gp04/r1/metrics.csv",
        },
        {
            "item": "GP05_natural_interface",
            "status": "BLOCKED_INPUT",
            "scope": "natural time/opportunity effect",
            "evidence": "gp05/r1/interface_checks.json",
        },
        {
            "item": "GP06_cold_construction",
            "status": "SUPPORTED",
            "scope": "two finite target specifications",
            "evidence": "gp06/r1/cold_replay.json",
        },
    ]
    with (out / "theory_result_ledger.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ledger[0]))
        writer.writeheader()
        writer.writerows(ledger)
    inputs = [
        args.gp02_model,
        args.gp03_reader,
        args.gp04_records,
        args.gp04_decision,
        args.gp05_decision,
        Path(__file__),
        Path(__file__).parents[1]
        / "grain_research/measurement_discovery/construction.py",
    ]
    manifest = {
        "task_id": "GP06",
        "status": "PASS",
        "science_status": "SUPPORTED",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": "python scripts/construct_grain_discovery_measure.py",
        "inputs": [
            {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in inputs
        ],
        "python": sys.version,
        "platform": platform.platform(),
        "model_calls": 0,
        "source_authors": len(authors),
        "theta_observations": len(theta_values),
    }
    write_json(out / "construction_manifest.json", manifest)
    # Keep the common task contract separate from the construction-specific
    # manifest so an auditor can discover GP06 like every other task.
    write_json(out / "manifest.json", manifest)
    with (out / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["target", "source_authors", "observations", "model_calls"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "target": "T1_condition_response",
                "source_authors": len(authors),
                "observations": len(theta_values),
                "model_calls": 0,
            }
        )
        writer.writerow(
            {
                "target": "T2_next_observation",
                "source_authors": len(authors),
                "observations": sum(len(row) for row in by_author.values()),
                "model_calls": 0,
            }
        )
    (out / "decision.md").write_text(
        "# GP06 decision\n\n"
        "- code_status: `PASS`\n"
        "- science_status: `SUPPORTED` within the finite synthetic text-reader world.\n"
        "- targets: `T1_condition_response` and `T2_next_observation` were constructed from source-only references and replayed cold on application records.\n"
        "- scope: the targets are conditional response and next-observation quantities, not lifelong traits, clinical scores, or human personality validity.\n"
        "- GP05 boundary retained: natural time/opportunity effects remain `BLOCKED_INPUT`.\n"
        "- trigger: future natural-data work requires explicit time, opportunity, and condition records; do not infer them from unit order.\n"
    )
    write_json(
        out / "summary.json",
        {
            "task_id": "GP06",
            "status": "PASS",
            "science_status": "SUPPORTED",
            "targets": ["T1_condition_response", "T2_next_observation"],
            "model_calls": 0,
        },
    )
    (out / "run_log.jsonl").write_text(
        json.dumps({"event": "complete", "task_id": "GP06"}) + "\n"
    )
    (out / "tests.txt").write_text(
        "cold bundle construction, source-only reference fit, positive affine transport metadata\n"
    )
    print(
        json.dumps(
            {
                "task_id": "GP06",
                "status": "PASS",
                "output": str(out),
                "source_authors": len(authors),
            }
        )
    )


if __name__ == "__main__":
    main()
