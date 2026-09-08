#!/usr/bin/env python3
"""Bounded cached endpoint-constraint comparison on the frozen M20 design."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from grain_research.vector_mechanisms.centered_adjoint import body_partition
from grain_research.vector_mechanisms.endpoint_constraint import constrain_action
from grain_research.vector_mechanisms.level_response import paired_quality
from grain_research.vector_mechanisms.odd_action import (
    fit_odd_fields,
    odd_token_readouts,
)
from scripts.analyze_grain_author_discovery import scalar_metrics
from scripts.run_grain_odd_action import (
    FIELD,
    METHODS,
    PARITY,
    SOURCE,
    verify,
    verify_plan,
)
from scripts.run_grain_real_author_path import BASE, read, sha, write
from scripts.run_grain_source_remainder import covariance

ODD = BASE / "odd_action_construction_v1"
PLAN = ROOT / "docs/GRAIN_ENDPOINT_ACTION_CONSTRAINT_PLAN_V1.md"


def run(out):
    """Constrain reference teachers only; freeze every score before source-label joins."""
    m, parent = verify_plan(ODD), verify(FIELD)
    if read(ODD / "summary.json")["status"] != "COMPLETE":
        raise ValueError("Completed M20 required")
    out.mkdir(parents=True, exist_ok=False)
    (out / "private_fields").mkdir()
    paths = [
        PLAN,
        Path(__file__),
        ROOT / "grain_research/vector_mechanisms/endpoint_constraint.py",
    ]
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_ENDPOINT_ACTION_CONSTRAINT_V1",
            "neural_calls": 0,
            "parent_manifest_sha256": sha(ODD / "manifest.json"),
            "parent_summary_sha256": sha(ODD / "summary.json"),
            "source_outcome_sha256": m["source_outcome_sha256"],
            "code_hashes": {str(p.relative_to(ROOT)): sha(p) for p in paths},
            "field_alpha": m["field_alpha"],
            "action_alpha": m["action_alpha"],
            "selection_by_source_labels": False,
            "max_seconds": 120,
        },
    )
    start = time.monotonic()
    design = read(FIELD / "private_design.json")
    scores, diagnostics, constraints = [], [], []
    max_parity = 0.0
    for query in ("TF_cont", "neuroticism"):
        packets = []
        for row in (r for r in design if r["query"] == query and r["gradient_check"]):
            path = ODD / "private_packets" / f"author_{row['index']:03d}.json"
            meta = read(path)
            if (
                meta["author_id"] != row["author_id"]
                or sha(path.with_suffix(".npz")) != meta["arrays_sha256"]
            ):
                raise ValueError("Frozen paired packet changed")
            with np.load(path.with_suffix(".npz"), allow_pickle=False) as a:
                packet = {**meta, **dict(a)}
            if row["role"] == "reference":
                corrected, info = constrain_action(
                    packet["displacement"],
                    packet["integrated_even"],
                    packet["labels"],
                    packet["actual_odd"],
                )
                packet["integrated_even"] = corrected
                constraints.append({"query": query, **info})
            packets.append(packet)
        if len(packets) != 8 or sum(p["role"] == "reference" for p in packets) != 4:
            raise ValueError("Exact four reference/four check bank required")
        with np.load(
            FIELD / "private_fields" / f"{query}.npz", allow_pickle=False
        ) as a:
            model = fit_odd_fields(
                packets,
                a["projection"],
                a["scale"],
                m["field_alpha"],
                m["action_alpha"],
            )
        np.savez_compressed(out / "private_fields" / f"{query}.npz", **model)
        for packet in packets:
            for method, values in odd_token_readouts(
                packet["displacement"], packet["labels"], model
            ).items():
                diagnostics.append(
                    {
                        "query": query,
                        "role": packet["role"],
                        "method": method,
                        "endpoint_error": float(values.mean() - packet["actual_odd"]),
                    }
                )
        for row in (r for r in design if r["query"] == query):
            path = SOURCE / "private_packets" / f"author_{row['index']:03d}.npz"
            if sha(path) != parent["sources"][str(path.relative_to(ROOT))]:
                raise ValueError("Frozen scoring state changed")
            endpoints = read(
                PARITY / "private_packets" / path.with_suffix(".json").name
            )["views"]
            with np.load(path, allow_pickle=False) as a:
                for view in row["views"]:
                    i = view["view"]
                    _, labels, _, d = body_partition(a[f"view_{i}"], view["groups"])
                    positive, negative = (
                        odd_token_readouts(d, labels, model),
                        odd_token_readouts(-d, labels, model),
                    )
                    max_parity = max(
                        max_parity,
                        max(
                            float(np.max(np.abs(positive[k] + negative[k])))
                            for k in METHODS
                        ),
                    )
                    scores.append(
                        {
                            "author_id": row["author_id"],
                            "query": query,
                            "role": row["role"],
                            "view": i,
                            "actual_odd": endpoints[i]["odd"],
                            **{k: float(v.mean()) for k, v in positive.items()},
                        }
                    )
        if time.monotonic() - start > 120:
            raise TimeoutError("Bounded cached comparison exceeded 120 seconds")
    if len(scores) != 144 or max_parity > 1e-9:
        raise ValueError("Complete odd prediction panel required")
    frame = pd.DataFrame(scores)
    path = out / "private_predictions_before_outcomes.csv"
    frame.to_csv(path, index=False)
    write(
        out / "predictions_frozen.json",
        {"sha256": sha(path), "human_labels_read": False},
    )
    pd.DataFrame(diagnostics).to_csv(out / "field_diagnostics.csv", index=False)
    write(out / "constraint_diagnostics.json", constraints)
    means = frame.groupby(["author_id", "query", "role"], as_index=False).mean(
        numeric_only=True
    )
    metrics = []
    for (query, role), group in means.groupby(["query", "role"]):
        for method in METHODS:
            metrics.append(
                {
                    "query": query,
                    "role": role,
                    "method": method,
                    **paired_quality(
                        group.actual_odd.to_numpy(),
                        group[method].to_numpy(),
                        np.zeros(len(group)),
                    ),
                }
            )
    pd.DataFrame(metrics).to_csv(out / "transport_metrics.csv", index=False)
    source = SOURCE / "private_anchor_source_connection.csv"
    if (
        sha(source) != m["source_outcome_sha256"]
        or sha(path) != read(out / "predictions_frozen.json")["sha256"]
    ):
        raise ValueError("Frozen source or predictions changed")
    target = pd.read_csv(source, dtype={"author_id": str})
    means = means.merge(
        target[["author_id", "query", "y_true"]],
        on=["author_id", "query"],
        validate="one_to_one",
    )
    if len(means) != 48:
        raise ValueError("Complete48-author source linkage required")
    source_metrics = []
    for (query, role), group in means.groupby(["query", "role"]):
        y = group.y_true.to_numpy()
        actual_cov = covariance(y, group.actual_odd.to_numpy())
        for method in METHODS:
            p = group[method].to_numpy()
            cov = covariance(y, p)
            remainder = covariance(y, group.actual_odd.to_numpy() - p)
            if abs(cov + remainder - actual_cov) > 1e-12:
                raise ValueError("Source covariance account failed")
            source_metrics.append(
                {
                    "query": query,
                    "role": role,
                    "method": method,
                    **scalar_metrics(y, p),
                    "cov_y": cov,
                    "odd_remainder_cov_y": remainder,
                }
            )
    means.to_csv(out / "private_source_connection.csv", index=False)
    pd.DataFrame(source_metrics).to_csv(out / "source_metrics.csv", index=False)
    write(
        out / "summary.json",
        {
            "status": "COMPLETE",
            "neural_calls": 0,
            "seconds": time.monotonic() - start,
            "parity_error_max": max_parity,
            "constraint_error_max": max(abs(c["gap_after"]) for c in constraints),
            "transport": metrics,
            "source": source_metrics,
            "scope": "minimum-change teacher representative; existing discovery panel, not true gradient recovery",
        },
    )
    print(
        pd.DataFrame(metrics)
        .query("role == 'evaluation'")[["query", "method", "r", "mse_ratio"]]
        .to_string(index=False)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=BASE / "endpoint_action_constraint_v1"
    )
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args.output)
