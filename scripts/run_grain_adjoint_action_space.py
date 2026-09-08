#!/usr/bin/env python3
"""Compare two reference-only norms in the same cached conditional-field basis."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from grain_research.vector_mechanisms.action_space import (
    action_diagnostics,
    fit_action_dictionary,
    read_action_dictionary,
    spectral_dictionary,
)
from grain_research.vector_mechanisms.centered_adjoint import body_partition
from grain_research.vector_mechanisms.level_response import paired_quality
from scripts.analyze_grain_author_discovery import scalar_metrics
from scripts.run_grain_centered_adjoint_field import CELLS, SOURCE, verify
from scripts.run_grain_real_author_path import BASE, read, sha, write
from scripts.run_grain_source_remainder import covariance

PARENT = BASE / "centered_adjoint_field_v1"
PLAN = ROOT / "docs/GRAIN_ADJOINT_ACTION_SPACE_PLAN_V1.md"
METHODS = ("original", "gradient_refit", "action_refit")


def run(out):
    """Freeze input hashes, fit without outcomes, then score cached transfer materials."""
    inherited = verify(PARENT)
    if read(PARENT / "summary.json")["status"] != "COMPLETE":
        raise ValueError("Completed M16 required")
    inputs = [
        PARENT / "private_design.json",
        PARENT / "private_predictions_before_outcomes.csv",
    ]
    for query in ("TF_cont", "neuroticism"):
        inputs.extend(
            [
                PARENT / "private_fields" / f"{query}.npz",
                PARENT / "private_fields" / f"{query}.json",
            ]
        )
    design = read(PARENT / "private_design.json")
    for row in design:
        if row["gradient_check"]:
            f = PARENT / "private_packets" / f"author_{row['index']:03d}.json"
            inputs.extend([f, f.with_suffix(".npz")])
    out.mkdir(parents=True, exist_ok=False)
    (out / "private_fields").mkdir()
    manifest = {
        "protocol": "GRAIN_ADJOINT_ACTION_SPACE_V1",
        "code_hashes": {
            str(p.relative_to(ROOT)): sha(p)
            for p in (
                Path(__file__),
                PLAN,
                ROOT / "grain_research/vector_mechanisms/action_space.py",
            )
        },
        "inputs": {str(p.relative_to(ROOT)): sha(p) for p in inputs},
        "source_outcome_sha256": inherited["source_outcome_sha256"],
        "relative_penalty": 0.01,
        "max_seconds": 600,
        "human_outcomes_in_fit": False,
        "new_neural_calls": 0,
    }
    write(out / "manifest.json", manifest)
    start = time.monotonic()
    write(out / "run_status.json", {"status": "RUNNING", "neural_calls": 0})
    predictions, diagnostics, families = [], [], []
    old = pd.read_csv(
        PARENT / "private_predictions_before_outcomes.csv", dtype={"author_id": str}
    )
    max_replay = 0.0
    try:
        for query in ("TF_cont", "neuroticism"):
            packets = []
            path = PARENT / "private_fields" / f"{query}.npz"
            if sha(path) != read(path.with_suffix(".json"))["arrays_sha256"]:
                raise ValueError("Reference field changed")
            with np.load(path, allow_pickle=False) as a:
                basis = spectral_dictionary(dict(a))
            for row in design:
                if row["query"] != query or not row["gradient_check"]:
                    continue
                p = PARENT / "private_packets" / f"author_{row['index']:03d}.json"
                meta = read(p)
                if (
                    meta["role"] != row["role"]
                    or meta["author_id"] != row["author_id"]
                    or sha(p.with_suffix(".npz")) != meta["arrays_sha256"]
                ):
                    raise ValueError("Adjoint packet identity changed")
                with np.load(p.with_suffix(".npz"), allow_pickle=False) as a:
                    packets.append({**meta, **dict(a)})
            if len(packets) != 8 or sum(p["role"] == "reference" for p in packets) != 4:
                raise ValueError("Exact same M16 reference/check bank required")
            model = fit_action_dictionary(packets, basis, manifest["relative_penalty"])
            np.savez_compressed(out / "private_fields" / f"{query}.npz", **model)
            families.append(
                {
                    "query": query,
                    "rank": len(model["original"]),
                    "positive": int((model["original"] > 0).sum()),
                    "negative": int((model["original"] < 0).sum()),
                    **{
                        k: float(v)
                        for k, v in model.items()
                        if k.endswith(("penalty", "condition"))
                    },
                }
            )
            for packet in packets:
                diagnostics.extend(action_diagnostics(packet, model))
            for row in (r for r in design if r["query"] == query):
                if time.monotonic() - start > manifest["max_seconds"]:
                    raise TimeoutError(
                        "Cached action-space budget reached; partial artifacts retained"
                    )
                path = SOURCE / "private_packets" / f"author_{row['index']:03d}.npz"
                cell = CELLS / "private_packets" / path.with_suffix(".json").name
                for p in (path, cell):
                    if sha(p) != inherited["sources"][str(p.relative_to(ROOT))]:
                        raise ValueError("Inherited source state or endpoint changed")
                cells = read(cell)["views"]
                with np.load(path, allow_pickle=False) as a:
                    for view in row["views"]:
                        i = view["view"]
                        _, labels, _, d = body_partition(a[f"view_{i}"], view["groups"])
                        scores = read_action_dictionary(d, labels, model)
                        match = old[
                            (old.author_id == row["author_id"])
                            & (old["query"] == query)
                            & (old["view"] == i)
                        ]
                        if len(match) != 1:
                            raise ValueError("Exact M16 matched material required")
                        max_replay = max(
                            max_replay,
                            abs(scores["original"] - match.iloc[0].rank64_field),
                        )
                        predictions.append(
                            {
                                "author_id": row["author_id"],
                                "query": query,
                                "role": row["role"],
                                "view": i,
                                "actual": cells[i]["f10"] - cells[i]["f00"],
                                **scores,
                            }
                        )
            print(
                {
                    "query": query,
                    "dictionary_rank": len(model["original"]),
                    "views": len(predictions),
                    "seconds": time.monotonic() - start,
                },
                flush=True,
            )
        if len(predictions) != 144 or max_replay > 1e-9:
            raise ValueError("M16 symmetric action replay or complete materials failed")
        frame = pd.DataFrame(predictions)
        frame.to_csv(out / "private_predictions_before_outcomes.csv", index=False)
        write(
            out / "predictions_frozen.json",
            {
                "sha256": sha(out / "private_predictions_before_outcomes.csv"),
                "human_labels_used": False,
            },
        )
        pd.DataFrame(diagnostics).to_csv(out / "gradient_diagnostics.csv", index=False)
        means = frame.groupby(["author_id", "query", "role"], as_index=False).mean(
            numeric_only=True
        )
        metrics = []
        for (query, role), g in means.groupby(["query", "role"]):
            for method in METHODS:
                metrics.append(
                    {
                        "query": query,
                        "role": role,
                        "method": method,
                        **paired_quality(
                            g.actual.to_numpy(), g[method].to_numpy(), np.zeros(len(g))
                        ),
                    }
                )
        pd.DataFrame(metrics).to_csv(out / "transport_metrics.csv", index=False)
        write(
            out / "fit_summary.json",
            {
                "families": families,
                "original_score_replay_max": max_replay,
                "transport": metrics,
            },
        )
        write(
            out / "run_status.json",
            {
                "status": "COMPLETE",
                "neural_calls": 0,
                "seconds": time.monotonic() - start,
                "author_materials": 144,
                "gradient_authors": 16,
            },
        )
        print(
            pd.DataFrame(metrics)[
                ["query", "role", "method", "r", "mse_ratio"]
            ].to_string(index=False)
        )
    except BaseException as exc:
        write(
            out / "run_status.json",
            {
                "status": "FAILED_RETAINED",
                "neural_calls": 0,
                "seconds": time.monotonic() - start,
            },
        )
        write(out / "error_log.json", {"type": type(exc).__name__, "message": str(exc)})
        raise


def evaluate(out):
    """Join only the same exposed source outcomes after field predictions freeze."""
    m = read(out / "manifest.json")
    for p, digest in m["code_hashes"].items():
        if sha(ROOT / p) != digest:
            raise ValueError("Frozen action-space procedure changed")
    if read(out / "run_status.json")["status"] != "COMPLETE":
        raise ValueError("Complete frozen transport required")
    path = out / "private_predictions_before_outcomes.csv"
    if sha(path) != read(out / "predictions_frozen.json")["sha256"]:
        raise ValueError("Frozen prediction file changed")
    source_path = SOURCE / "private_anchor_source_connection.csv"
    if sha(source_path) != m["source_outcome_sha256"]:
        raise ValueError("Source link changed")
    frame = (
        pd.read_csv(path, dtype={"author_id": str})
        .groupby(["author_id", "query", "role"], as_index=False)
        .mean(numeric_only=True)
    )
    source = pd.read_csv(source_path, dtype={"author_id": str})
    frame = frame.merge(
        source[["author_id", "query", "y_true", "native", "PQ"]],
        on=["author_id", "query"],
        validate="one_to_one",
    )
    if len(frame) != 48:
        raise ValueError("Same48-author linkage required")
    rows = []
    for (query, role), g in frame.groupby(["query", "role"]):
        for method in ("actual", *METHODS, "native", "PQ"):
            rows.append(
                {
                    "query": query,
                    "role": role,
                    "method": method,
                    **scalar_metrics(g.y_true.to_numpy(), g[method].to_numpy()),
                    "cov_y": covariance(g.y_true.to_numpy(), g[method].to_numpy()),
                }
            )
    frame.to_csv(out / "private_source_connection.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "source_metrics.csv", index=False)
    write(
        out / "summary.json",
        {
            "status": "COMPLETE",
            "cost": read(out / "run_status.json"),
            "fit": read(out / "fit_summary.json"),
            "source_metrics": rows,
        },
    )
    print(
        pd.DataFrame(rows)[["query", "role", "method", "pearson_r", "cov_y"]].to_string(
            index=False
        )
    )


def main():
    """Execute one frozen cached comparison, with outcome linkage as a later phase."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("run", "evaluate"))
    parser.add_argument(
        "--output-dir", type=Path, default=BASE / "adjoint_action_space_v1"
    )
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(BASE.resolve()):
        raise ValueError("Private results must remain ignored")
    with threadpool_limits(limits=2):
        {"run": run, "evaluate": evaluate}[args.phase](out)


if __name__ == "__main__":
    main()
