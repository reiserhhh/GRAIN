#!/usr/bin/env python3
"""Test background/layout-conditioned adjoint actions from fixed cached evidence."""

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

from grain_research.vector_mechanisms.centered_adjoint import (
    body_partition,
    center_groups,
)
from grain_research.vector_mechanisms.conditioned_action import (
    CONDITIONS,
    fit_conditioned_fields,
    token_readouts,
)
from grain_research.vector_mechanisms.level_response import paired_quality
from scripts.analyze_grain_author_discovery import scalar_metrics
from scripts.run_grain_centered_adjoint_field import CELLS, SOURCE, verify
from scripts.run_grain_real_author_path import BASE, read, sha, write
from scripts.run_grain_source_remainder import covariance

FIELD = BASE / "centered_adjoint_field_v1"
PARENT = BASE / "adjoint_action_space_v1"
PLAN = ROOT / "docs/GRAIN_CONDITIONED_ACTION_FIELD_PLAN_V1.md"
METHODS = ("base", *CONDITIONS)


def run(out):
    """Construct all predeclared conditional fields before accessing source outcomes."""
    inherited = verify(FIELD)
    if read(PARENT / "summary.json")["status"] != "COMPLETE":
        raise ValueError("M17 must be complete")
    design = read(FIELD / "private_design.json")
    out.mkdir(parents=True, exist_ok=False)
    (out / "private_fields").mkdir()
    inputs = [
        FIELD / "private_design.json",
        FIELD / "manifest.json",
        PARENT / "manifest.json",
        PARENT / "private_predictions_before_outcomes.csv",
    ]
    for query in ("TF_cont", "neuroticism"):
        inputs.extend(
            [
                FIELD / "private_fields" / f"{query}.npz",
                PARENT / "private_fields" / f"{query}.npz",
            ]
        )
    for row in design:
        if row["gradient_check"]:
            p = FIELD / "private_packets" / f"author_{row['index']:03d}.json"
            inputs.extend([p, p.with_suffix(".npz")])
    manifest = {
        "protocol": "GRAIN_CONDITIONED_ACTION_FIELD_V1",
        "code_hashes": {
            str(p.relative_to(ROOT)): sha(p)
            for p in (
                Path(__file__),
                PLAN,
                ROOT / "grain_research/vector_mechanisms/conditioned_action.py",
            )
        },
        "inputs": {str(p.relative_to(ROOT)): sha(p) for p in inputs},
        "source_outcome_sha256": inherited["source_outcome_sha256"],
        "max_seconds": 600,
        "alpha": 0.01,
        "human_labels_in_fit": False,
    }
    write(out / "manifest.json", manifest)
    start = time.monotonic()
    write(out / "run_status.json", {"status": "RUNNING", "neural_calls": 0})
    pred, diagnostics, families = [], [], []
    old = pd.read_csv(
        PARENT / "private_predictions_before_outcomes.csv", dtype={"author_id": str}
    ).set_index(["author_id", "query", "view"])
    if (
        sha(PARENT / "private_predictions_before_outcomes.csv")
        != read(PARENT / "predictions_frozen.json")["sha256"]
    ):
        raise ValueError("M17 frozen scores changed")
    replay = 0.0
    try:
        for query in ("TF_cont", "neuroticism"):
            with np.load(
                PARENT / "private_fields" / f"{query}.npz", allow_pickle=False
            ) as a:
                base = dict(a)
            with np.load(
                FIELD / "private_fields" / f"{query}.npz", allow_pickle=False
            ) as a:
                projection = a["projection"][:, :8]
            packets = []
            for row in design:
                if row["query"] == query and row["gradient_check"]:
                    p = FIELD / "private_packets" / f"author_{row['index']:03d}.json"
                    meta = read(p)
                    if (
                        meta["role"] != row["role"]
                        or meta["author_id"] != row["author_id"]
                        or sha(p.with_suffix(".npz")) != meta["arrays_sha256"]
                    ):
                        raise ValueError("Reference/check identity changed")
                    with np.load(p.with_suffix(".npz"), allow_pickle=False) as a:
                        packets.append({**meta, **dict(a)})
            if len(packets) != 8 or sum(p["role"] == "reference" for p in packets) != 4:
                raise ValueError("Exactly four reference/check authors required")
            model = fit_conditioned_fields(packets, base, projection, manifest["alpha"])
            np.savez_compressed(out / "private_fields" / f"{query}.npz", **model)
            for method in CONDITIONS:
                coef = model[f"{method}_coef"]
                families.append(
                    {
                        "query": query,
                        "method": method,
                        "coefficients": coef.size,
                        "background_dimensions": 8 if method != "layout" else 0,
                        "layout_dimensions": 4 if method != "background" else 0,
                        "coefficient_norm": float(np.linalg.norm(coef)),
                        "condition_coefficient_norm": float(
                            np.linalg.norm(coef[:, 1:])
                        ),
                    }
                )
            for p in packets:
                d = center_groups(p["displacement"], p["labels"])
                action = np.sum(
                    d * (len(d) * center_groups(p["integrated"], p["labels"])), axis=1
                )
                for method, values in token_readouts(
                    p["body"], p["labels"], model
                ).items():
                    diagnostics.append(
                        {
                            "query": query,
                            "role": p["role"],
                            "method": method,
                            "token_action_mse": float(np.mean((values - action) ** 2)),
                            "integral_error": float(values.mean() - action.mean()),
                            "endpoint_error": float(values.mean() - p["actual"]),
                            "quadrature_error": float(action.mean() - p["actual"]),
                        }
                    )
            for row in (r for r in design if r["query"] == query):
                if time.monotonic() - start > manifest["max_seconds"]:
                    raise TimeoutError(
                        "Cached field budget reached; partial fields retained"
                    )
                p = SOURCE / "private_packets" / f"author_{row['index']:03d}.npz"
                cell = CELLS / "private_packets" / p.with_suffix(".json").name
                for source in (p, cell):
                    if (
                        sha(source)
                        != inherited["sources"][str(source.relative_to(ROOT))]
                    ):
                        raise ValueError("Source state or endpoint changed")
                cells = read(cell)["views"]
                with np.load(p, allow_pickle=False) as a:
                    for view in row["views"]:
                        i = view["view"]
                        _, labels, body, _ = body_partition(
                            a[f"view_{i}"], view["groups"]
                        )
                        scores = {
                            k: float(v.mean())
                            for k, v in token_readouts(body, labels, model).items()
                        }
                        previous = old.loc[(row["author_id"], query, i), "action_refit"]
                        replay = max(replay, abs(scores["base"] - previous))
                        pred.append(
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
                    "views": len(pred),
                    "seconds": time.monotonic() - start,
                },
                flush=True,
            )
        if len(pred) != 144 or replay > 1e-9:
            raise ValueError("M17 base replay or scoring completeness failed")
        frame = pd.DataFrame(pred)
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
            {"transport": metrics, "families": families, "base_replay_max": replay},
        )
        write(
            out / "run_status.json",
            {
                "status": "COMPLETE",
                "neural_calls": 0,
                "seconds": time.monotonic() - start,
                "author_materials": 144,
                "reference_authors": 8,
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
                "seconds": time.monotonic() - start,
                "neural_calls": 0,
            },
        )
        write(out / "error_log.json", {"type": type(exc).__name__, "message": str(exc)})
        raise


def evaluate(out):
    """Connect frozen conditional functions to the same exposed outcomes, without selection."""
    m = read(out / "manifest.json")
    for p, digest in m["code_hashes"].items():
        if sha(ROOT / p) != digest:
            raise ValueError("Frozen construction changed")
    path = out / "private_predictions_before_outcomes.csv"
    source_path = SOURCE / "private_anchor_source_connection.csv"
    if (
        read(out / "run_status.json")["status"] != "COMPLETE"
        or sha(path) != read(out / "predictions_frozen.json")["sha256"]
        or sha(source_path) != m["source_outcome_sha256"]
    ):
        raise ValueError("Frozen prediction/source linkage failed")
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
        raise ValueError("Exactly the original48 authors required")
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
    """Run the fixed cached comparison and then its separate source connection phase."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("run", "evaluate"))
    parser.add_argument(
        "--output-dir", type=Path, default=BASE / "conditioned_action_field_v1"
    )
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(BASE.resolve()):
        raise ValueError("Private results must remain ignored")
    with threadpool_limits(limits=2):
        {"run": run, "evaluate": evaluate}[args.phase](out)


if __name__ == "__main__":
    main()
