#!/usr/bin/env python3
"""Analyze the frozen NEXT30 packet by explicit design, excluding filesystem sidecars."""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.native_material_reader import cosine, direction
from scripts.run_grain_native_material_reader import QUERIES, cached_unit, verify
from scripts.run_grain_real_author_path import BASE, read, sha, write


def planned_packets(out, design, queries):
    """Enumerate only declared author/query units, never glob platform sidecars."""
    indices = [a["index"] for a in design]
    if len(set(indices)) != len(indices) or len(set(queries)) != len(queries):
        raise ValueError("Duplicate design coordinates")
    if any(not isinstance(i, int) or i < 0 for i in indices):
        raise ValueError("Nonnegative integer indices required")
    return [
        out / "private_packets" / f"a{i:02d}_{q}.json" for i in indices for q in queries
    ]


def analyze(out, name):
    """Reproduce all numerical results without new neural calls or label access."""
    manifest = verify(out)
    status = read(out / "run_status.json")
    if status["status"] != "COMPLETE":
        raise ValueError("Complete packet required")
    design = read(out / "private_design.json")
    expected = planned_packets(out, design, QUERIES)
    if len(expected) != manifest["units"]:
        raise ValueError("Incomplete design")
    dest = out / name
    dest.mkdir(exist_ok=False)
    checks = [read(p) for p in expected]
    for path, p in zip(expected, checks, strict=True):
        if sha(path.with_suffix(".npz")) != p["arrays_sha256"]:
            raise ValueError("Gradient bytes changed")
    write(
        dest / "manifest.json",
        {
            "analyzer_sha256": sha(Path(__file__)),
            "source_manifest_sha256": sha(out / "manifest.json"),
            "packet_sha256": {str(p.relative_to(out)): sha(p) for p in expected},
            "new_neural_calls": 0,
            "labels_read": False,
            "correction": "Initial summary-only glob read an AppleDouble sidecar; numeric tables and111 native calls are retained unchanged.",
        },
    )
    with np.load(out / "private_coordinates.npz", allow_pickle=False) as z:
        coordinates = dict(z)
    rows, pairs, jacobians = [], [], {}
    contraction_replay_max = 0.0
    lookup = {(p["index"], p["query"]): p for p in checks}
    if len(lookup) != len(expected):
        raise ValueError("Duplicate packet address")
    for a in design:
        index = a["index"]
        pos, g = coordinates[f"a{index}__positions"], coordinates[f"a{index}__grain"]
        with np.load(
            cached_unit(index, QUERIES[0]).with_suffix(".npz"), allow_pickle=False
        ) as z:
            h = z["full_state"][pos].astype(float)
        for q in QUERIES:
            dst = out / "private_packets" / f"a{index:02d}_{q}.json"
            p = lookup[(index, q)]
            if p["role"] != a["role"] or len(p["rows"]) != 4:
                raise ValueError("Packet role/factor coverage changed")
            with np.load(dst.with_suffix(".npz"), allow_pickle=False) as z:
                jacobians[(index, q)] = z["body_gradient"]
            for k, row in enumerate(p["rows"]):
                if row["factor"] != f"whole:F{k + 1:02d}":
                    raise ValueError("Factor order changed")
                d, _ = direction(h, g, k)
                direct = float(np.sum(d * jacobians[(index, q)]))
                contraction_replay_max = max(
                    contraction_replay_max, abs(direct - row["total"])
                )
                rows.append({"index": index, "role": a["role"], "query": q, **row})
        for qa, qb in itertools.combinations(QUERIES, 2):
            ja, jb = (jacobians[(index, q)] for q in (qa, qb))
            pairs.append(
                {
                    "index": index,
                    "query_a": qa,
                    "query_b": qb,
                    "gradient_cosine": cosine(ja, jb),
                    "mean_reader_cosine": cosine(ja.mean(0), jb.mean(0)),
                }
            )
    if contraction_replay_max > 1e-10:
        raise ValueError("Saved tangent does not replay its gradient contraction")
    frame, pairframe = pd.DataFrame(rows), pd.DataFrame(pairs)
    aggregate = []
    for q, g in frame.groupby("query", sort=True):
        norm = np.linalg.norm(g.total)
        aggregate.append(
            {
                "query": q,
                "authors": g["index"].nunique(),
                "directions": len(g),
                "total_rms": float(np.sqrt(np.mean(g.total**2))),
                "common_rms": float(np.sqrt(np.mean(g.common**2))),
                "localized_rms": float(np.sqrt(np.mean(g.localized**2))),
                "common_error_over_total": float(
                    np.linalg.norm(g.total - g.common) / norm
                )
                if norm
                else None,
                "finite_delta_error_max": float(g.finite_delta_error.max()),
                "finite_derivative_relative_l2_error": float(
                    np.linalg.norm(g.total - g.finite_derivative) / norm
                )
                if norm
                else None,
                "sign_agreement": float(
                    np.mean(np.sign(g.total) == np.sign(g.finite_derivative))
                ),
            }
        )
    frame.to_csv(dest / "private_tangents.csv", index=False)
    pairframe.to_csv(dest / "private_query_pairs.csv", index=False)
    pd.DataFrame(aggregate).to_csv(dest / "reader_summary.csv", index=False)
    pairframe.groupby(["query_a", "query_b"])[
        ["gradient_cosine", "mean_reader_cosine"]
    ].agg(["mean", "min", "max"]).to_csv(dest / "query_pair_summary.csv")
    frame.groupby(["query", "factor"])[
        ["total", "common", "localized", "finite_derivative"]
    ].agg(["mean", "std"]).to_csv(dest / "factor_summary.csv")
    summary = {
        "status": "COMPLETE",
        "units": len(checks),
        "directions": len(frame),
        "contraction_replay_max": contraction_replay_max,
        "score_replay_max": max(p["score_replay_error"] for p in checks),
        "canonicalization_max": max(p["canonicalization_max"] for p in checks),
        "identity_error_max": max(p["identity_error"] for p in checks),
        "gradient_replay_max": max(
            p["gradient_replay"]["gradient_max"] for p in checks if p["gradient_replay"]
        ),
        "finite_delta_error_max": float(frame.finite_delta_error.max()),
        "sign_matches": int(
            sum(np.sign(frame.total) == np.sign(frame.finite_derivative))
        ),
        "fits": 0,
        "labels_read": False,
        "calls": status["calls"],
        "seconds": status["seconds"],
        "files": {
            n: sha(dest / n)
            for n in [
                "private_tangents.csv",
                "private_query_pairs.csv",
                "reader_summary.csv",
                "query_pair_summary.csv",
                "factor_summary.csv",
            ]
        },
    }
    write(dest / "summary.json", summary)
    print(summary, flush=True)


def main():
    """Use explicit separate analysis folders; never repeat the native collection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=BASE / "logic_native_material_reader_v1_r2"
    )
    parser.add_argument("--name", default="analysis_r2")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if (
        not out.is_relative_to(BASE.resolve())
        or not args.name.replace("_", "").isalnum()
    ):
        raise ValueError("Use ignored results and simple output name")
    with threadpool_limits(limits=2):
        analyze(out, args.name)


if __name__ == "__main__":
    main()
