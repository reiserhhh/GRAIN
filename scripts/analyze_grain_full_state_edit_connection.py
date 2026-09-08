#!/usr/bin/env python3
"""Summarize actual context/query interventions without fitting outcomes or labels."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.vector_mechanisms.construction_bridge import correlation
from grain_research.vector_mechanisms.full_state_edit import grid_components
from scripts.run_grain_real_author_path import BASE, read, sha, write

COMPONENTS = (
    "rebase",
    "context_main",
    "query_main",
    "interaction",
    "context_allocated",
    "query_allocated",
)


def cancellation(values):
    """Bounded signed cancellation index; depends on the declared additive account."""
    x = np.asarray(values, float)
    total = float(np.abs(x).sum())
    return float(1 - abs(x.sum()) / total) if total > 1e-15 else None


def run(source):
    """Retain all twelve units, reference roles, signed effects and exact checks."""
    if read(source / "run_status.json")["status"] != "COMPLETE":
        raise ValueError("Complete acquisition required")
    inputs = read(source / "private_inputs.json")
    rows, hashes, replay = [], {}, []
    for a in inputs:
        for q in read(source / "manifest.json")["queries"]:
            path = source / "private_packets" / f"unit_{a['index']:02d}_{q}.json"
            packet = read(path)
            if (
                packet["author_id"] != a["author_id"]
                or packet["query"] != q
                or sha(path.with_suffix(".npz")) != packet["arrays_sha256"]
            ):
                raise ValueError("Source packet mismatch")
            s = {k: v["conditional_score"] for k, v in packet["cells"].items()}
            parts = grid_components(packet["full"]["conditional_score"], **s)
            np.testing.assert_allclose(
                list(parts.values()), list(packet["components"].values()), atol=1e-14
            )
            rows.append(
                {
                    "author_id": a["author_id"],
                    "unit": a["index"],
                    "role": a["role"],
                    "query": q,
                    **parts,
                    "opposed_allocations": parts["context_allocated"]
                    * parts["query_allocated"]
                    < 0,
                    "query_sign_switch": parts["query_main"]
                    * (parts["query_main"] + parts["interaction"])
                    < 0,
                    "cancellation": cancellation(
                        [
                            parts["rebase"],
                            parts["context_allocated"],
                            parts["query_allocated"],
                        ]
                    ),
                }
            )
            hashes[str(path.relative_to(source))] = sha(path)
            replay.extend(packet["replay_errors"].values())
    frame = pd.DataFrame(rows)
    metrics, relations, covariance = [], [], []
    for query, group in frame.groupby("query"):
        for c in ("natural", *COMPONENTS):
            metrics.append(
                {
                    "query": query,
                    "component": c,
                    "authors": len(group),
                    "mean": float(group[c].mean()),
                    "rms": float(np.sqrt(np.mean(group[c] ** 2))),
                    "r_with_natural": correlation(group[c], group.natural),
                }
            )
    for q, r in itertools.combinations(read(source / "manifest.json")["queries"], 2):
        a = frame[frame["query"].eq(q)].sort_values("unit")
        b = frame[frame["query"].eq(r)].sort_values("unit")
        for c in ("natural", *COMPONENTS):
            relations.append(
                {
                    "left": q,
                    "right": r,
                    "component": c,
                    "authors": 4,
                    "r": correlation(a[c], b[c]),
                }
            )
        for c, d in itertools.product(
            ("rebase", "context_allocated", "query_allocated"), repeat=2
        ):
            covariance.append(
                {
                    "left": q,
                    "right": r,
                    "left_component": c,
                    "right_component": d,
                    "covariance": float(
                        np.mean(
                            (a[c] - a[c].mean()).to_numpy()
                            * (b[d] - b[d].mean()).to_numpy()
                        )
                    ),
                }
            )
    out = source / "analysis"
    out.mkdir(exist_ok=False)
    frame.to_csv(out / "private_effects.csv", index=False)
    for name, values in (
        ("component_metrics", metrics),
        ("query_relations", relations),
        ("signed_covariance", covariance),
    ):
        pd.DataFrame(values).to_csv(out / (name + ".csv"), index=False)
    result = {
        "status": "COMPLETE",
        "scope": "four-author mechanistic observations; no population significance claim",
        "units": len(frame),
        "authors": frame.author_id.nunique(),
        "opposed_allocations": int(frame.opposed_allocations.sum()),
        "query_sign_switches": int(frame.query_sign_switch.sum()),
        "max_replay_error": max(replay),
        "max_account_error": float(
            np.max(
                np.abs(
                    frame.natural
                    - frame.rebase
                    - frame.context_allocated
                    - frame.query_allocated
                )
            )
        ),
        "cost": read(source / "run_status.json"),
        "input_hashes": hashes,
        "analysis_script_sha256": sha(__file__),
        "metrics": metrics,
        "query_relations": relations,
    }
    write(out / "summary.json", result)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("input_hashes", "metrics", "query_relations")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source", type=Path, default=BASE / "full_state_edit_connection_v1"
    )
    run(p.parse_args().source)
