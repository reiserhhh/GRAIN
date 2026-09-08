#!/usr/bin/env python3
"""Compare the same source authors' old native-query and new anchor pairing paths."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.level_response import correlation
from scripts.run_grain_real_author_path import BASE, read, sha, write
from scripts.run_grain_source_remainder import covariance


def main():
    """Join identical view-zero observations; do not compare different sample r values."""
    out = BASE / "anchored_pairing_path_v1"
    old = BASE / "reader_pattern_s18_r1"
    design = read(BASE / "conditional_baseline_construction_v1/private_design.json")
    if read(out / "summary.json")["status"] != "COMPLETE":
        raise ValueError("Complete anchored pairing packet required")
    # S19 preserved the exact S18 packet hashes used in its original intervention.
    old_manifest = read(BASE / "functional_gradient_s19_r2/manifest.json")
    hashes = old_manifest["source_packet_hashes"]
    lookup = {}
    for path in sorted((old / "private_packets").glob("author_*.json")):
        if sha(path) != hashes[path.name]:
            raise ValueError("Historical native-query packet changed")
        row = read(path)
        if row["view"] != 0:
            raise ValueError("Old intervention is not view zero")
        lookup[row["author_id"]] = row
    rows = []
    for row in design:
        if row["query"] != "neuroticism":
            continue
        native = lookup[row["author_id"]]
        packet = read(out / "private_packets" / f"author_{row['index']:03d}.json")
        anchored = next(v for v in packet["views"] if v["view"] == 0)
        rows.append(
            {
                "author_id": row["author_id"],
                "native_pair": native["native"] - native["full_removed"],
                "anchor_pair": anchored["pair_effect"],
                "native_score": native["native"],
                "anchor_score": anchored["anchor"],
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) != 24 or len(frame.author_id.unique()) != 24:
        raise ValueError("Exactly the same24 source authors required")
    source = pd.read_csv(
        out / "private_source_connection.csv", dtype={"author_id": str}
    )
    frame = frame.merge(
        source[source["query"] == "neuroticism"][["author_id", "y_true"]],
        on="author_id",
        validate="one_to_one",
    )
    metrics = []
    for column in ("native_pair", "anchor_pair", "native_score", "anchor_score"):
        metrics.append(
            {
                "component": column,
                "authors": 24,
                "views_per_author": 1,
                "r_y": correlation(frame.y_true.to_numpy(), frame[column].to_numpy()),
                "cov_y": covariance(frame.y_true.to_numpy(), frame[column].to_numpy()),
                "rms": float(np.sqrt(np.mean(frame[column] ** 2))),
            }
        )
    result = {
        "status": "COMPLETE",
        "scope": "same24 N authors, old S18 and new M14 view0; do not mix with 3view source mean",
        "native_anchor_pair_r": correlation(
            frame.native_pair.to_numpy(), frame.anchor_pair.to_numpy()
        ),
        "metrics": metrics,
        "neural_calls": 0,
        "source_packet_hashes": hashes,
    }
    frame.to_csv(out / "private_query_condition_comparison.csv", index=False)
    write(out / "query_condition_comparison.json", result)
    print(pd.DataFrame(metrics).to_string(index=False))
    print({"native_anchor_pair_r": result["native_anchor_pair_r"]})


if __name__ == "__main__":
    main()
