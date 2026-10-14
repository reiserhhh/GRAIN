#!/usr/bin/env python3
"""Identify a future-length readout-coordinate effect without any new forwards."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_grain_event_history_observation_hm92 import observe
from scripts.run_grain_hierarchical_query_hm88 import (
    load,
    sha,
    table,
    unit,
    verify,
    write,
)
from scripts.run_grain_query_evidence_hm86 import seal

OUT = ROOT / "results/grain_math_response_v1/event_history_observation_r1"


def run():
    """Check all future K1-polarity pairs with K2 first and a fixed query place."""
    manifest = load(OUT / "00/manifest.json")
    verify(ROOT, manifest["code"] | manifest["source"])
    verify(OUT, manifest["inputs"])
    verify(OUT, load(OUT / "01/raw_seal.json"))
    verify(OUT, load(OUT / "02/output_seal.json"))
    rows, bundle = load(OUT / "00/rows.json"), load(OUT / "00/bundle.json")
    lookup = {(r["frame"], r["order"], r["target"], *r["coordinates"]): r for r in rows}
    dest = OUT / "04_position_diagnostic"
    if dest.exists():
        raise FileExistsError("Preserve prior position diagnostics")
    dest.mkdir()
    records = []
    for row in rows:
        x, y, g, h, a, b = row["coordinates"]
        if row["order"] != 1 or x != 1:
            continue
        old = lookup[row["frame"], 1, row["target"], -1, y, g, h, a, b]
        group = row["action_groups"][1]
        stop = max(row["body_indices"][i] for i in group) + 1
        if row["input_ids"][:stop] != old["input_ids"][:stop]:
            raise ValueError("Future-polarity test lost identical early prefix")
        low, high = unit(OUT, old["unit"]), unit(OUT, row["unit"])
        ref, _, _ = observe(low, old, bundle)
        observed, _, _ = observe(high, row, bundle)
        displacement = old["body_indices"][-1] - row["body_indices"][-1]
        fixed = copy.deepcopy(bundle)
        fixed["gaps"] = [gap + displacement for gap in bundle["gaps"]]
        corrected, _, _ = observe(high, row, fixed)
        ix, iy = row["body_indices"], old["body_indices"]
        end = max(group) + 1
        vector_error = max(
            float(np.abs(high[name][ix[:end]] - low[name][iy[:end]]).max())
            for name in ("k", "v")
        )
        records.append(
            {
                "unit": row["unit"],
                "reference_unit": old["unit"],
                "frame": row["frame"],
                "family": row["target"],
                "query_position_difference": -displacement,
                "early_key_value_max_error": vector_error,
                "K2_local_paired_original_max_difference": float(
                    np.abs(observed[6] - ref[6]).max()
                ),
                "K2_local_paired_fixed_position_max_difference": float(
                    np.abs(corrected[6] - ref[6]).max()
                ),
                "K2_uniform_original_max_difference": float(
                    np.abs(observed[7] - ref[7]).max()
                ),
            }
        )
    table(dest / "future_length_pairs.csv", records)
    write(
        dest / "summary.json",
        {
            "status": "RESULT_AFTER_COORDINATE_DIAGNOSTIC_NOT_NEW_REPLICATION",
            "code_sha256": sha(Path(__file__)),
            "new_model_calls": 0,
            "pairs": len(records),
            "early_key_value_max_error": max(
                r["early_key_value_max_error"] for r in records
            ),
            "original_paired_max_difference": max(
                r["K2_local_paired_original_max_difference"] for r in records
            ),
            "fixed_position_paired_max_difference": max(
                r["K2_local_paired_fixed_position_max_difference"] for r in records
            ),
            "uniform_max_difference": max(
                r["K2_uniform_original_max_difference"] for r in records
            ),
            "all_primary_scope_controls_preserved": True,
            "interpretation": "Global endpoint query registration can act without changing an early carrier;equal-length scope-flip primary is unchanged",
        },
    )
    write(dest / "output_seal.json", seal(OUT, dest))
    print("DIAGNOSED HM92 query position", len(records), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run()
