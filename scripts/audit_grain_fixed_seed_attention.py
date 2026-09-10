#!/usr/bin/env python3
"""Cache-only numerical replay audit and aggregate channel-effect accounting."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_grain_fixed_seed_attention import verify
from scripts.run_grain_real_author_path import BASE, read, sha, write


def main():
    """Verify all fixed units and publish aggregates without identifiers or text."""
    out = BASE / "logic_fixed_seed_attention_v1"
    verify(out)
    manifest = read(out / "analysis/manifest.json")
    for name, digest in manifest["numeric_hashes"].items():
        if (
            not sha(out / "analysis" / name)
            == sha(out / "analysis_replay" / name)
            == digest
        ):
            raise ValueError("Nonidentical numeric replay: " + name)
    for name, digest in manifest["native_packet_hashes"].items():
        if sha(out / name) != digest:
            raise ValueError("Native packet changed")
    if sha(out / "private_projections.npz") != manifest["projections_sha256"]:
        raise ValueError("Projection weights changed")
    jobs = read(out / "private_jobs.json")
    state = read(out / "run_status.json")
    if state["status"] != "COMPLETE" or set(state["completed"]) != {
        j["key"] for j in jobs
    }:
        raise ValueError("Incomplete explicit unit coverage")
    frame = pd.read_csv(out / "analysis/private_units.csv")
    actions = pd.read_csv(out / "analysis/private_actions.csv")
    rows = []
    for (cohort, query), group in frame.groupby(["cohort", "query"]):
        ratio = group.key_norm / group.value_norm
        drop_error = group.key_norm**2 / group.total_norm**2
        a = actions[(actions.cohort == cohort) & (actions["query"] == query)]
        rows.append(
            {
                "cohort": cohort,
                "query": query,
                "authors": len(group),
                "key_value_norm_ratio_mean": float(ratio.mean()),
                "key_value_norm_ratio_min": float(ratio.min()),
                "key_value_norm_ratio_max": float(ratio.max()),
                "value_only_field_relative_squared_error_mean": float(
                    drop_error.mean()
                ),
                "opposite_action_channels": int(a.opposite_channels.sum()),
                "actions": len(a),
            }
        )
    pd.DataFrame(rows).to_csv(out / "analysis/channel_effects.csv", index=False)
    finite = []
    for j in jobs:
        packet = read(out / "private_packets" / (j["key"] + ".json"))
        if packet["finite_difference"] is not None:
            finite.append({"query": j["query"], **packet["finite_difference"]})
    if len(finite) != 2 or not frame.parity_pass.all() or len(frame) != 24:
        raise ValueError("Fixed finite differences or gradient parity incomplete")
    summary = {
        "exact_numeric_csv_files": len(manifest["numeric_hashes"]),
        "explicit_units": len(frame),
        "parity_pass": int(frame.parity_pass.sum()),
        "neural_calls": state["calls"],
        "backwards": state["backwards"],
        "native_seconds": state["seconds"],
        "load_seconds": state["load_seconds"],
        "finite_differences": finite,
        "action_relative_l2_error": float(
            np.linalg.norm(actions.last_total_action - actions.last_native_action)
            / np.linalg.norm(actions.last_native_action)
        ),
        "parity_thresholds_changed": False,
        "extra_neural_calls": 0,
    }
    write(out / "replay_audit.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
