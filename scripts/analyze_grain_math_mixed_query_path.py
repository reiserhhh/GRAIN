#!/usr/bin/env python3
"""Descriptive HM35 node-state and signed-support diagnostics; zero model calls."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_response import write_csv, write_json
from scripts.verify_grain_math_crossed_field import table

SOURCE = ROOT / "results/grain_math_response_v1/mixed_query_path_r2"
OUT = ROOT / "results/grain_math_response_v1/mixed_query_path_diagnostic_r1"


def analyze():
    """Keep all12 role/query/arm groups and no result-selected acceptance threshold."""
    if OUT.exists():
        raise FileExistsError("preserve previous descriptive report")
    verified = (
        ROOT
        / "results/grain_math_response_v1/mixed_query_path_verification_r2/summary.json"
    )
    if read_json(verified)["status"] != "VERIFIED_FINITE_HM35_NOT_GLOBAL":
        raise ValueError("actual complete HM35 verification required")
    files = [
        verified,
        *(
            SOURCE / "02" / name
            for name in ("nodes.csv", "path_comparisons.csv", "metrics.csv")
        ),
        Path(__file__),
    ]
    inputs = {str(p.relative_to(ROOT)): digest(p) for p in files}
    nodes = table(SOURCE / "02/nodes.csv")
    rows = table(SOURCE / "02/path_comparisons.csv")
    result = []
    for metric in table(SOURCE / "02/metrics.csv"):
        key = metric["role"], metric["query"], metric["arm"]
        chosen = [x for x in rows if (x["role"], x["query"], x["arm"]) == key]
        values = []
        for row in chosen:
            selected = sorted(
                [
                    x
                    for x in nodes
                    if (x["role"], x["query"], x["arm"], x["path"])
                    == (*key, row["path"])
                ],
                key=lambda x: int(x["node"]),
            )
            if [float(x["t"]) for x in selected] != [0, 0.5, 1]:
                raise ValueError("all three original nodes required")
            values.append([float(x["full"]) for x in selected])
        a = np.asarray(values)
        y = np.array([float(x["actual_gradient_halfcontrast"]) for x in chosen])
        rms = float(np.sqrt(np.mean(y * y)))
        full, supported, unsupported = (
            float(metric[k + "_rms"])
            for k in ("simpson_full", "simpson_supported", "simpson_unsupported")
        )
        cross = float(metric["support_cross_mean"])
        identity = abs(
            full * full - supported * supported - unsupported * unsupported - cross
        )
        if identity > 1e-15:
            raise ValueError("signed action-energy identity failed")
        result.append(
            {
                "role": key[0],
                "query": key[1],
                "arm": key[2],
                "paths": len(chosen),
                "supported_rms_over_full_rms": supported / full,
                "unsupported_rms_over_full_rms": unsupported / full,
                "cross_over_full_energy": cross / (full * full),
                "signed_energy_identity_error": identity,
                "initial_node_halfaction_relative_rmse": float(
                    np.sqrt(np.mean((a[:, 0] / 2 - y) ** 2)) / rms
                ),
                "midpoint_halfaction_relative_rmse": float(
                    np.sqrt(np.mean((a[:, 1] / 2 - y) ** 2)) / rms
                ),
                "node_variation_rms_over_node_rms": float(
                    np.sqrt(
                        np.mean((a - a.mean(axis=1, keepdims=True)) ** 2)
                        / np.mean(a * a)
                    )
                ),
                "simpson_relative_gradient_rmse": float(
                    metric["simpson_relative_gradient_rmse"]
                ),
                "trapezoid_relative_gradient_rmse": float(
                    metric["trapezoid_relative_gradient_rmse"]
                ),
                "finite_h_relative_rmse": float(metric["finite_error_rms"])
                / float(metric["native_rms"]),
            }
        )
    OUT.mkdir(parents=True)
    write_csv(OUT / "node_state_support.csv", result)
    write_json(
        OUT / "summary.json",
        {
            "status": "DESCRIPTIVE_HM35_ALL_GROUPS_NOT_NEW_REPLICATION",
            "rows": result,
            "input_hashes": inputs,
            "new_model_calls": 0,
            "new_fits": 0,
            "scope": "same full P path, signed effects; ratios are not psychological variance shares",
        },
    )
    write_json(
        OUT / "artifact_hashes.json",
        {p.name: digest(p) for p in OUT.iterdir() if p.is_file()},
    )
    print(
        {
            "status": "DESCRIPTIVE_REPLAY_COMPLETE",
            "groups": len(result),
            "new_model_calls": 0,
        },
        flush=True,
    )


if __name__ == "__main__":
    analyze()
