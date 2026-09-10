#!/usr/bin/env python3
"""Replay evidence and exact cotangent-error accounting in GRAIN action geometry."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_grain_real_author_path import BASE, read, sha, write
from scripts.run_grain_source_cotangent import checked_arrays, verify


def main():
    """Check original/replayed packets and action error from the g difference."""
    out = BASE / "logic_source_cotangent_reader_v1"
    verify(out)
    freeze, replay = (
        read(out / "prediction_freeze.json"),
        read(out / "replay_freeze.json"),
    )
    for name, digest in freeze["files"].items():
        if (
            not sha(out / name)
            == sha(out / "private_replay" / Path(name).name)
            == digest
        ):
            raise ValueError("Prediction replay mismatch")
    results = read(out / "analysis/summary.json")
    for name, digest in results["files"].items():
        if (
            not sha(out / "analysis" / name)
            == sha(out / "analysis_replay" / name)
            == digest
        ):
            raise ValueError("Analysis replay mismatch")
    error = 0.0
    for j in read(out / "private_design.json"):
        with np.load(checked_arrays(j), allow_pickle=False) as z:
            g = z["g"].astype(float)
        with np.load(
            out / "private_models" / (j["key"] + ".npz"), allow_pickle=False
        ) as z:
            means = dict(z)
        with np.load(
            out / "private_predictions" / (j["key"] + ".npz"), allow_pickle=False
        ) as z:
            lam = z["response_value"] + z["response_key"]
            target = np.einsum("fhd,hd->f", lam, g)
            for mode in ("own", "pooled"):
                observed = (
                    z[mode + "_value_actions"] + z[mode + "_key_actions"] - target
                )
                expected = np.einsum("fhd,hd->f", lam, means[mode + "_g"] - g)
                error = max(error, float(abs(observed - expected).max()))
    if error > 1e-10:
        raise ValueError("Cotangent action error identity failed")
    summary = {
        "exact_prediction_packets": len(freeze["files"]),
        "exact_analysis_csvs": len(results["files"]),
        "first_runtime": freeze["runtime"],
        "replay_runtime": replay["runtime"],
        "cotangent_action_error_identity_max": error,
        "native_forward_duality_error_max": results["native_forward_duality_error_max"],
        "new_neural_calls": 0,
        "new_fits": 0,
    }
    write(out / "replay_audit.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
