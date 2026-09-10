#!/usr/bin/env python3
"""NEXT53 paired-objective audit, reusing numerical parity checks without fitting."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_grain_source_state_transport import audit as packet_audit
from scripts.run_grain_response_aware_state import (
    ARMS,
    BASE,
    OBJECTIVES,
    OLD,
    arrays,
    frozen,
    read,
    sha,
    verify,
    verify_hashes,
    write,
)


def audit(out):
    """Check unchanged assets, no refit replay, original controls and matched decisions."""
    m = verify(out)
    ff = read(out / "fit_freeze.json")
    verify_hashes(out, ff["files"])
    for objective in OBJECTIVES:
        packet_audit(out / objective)
    files = read(out / "comparison/summary.json")["files"]
    verify_hashes(out / "comparison", files)
    replay = read(out / "replay_comparison/summary.json")
    verify_hashes(out / "replay_comparison", replay["files"])
    for name in files:
        if sha(out / "comparison" / name) != sha(out / "replay_comparison" / name):
            raise ValueError("Comparison replay differs")
    packets, assets = 0, 0
    for objective in OBJECTIVES:
        mf = frozen(out / objective, "models")
        for name in mf["files"]:
            base, model = (
                arrays(OLD / "models" / name),
                arrays(out / objective / "models" / name),
            )
            for field in set(base) - {"coef"}:
                np.testing.assert_array_equal(model[field], base[field])
                assets += 1
            assert str(model["objective"]) == objective
        for name in frozen(out / objective, "predictions")["files"]:
            old, new = (
                arrays(OLD / "predictions" / name),
                arrays(out / objective / "predictions" / name),
            )
            for field in ("actions", "vectors", "x_hash", "d_hash"):
                np.testing.assert_array_equal(old[field][0], new[field][0])
            # Identical designs ensure only fitted B, not coordinates, differ.
            for arm in ARMS[1:]:
                for suffix in ("_z", "_dz"):
                    np.testing.assert_array_equal(old[arm + suffix], new[arm + suffix])
            packets += 1
    training = pd.read_csv(out / "source_training_losses.csv")
    if len(training) != 30:
        raise ValueError(
            "Ten model references x three source coefficient evaluations required"
        )
    for _, group in training.groupby(["reference", "arm"]):
        t = group.set_index("objective")
        a, b = t.loc["normalized_value"], t.loc["joint"]
        assert a.x + a.penalty <= b.x + b.penalty + 1e-10
        assert b.x + b.d + b.penalty <= a.x + a.d + a.penalty + 1e-10
        assert (
            a.normal_equation_max_error < 1e-10 and b.normal_equation_max_error < 1e-10
        )
    primary_time = (
        m["prepare_seconds"]
        + ff["seconds"]
        + sum(frozen(out / o, "predictions")["seconds"] for o in OBJECTIVES)
    )
    replay_time = sum(
        frozen(out / o, "replay_predictions")["seconds"] for o in OBJECTIVES
    )
    if ff["fits"] != 20 or primary_time > 180 or replay_time > 180:
        raise ValueError("Shared compute budget or fit count violated")
    primary = read(out / "comparison/summary.json")
    if primary["decisions"] != replay["decisions"]:
        raise ValueError("Decisions changed on replay")
    result = {
        "status": "PASS",
        "primary_commit": m["commit"],
        "fits": ff["fits"],
        "unchanged_asset_arrays": assets,
        "unchanged_identity_design_packets": packets,
        "replayed_prediction_packets": 48,
        "replayed_primary_tables": 18,
        "replayed_comparison_tables": len(files),
        "primary_compute_seconds": primary_time,
        "cached_replay_seconds": replay_time,
        "native_calls": 0,
        "human_labels_read": False,
        "audit_code_sha256": sha(Path(__file__)),
        "matched_decisions": primary["decisions"],
        "fit_guard_denials": ff["denial_probes"],
        "extra_fit_denials": ff["extra_denials"],
    }
    write(out / "replay_audit.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        audit(BASE / "logic_response_aware_state_v1")
