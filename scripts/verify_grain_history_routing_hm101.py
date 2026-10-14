#!/usr/bin/env python3
"""Independent finite cycle/Mobius readback without rerunning the neural observer."""

from __future__ import annotations

import hashlib
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction.math_history_routing import route_mask
from scripts.run_grain_history_routing_hm101 import check, settings, unit
from scripts.run_grain_observer_parity_hm96 import load, verify, write


def run(out):
    """Check actual captures, independent cycle sums and intervention reconstruction."""
    manifest = check(out)
    verify(out, load(out / "01/raw_seal.json"))
    verify(out, load(out / "02/output_seal.json"))
    with np.load(out / "02/accounts.npz", allow_pickle=False) as z:
        fields, contrasts, components = (
            z[n] for n in ("fields", "contrasts", "components")
        )
    max_error = 0.0
    for corpus, bridge, family, arm in itertools.product(
        range(2), range(2), range(6), range(4)
    ):
        f = fields[corpus, bridge, family, :, :, arm]
        cycle = (f[0, 1] + f[1, 2] + f[2, 0] - f[1, 0] - f[2, 1] - f[0, 2]) / 9
        max_error = max(
            max_error,
            float(np.max(np.abs(cycle - contrasts[corpus, bridge, family, arm]))),
        )
    # Independently expand all four observed route arms from the saved coefficients.
    for arm, (d, r) in enumerate(manifest["spec"]["arms"]):
        rebuilt = (
            components[..., 0, :, :]
            + d * components[..., 1, :, :]
            + r * components[..., 2, :, :]
            + d * r * components[..., 3, :, :]
        )
        max_error = max(
            max_error, float(np.max(np.abs(rebuilt - contrasts[..., arm, :, :])))
        )
    for row in load(out / "00/rows.json"):
        _, meta = unit(out, row["unit"])
        allowed = route_mask(
            len(row["input_ids"]),
            row["history"],
            row["relay"],
            row["current"],
            row["padding"],
            row["direct"],
            row["indirect"],
        )
        if (
            meta["trace"]["permission_sha256"]
            != hashlib.sha256(allowed.tobytes()).hexdigest()
        ):
            raise ValueError("Applied permission graph differs from the frozen design")
        if meta["trace"]["masked_blocks"] != list(range(17)):
            raise ValueError("Incomplete route intervention trace")
    if max_error > manifest["spec"]["arithmetic_atol"]:
        raise ValueError("Independent author/corner reconstruction failed")
    summary = load(out / "02/summary.json")
    if (
        summary["all_cut_current_kv_max"] > manifest["spec"]["carrier_atol"]
        or summary["noop_hook_kv_max"] > manifest["spec"]["carrier_atol"]
    ):
        raise ValueError("Declared instrumentation controls did not hold")
    result = {
        **summary,
        "status": "VERIFIED_FINITE_HM101_NOT_GLOBAL",
        "alternate_cycle_and_corner_max_error": max_error,
        "same_agent_not_third_party": True,
        "actual_mask_traces": len(load(out / "00/rows.json")),
        "new_neural_calls_in_verifier": 0,
    }
    write(out / "verification.json", result)
    print("VERIFIED HM101", result, flush=True)


if __name__ == "__main__":
    run(settings()[1])
