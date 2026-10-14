#!/usr/bin/env python3
"""Independent-formula numeric audit of the masked-cache path witness."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_author_evidence_merge_hm102 as H


def audit(parent):
    """Recompute every endpoint and focus difference without neural libraries."""
    out = parent / "support_gauge_r1"
    plan, reported = H.load(out / "plan.json"), H.load(out / "summary.json")
    H.verify_hashes(ROOT, plan["code"] | plan["sources"])
    H.verify_hashes(out, H.load(out / "seal.json"))
    chunks, ids = [], []
    for path in sorted(out.glob("batch_*.npz")):
        with np.load(path, allow_pickle=False) as z:
            chunks.append(z["readings"])
            ids.append(z["ids"])
    x, ids = np.concatenate(chunks, axis=1), np.concatenate(ids)
    with np.load(parent / "02/moments.npz", allow_pickle=False) as z:
        endpoints = z["endpoints"]
    spec = H.load(parent / "00/manifest.json")["input_descriptor"]
    valid = np.array(spec["valid"])
    changed = np.array(
        [
            any(a != b for a, b in zip(valid[lo], valid[hi], strict=True))
            for lo, hi in ids
        ]
    )
    values = np.empty((len(ids), 2, 6))
    for p, mode, q in np.ndindex(len(ids), 2, 6):
        # Replay float32 grouping and independently compare with a float64 sum.
        da = x[3, p, 0, mode, q] - x[2, p, 0, mode, q]
        db = x[3, p, 1, mode, q] - x[2, p, 1, mode, q]
        values[p, mode, q] = da - db
    errors = {
        "endpoint": abs(
            float(abs(x[:2] - endpoints).max()) - reported["endpoint_max_difference"]
        ),
        "midpoint": abs(
            float(abs(x[3] - x[2]).max()) - reported["midpoint_reading_max_difference"]
        ),
        "unchanged": abs(
            float(abs(values[~changed]).max())
            - reported["midpoint_focus_max_difference_unchanged"]
        ),
    }
    # The reported RMS accumulated in float32. Bound its rounding separately,
    # rather than call that the semantic path discrepancy or adjust the data.
    raw = (x[3, :, 0] - x[2, :, 0]) - (x[3, :, 1] - x[2, :, 1])
    rms32 = np.sqrt(np.mean(raw**2, axis=(0, 2)))
    rms64 = np.sqrt(np.mean(values**2, axis=(0, 2)))
    for mode in range(2):
        errors[f"mode{mode}_RMS"] = abs(
            float(rms32[mode]) - reported["midpoint_focus_difference_RMS"][mode]
        )
    if (
        max(errors.values()) > 1e-12
        or changed.sum() != reported["changed_availability_pairs"]
    ):
        raise AssertionError(errors)
    if "torch" in sys.modules or "transformers" in sys.modules:
        raise AssertionError("Readback loaded a neural runtime")
    H.write(
        out / "readback.json",
        {
            "status": "NUMERIC_GAUGE_WITNESS_READBACK_PASSED_NOT_GLOBAL",
            "errors": errors,
            "readings": int(x.size),
            "pairs": len(ids),
            "float64_RMS": rms64.tolist(),
            "float32_reduction_rounding_max": float(abs(rms64 - rms32).max()),
            "same_agent_not_third_party": True,
            "torch_or_transformers_loaded": False,
            "rounding_scope": "Original float32 response arithmetic reproduced;large midpoint effect not a machine-precision artifact",
            "primary_quadrature_failure_retained": True,
            "audit_code_sha256": H.sha(__file__),
        },
    )
    H.write(out / "final_seal.json", H.seals(out, out))
    print("Gauge readback", errors, len(ids), int(x.size), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    audit(parser.parse_args().parent.resolve())
