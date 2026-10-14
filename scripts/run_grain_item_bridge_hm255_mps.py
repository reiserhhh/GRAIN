#!/usr/bin/env python3
"""Execute a separate CPU64 companion, preserving the original failed HM255 seal."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_item_bridge_mps as C
from scripts import run_grain_item_bridge_hm255 as J


def main():
    """Check the technical failure boundary and reuse the unchanged finite plan."""
    config = "configs/grain_item_bridge_hm255_mps.json"
    cfg = J.R.read(ROOT / config)
    failure = ROOT / cfg["failure_receipt"]
    old = J.R.read(failure)
    if (
        J.R.digest(failure) != cfg["failure_receipt_sha256"]
        or old["status"] != "FAILED"
        or old["completed_batches"] != 0
        or list((failure.parent / "batches").glob("*.npz"))
    ):
        raise ValueError("Changed zero-outcome recovery boundary; do not restart")
    if (
        abs(cfg["max_model_seconds"] + old["model_seconds"] - cfg["original_total_cap"])
        > 1e-10
    ):
        raise ValueError("Technical recovery would renew the finite budget")
    J.CONFIG = config
    J.DEFAULT = failure.parent / "mps_cpu64_recovery_r1"
    J.SCIENCE += (
        config,
        "docs/GRAIN_ITEM_BRIDGE_MPS_RECOVERY_HM255.md",
        "grain_research/construction/math_item_bridge_mps.py",
        "scripts/run_grain_item_bridge_hm255_mps.py",
        "tests/test_grain_item_bridge_mps.py",
    )
    J.M.native_items = C.native_items
    J.main()


if __name__ == "__main__":
    main()
