#!/usr/bin/env python3
"""One existing MPS export repair, preserving the failed scientific primary."""

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_item_bridge_mps as C
from scripts import run_grain_observer_operation_hm280 as E

DEFAULT = E.DEFAULT / "mps_cpu64_recovery_r1"
SCIENCE = (
    "docs/GRAIN_OBSERVER_OPERATION_MPS_RECOVERY_HM280.md",
    "scripts/run_grain_observer_operation_hm280_recovery.py",
    "grain_research/construction/math_item_bridge_mps.py",
    "tests/test_grain_item_bridge_mps.py",
)


def prepare(out):
    """Require the exact terminal zero-outcome failure and deduct its original budget."""
    if out.exists():
        raise FileExistsError("No second companion")
    manifest = E.R.check(E.DEFAULT)
    runtime = E.R.read(E.DEFAULT / "runtime.json")
    if (
        runtime["state"] != "FAILED"
        or runtime["families"] != 0
        or "MPS Tensor to float64" not in runtime["error"]
        or list((E.DEFAULT / "batches").glob("*.npz"))
        or (E.DEFAULT / "ACTIVE.lock").exists()
    ):
        raise ValueError("Only this zero-outcome mechanical fault can be repaired")
    used = runtime["model_seconds"]
    cap = manifest["config"]["max_model_seconds"]
    if not 0 < used < cap:
        raise ValueError("No remaining ORIGINAL finite cap")
    (out / "batches").mkdir(parents=True)
    names = tuple(E.R.read(E.DEFAULT / "prepared_seal.json"))
    for name in names:
        shutil.copyfile(E.DEFAULT / name, out / name)
    manifest["config"] = {**manifest["config"], "max_model_seconds": cap - used}
    manifest["code"] = {
        **manifest["code"],
        **{p: E.R.digest(ROOT / p) for p in SCIENCE},
    }
    manifest["inputs"] = {
        **manifest["inputs"],
        **{
            str((E.DEFAULT / p).relative_to(ROOT)): E.R.digest(E.DEFAULT / p)
            for p in names
            + ("manifest.json", "prepared_seal.json", "runtime.json", "head.npz")
        },
    }
    manifest["repair"] = {
        "failed_primary": str(E.DEFAULT.relative_to(ROOT)),
        "primary_model_seconds": used,
        "original_total_cap": cap,
        "zero_saved_outcomes": True,
        "reused_HM255_CPU_normalizer_no_model_or_science_change": True,
    }
    E.R.write(out / "manifest.json", manifest)
    E.R.seal(out, "prepared_seal.json", names)
    E.H.status(out, {"state": "PREPARED", "families": 0, "model_seconds": 0})
    print("One companion prepared, remaining ORIGINAL cap", cap - used, flush=True)


def main():
    """Use the frozen primary stages with only the existing MPS-compatible export."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "stage", choices=("prepare", "collect", "analyze", "verify", "check")
    )
    a = p.parse_args()
    if a.stage == "prepare":
        prepare(DEFAULT)
    elif a.stage == "check":
        E.R.check(
            DEFAULT,
            ("collected_seal.json", "analyzed_seal.json", "verification_seal.json"),
        )
        print("HM280 primary and recovery seals PASS")
    else:
        E.I.native_items = C.native_items
        getattr(E, a.stage)(DEFAULT)


if __name__ == "__main__":
    main()
