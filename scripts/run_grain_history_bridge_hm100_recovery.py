#!/usr/bin/env python3
"""Preserve r1 quadrature failure and run identical finite science at 32 nodes."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_history_bridge_hm100 as R
from scripts.run_grain_observer_parity_hm96 import seal, write


def main():
    """Use an explicitly new protocol without rewriting frozen r1 code/results."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=("prepare", "compile", "export", "cold", "evaluate", "verify")
    )
    args = parser.parse_args()
    R.SPEC = ROOT / "configs/grain_history_bridge_hm100_recovery.json"
    R.CODE += (
        "configs/grain_history_bridge_hm100_recovery.json",
        "scripts/run_grain_history_bridge_hm100_recovery.py",
    )
    spec, out = R.settings()
    previous = ROOT / spec["failed_predecessor"]
    R.check(previous)
    for n in (1, 2, 3):
        R.stage(previous, n)
    with threadpool_limits(limits=1):
        if args.stage == "prepare":
            R.prepare(spec, out)
            write(
                out / "00/recovery.json",
                {
                    "r1_failure": "16point integral closure2.4813706644977174e-10 exceeds fixed1e-10",
                    "r1_exact_shapley_error": 2.220446049250313e-15,
                    "old_results_preserved": True,
                    "new_nodes": [32, 64],
                    "scientific_conditions_or_tolerance_changed": False,
                },
            )
        elif args.stage == "compile":
            R.compile_source(spec, out)
        elif args.stage == "export":
            R.check(out)
            R.stage(out, 1)
            dest = out / "02"
            dest.mkdir()
            for name in ("primitives.npz", "summary.json"):
                shutil.copyfile(previous / "02" / name, dest / name)
            write(out / "02_seal.json", seal(out, dest))
            print("REUSED frozen query-independent export;0calls", flush=True)
        elif args.stage == "cold":
            R.cold(out)
        elif args.stage == "evaluate":
            R.evaluate(spec, out)
        else:
            from scripts.verify_grain_history_bridge_hm100 import run

            run()


if __name__ == "__main__":
    main()
