#!/usr/bin/env python3
"""Cold numerical scoring: no text encoder, model weights or hidden truth input."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction.math_conditional_chart import score_chart


def main():
    """Apply a fixed calibration to only the supplied observation ledger."""
    p = argparse.ArgumentParser()
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--observations", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=("source", "local"), required=True)
    a = p.parse_args()
    calibration = json.loads(a.calibration.read_text())
    observations = json.loads(a.observations.read_text())
    results = []
    for r in observations:
        bank = 0 if a.mode == "source" else r["bank"]
        chart = calibration[f"{bank}:{r['family']}:{r['unit']}"]
        results.append(
            {
                "id": r["id"],
                "mode": a.mode,
                **score_chart(chart, r["means"], r["counts"], r["qualification"]),
            }
        )
    if "torch" in sys.modules or "transformers" in sys.modules:
        raise RuntimeError("Cold chart imported a neural runtime")
    with a.output.open("x") as f:
        json.dump(
            {"results": results, "neural_runtime_loaded": False, "truth_input": False},
            f,
            indent=2,
            allow_nan=False,
        )
    print("COLD", a.mode, len(results), "observation records")


if __name__ == "__main__":
    main()
