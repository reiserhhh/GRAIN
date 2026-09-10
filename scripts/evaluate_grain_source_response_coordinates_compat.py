#!/usr/bin/env python3
"""Run the frozen NEXT75 evaluator with its expected pandas object-string dtype."""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_source_response_coordinates as primary


def main():
    """Record a runtime-only compatibility amendment, leaving primary code untouched."""
    p = argparse.ArgumentParser()
    p.add_argument("--folder", default="predictions", choices=("predictions", "replay"))
    p.add_argument("--output", type=Path, default=primary.OUT)
    args = p.parse_args()
    old, out = primary.old, args.output
    m = old.verify(out)
    path = out / "analysis_runtime_amendment.json"
    if not path.exists():
        for folder in ("models", "predictions", "replay"):
            primary.prev.freeze(out, folder)
        old.write(
            path,
            {
                "reason": "pandas3 inferred StringDtype was not classified as object by the frozen baseline comparator",
                "fix": "pd.option_context(future.infer_string=False); numeric code unchanged",
                "primary_commit": m["commit"],
                "amendment_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                ).strip(),
                "wrapper_sha256": old.sha(Path(__file__)),
                "test_sha256": old.sha(
                    ROOT / "tests/test_grain_response_coordinates_compat.py"
                ),
                "frozen_stage_hashes": {
                    n: old.sha(out / n / "freeze.json")
                    for n in ("models", "predictions", "replay")
                },
                "failed_analysis_retained": "analysis_failed_stringdtype_v1",
                "failed_analysis_cpu_seconds": None,
                "extra_fits": 0,
                "extra_predictions": 0,
            },
        )
    amendment = old.read(path)
    if amendment["wrapper_sha256"] != old.sha(Path(__file__)):
        raise ValueError("Runtime amendment code changed")
    for folder, sha in amendment["frozen_stage_hashes"].items():
        if old.sha(out / folder / "freeze.json") != sha:
            raise ValueError("Pre-amendment source model or prediction freeze changed")
    with threadpool_limits(limits=1), pd.option_context("future.infer_string", False):
        primary.evaluate(out, args.folder)


if __name__ == "__main__":
    main()
