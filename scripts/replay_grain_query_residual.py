#!/usr/bin/env python3
"""Reuse the verified file guard with the new source-only residual predictor."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.vector_mechanisms.query_residual_input import predict_extension
from scripts import replay_grain_content_coefficients as runtime

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--jobs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--forbid", type=Path, action="append", default=[])
    a = p.parse_args()
    runtime.predict_coefficients = predict_extension
    runtime.replay(a.jobs, a.output, a.forbid)
