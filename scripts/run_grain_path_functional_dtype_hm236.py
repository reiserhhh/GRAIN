#!/usr/bin/env python3
"""Exact binary dtype compatibility, preserving the original pre-outcome failure."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_path_functional_hm236 as P

ORIGINAL = P.DEFAULT
RECOVERY = ORIGINAL / "dtype_recovery_r1"


def exact_binary_codes(values):
    """Convert exactly binary finite labels to integers without rounding values."""
    y = np.asarray(values)
    if not np.isfinite(y).all() or not np.isin(y, (0, 1)).all():
        raise ValueError("Exact finite binary values required;no rounding or clipping")
    result = y.astype(np.int64)
    if not np.array_equal(y, result):
        raise ValueError("Binary dtype conversion changes SOURCE values")
    return result


def prepare_recovery(out):
    """Save a separate exact-valued input before running unchanged scientific code."""
    old = P.R.check(ORIGINAL)
    failure = P.R.read(ORIGINAL / "compile_failure.json")
    if (
        failure["type"] != "IndexError"
        or "indices must be of integer" not in failure["message"]
    ):
        raise ValueError("Expected original index-type failure")
    z = P.S.selected(ORIGINAL / "source_inputs.npz", ("source_paths", "targets"))
    y = exact_binary_codes(z["targets"])
    out.mkdir(parents=True, exist_ok=False)
    own = str(Path(__file__).resolve().relative_to(ROOT))
    test = "tests/test_grain_path_functional_dtype.py"
    files = (
        "manifest.json",
        "prepared_seal.json",
        "source_inputs.npz",
        "compile_failure.json",
    )
    P.R.write(
        out / "manifest.json",
        {
            "config": old["config"],
            "code": old["code"] | {p: P.R.digest(ROOT / p) for p in (own, test)},
            "inputs": old["inputs"]
            | {
                str((ORIGINAL / p).relative_to(ROOT)): P.R.digest(ORIGINAL / p)
                for p in files
            },
            "compatibility_only": "exact0/1float to int64 for array addresses;formulas unchanged",
            "original_failure_preserved": failure,
            "new_question": old["new_question"],
            "original_scope_complete": False,
        },
    )
    P.R.archive(out / "source_inputs.npz", source_paths=z["source_paths"], targets=y)
    quality = P.R.read(ORIGINAL / "input_quality.json") | {
        "SOURCE_binary_value_conversion_max_error": float(
            np.max(abs(z["targets"] - y))
        ),
        "SOURCE_paths_unchanged": True,
        "before_new_science_outcomes": True,
        "no_new_partition_threshold_solver_or_scoring": True,
    }
    P.R.write(out / "input_quality.json", quality)
    P.R.seal(out, "prepared_seal.json", ("source_inputs.npz", "input_quality.json"))
    print(quality, flush=True)


if __name__ == "__main__":
    P.DEFAULT = RECOVERY
    P.prepare = prepare_recovery
    P.main()
