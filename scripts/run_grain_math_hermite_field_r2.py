#!/usr/bin/env python3
"""HM26 field-name recovery; preserve the frozen, unexecuted r1 implementation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import read_json as native_read_json
from scripts import run_grain_math_hermite_field as base


def read_json(path):
    """Expose the existing conditional score under the legacy in-memory alias only."""
    result = native_read_json(path)
    if isinstance(result, dict) and isinstance(result.get("native"), dict):
        native = result["native"]
        if "conditional_score" in native:
            if "score" in native and native["score"] != native["conditional_score"]:
                raise ValueError("ambiguous readout units")
            native["score"] = native["conditional_score"]
    return result


def configure():
    """Select fresh paths and include both frozen implementation and recovery files."""
    base.read_json = read_json
    base.SPEC = ROOT / "configs/grain_math_hermite_field_v2.json"
    base.PLAN = ROOT / "docs/GRAIN_HERMITE_FIELD_CACHE_RECOVERY_V2.md"
    base.OUT = ROOT / "results/grain_math_response_v1/hermite_field_r2"
    base.CODE += (
        "scripts/run_grain_math_hermite_field_r2.py",
        "scripts/verify_grain_math_hermite_field_r2.py",
        "tests/test_grain_math_hermite_field_recovery.py",
        str(base.SPEC.relative_to(ROOT)),
        str(base.PLAN.relative_to(ROOT)),
    )


if __name__ == "__main__":
    configure()
    base.main()
