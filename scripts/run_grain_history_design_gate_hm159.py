#!/usr/bin/env python3
"""Separately sealed, explicitly post-outcome observation-law closure probe."""

from __future__ import annotations

import hashlib
import json
import sys
from fractions import Fraction as F
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_history_design_gate as G
from grain_research.construction import math_history_resolution as M

OUT = (
    ROOT
    / "results/grain_math_response_v1/history_resolution_r1/design_gate_followup_r1"
)
CODE = (
    "grain_research/construction/math_history_design_gate.py",
    "scripts/run_grain_history_design_gate_hm159.py",
    "tests/test_grain_history_design_gate.py",
    "docs/GRAIN_HISTORY_DESIGN_GATE_FOLLOWUP_HM159.md",
)


def sha(path):
    """Read-only artifact digest."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    """Retain all attempts by exclusive output creation."""
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    """Freeze before calculation, or execute the exact predeclared cases."""
    if sys.argv[1:] == ["prepare"]:
        OUT.mkdir()
        files = [ROOT / p for p in CODE] + [
            ROOT / "grain_research/construction/math_history_resolution.py"
        ]
        write(
            OUT / "manifest.json",
            {
                "post_outcome": True,
                "code": {str(p): sha(p) for p in files},
                "primary_summary_hash": sha(OUT.parent / "summary.json"),
            },
        )
        return
    if sys.argv[1:] != ["run"]:
        raise ValueError("Choose prepare or run")
    manifest = json.loads((OUT / "manifest.json").read_text())
    assert sha(OUT.parent / "summary.json") == manifest["primary_summary_hash"]
    for path, expected in manifest["code"].items():
        assert sha(path) == expected, path
    p, even, odd = M.matched_populations()
    rows = []
    for mode in ("shared_independent", "distinct_parameters", "copied_channel"):
        left, right = G.history_law(p, even, mode), G.history_law(p, odd, mode)
        tv = sum(abs(a - b) for a, b in zip(left, right, strict=True)) / 2
        checks = G.source_observable_checks(left)
        rows.append(
            {
                "mode": mode,
                "TV": float(tv),
                "TV_exact": str(tv),
                "same_observation_law": left == right,
                **{key: str(value) for key, value in checks.items()},
            }
        )
    support, mass = [F(1, 4), F(3, 4)], [F(1, 2), F(1, 2)]
    dependent = G.history_law(support, mass, "balanced_dependence")
    independent = G.history_law(support, mass, "shared_independent")
    difference = (
        sum(abs(a - b) for a, b in zip(dependent, independent, strict=True)) / 2
    )
    cancellation = {
        key: str(value) for key, value in G.source_observable_checks(dependent).items()
    }
    result = {
        "post_outcome": True,
        "counterexamples": rows,
        "balanced_dependence": cancellation,
        "balanced_vs_independent_TV": float(difference),
        "balanced_vs_independent_TV_exact": str(difference),
        "decision": "Do not allocate two independent observations merely from two channels or zero pooled covariance. Positive compiler requires the stated shared-parameter independent law; keep alternative kernels and event identity.",
        "new_neural_calls": 0,
        "parameter_search": False,
        "original_scope_complete": False,
    }
    write(OUT / "summary.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
