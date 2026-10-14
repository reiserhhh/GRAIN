#!/usr/bin/env python3
"""HM34 companion: retain the actually executed float32 displacement order."""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_response import write_json

PRIMARY = ROOT / "scripts/verify_grain_math_query_response.py"
PARENT = ROOT / "results/grain_math_response_v1/early_query_response_operator_r1"
OUT = ROOT / "results/grain_math_response_v1/early_query_response_verification_r2"
OLD = """                            step = (
                                arms[arm]
                                * (
                                    am["spec"]["radius_relative_span_norm"]
                                    * np.linalg.norm(h)
                                    / np.linalg.norm(arms[arm])
                                )
                            ).astype(np.float32)"""
NEW = """                            step = _actual_step(
                                h, arms[arm], am["spec"]["radius_relative_span_norm"]
                            )"""


def actual_step(h, direction, radius):
    """Match the primary perturbation's divide/multiply order before float32."""
    d = np.asarray(direction, float)
    h = np.asarray(h, np.float32)
    return (d / np.linalg.norm(d) * radius * np.linalg.norm(h)).astype(np.float32)


def verify(parent, out):
    """Execute unchanged alternate formulas except one explicit physical adapter."""
    m = read_json(parent / "00/manifest.json")
    if digest(PRIMARY) != m["files"][str(PRIMARY.relative_to(ROOT))]:
        raise ValueError("primary verifier changed after freeze")
    source = PRIMARY.read_text()
    if source.count(OLD) != 1:
        raise ValueError("exact single physical operation adapter required")
    files = {
        name: digest(ROOT / name)
        for name in (
            "scripts/verify_grain_math_query_response_r2.py",
            "tests/test_grain_math_query_response_r2.py",
        )
    }
    module = types.ModuleType("grain_hm34_exact_displacement_verifier")
    module.__file__ = str(PRIMARY)
    module.__dict__["_actual_step"] = actual_step
    # Hash-checked frozen local code, with exactly one numeric-expression adapter.
    exec(compile(source.replace(OLD, NEW), str(PRIMARY), "exec"), module.__dict__)  # noqa: S102
    module.verify(parent, out)
    if any(digest(ROOT / name) != sha for name, sha in files.items()):
        raise ValueError("physical adapter changed during replay")
    write_json(
        out / "adapter_manifest.json",
        {
            "files": files,
            "primary_verifier_sha256": digest(PRIMARY),
            "primary_result_manifest_sha256": digest(parent / "00/manifest.json"),
            "replacement_count": 1,
            "scope": "only float32 physical displacement operation order; Gram/SVD/source/roles/metrics/tolerances unchanged",
            "previous_failure": "alternate algebraic reassociation changed actual float32 endpoint displacement; error1.0179531240162021e-7 preserved",
            "new_model_calls": 0,
            "new_fits": 0,
        },
    )
    write_json(
        out / "adapter_artifact_hashes.json",
        {
            name: digest(out / name)
            for name in ("summary.json", "adapter_manifest.json")
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=PARENT)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    verify(args.parent, args.out)
