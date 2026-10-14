#!/usr/bin/env python3
"""Repair only CSV bool serialization for the frozen HM32 alternate verifier."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts import verify_grain_math_query_cube as V
from scripts.run_grain_math_response import write_json

ORIGINAL_TABLE = V.table
BOOL_COLUMNS = (
    "greedy_first_token_is_candidate",
    "complete_response_parse_rate_measured",
)
OUT = ROOT / "results/grain_math_response_v1/query_context_cube_verification_r2"


def normalize(rows):
    """Convert only declared native bool columns; retain inherited role strings."""
    result = [dict(r) for r in rows]
    for row in result:
        for name in BOOL_COLUMNS:
            if name in row:
                if row[name] not in ("True", "False"):
                    raise ValueError("declared boolean CSV column changed type")
                row[name] = "1" if row[name] == "True" else "0"
    return result


def table_with_native_booleans(path):
    """Adapt serialization, without changing any score or reconstruction formula."""
    return normalize(ORIGINAL_TABLE(path))


def run(parent, out):
    """Run the original alternate verifier with an explicit audited table adapter."""
    if out.exists():
        raise FileExistsError("preserve prior companion verification")
    frozen = V.check(parent)
    original = "scripts/verify_grain_math_query_cube.py"
    if digest(ROOT / original) != frozen["files"][original]:
        raise ValueError("frozen verifier changed")
    manifest = {
        "frozen_original_verifier": original,
        "original_sha256": digest(ROOT / original),
        "files": {
            "scripts/run_grain_math_query_cube_verification_companion.py": digest(
                Path(__file__)
            ),
            "tests/test_grain_math_query_cube_verification_companion.py": digest(
                ROOT / "tests/test_grain_math_query_cube_verification_companion.py"
            ),
        },
        "repair": "CSV True/False only in two native bool columns -> numeric1/0 for float comparisons; unchanged inherited role strings, scores and formulas",
        "primary_failure": "ValueError: could not convert string to float: True in native endpoint replay",
        "new_model_calls": 0,
        "new_fits": 0,
    }
    old_table = V.table
    try:
        V.table = table_with_native_booleans
        V.verify(parent, out)
    finally:
        V.table = old_table
    write_json(out / "csv_adapter_manifest.json", manifest)
    summary = read_json(out / "summary.json")
    write_json(
        out / "summary.json",
        {
            **summary,
            "csv_adapter": manifest["repair"],
            "original_verifier_unchanged": True,
        },
    )
    write_json(
        out / "artifact_hashes.json",
        {
            str(p.relative_to(out)): digest(p)
            for p in out.iterdir()
            if p.is_file()
            and not p.name.startswith("._")
            and p.name != "artifact_hashes.json"
        },
    )
    print(read_json(out / "summary.json"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=V.PARENT)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    run(args.parent, args.out)
