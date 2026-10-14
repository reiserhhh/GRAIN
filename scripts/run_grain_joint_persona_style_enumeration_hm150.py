#!/usr/bin/env python3
"""Resume frozen HM150 stages without interpreting AppleDouble files as data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_joint_persona_style_hm150 as J


class ScientificOutputPath(type(Path())):
    """Keep ordinary files, excluding only authenticated AppleDouble metadata."""

    def glob(self, pattern, **kwargs):
        """Filter metadata locally without changing global pathlib or old code."""
        for path in super().glob(pattern, **kwargs):
            if path.name.startswith("._") and path.is_file():
                with path.open("rb") as stream:
                    if stream.read(4) == b"\x00\x05\x16\x07":
                        continue
            yield path


def recover(out, stage):
    """Archive the terminal pre-forward failure before resuming numeric stages."""
    out = ScientificOutputPath(out)
    record = out / "enumeration_recovery.json"
    if not record.exists():
        old = J.H.load(out / "observation_status.json")
        if old["status"] != "FAILED" or "UnicodeDecodeError" not in old["error"]:
            raise ValueError("Expected recorded pre-observation metadata failure")
        files = (
            Path(__file__),
            ROOT / "tests/test_grain_joint_persona_style_enumeration.py",
            ROOT / "docs/GRAIN_JOINT_ENUMERATION_RECOVERY_HM150.md",
        )
        J.H.write_json(
            record,
            {
                "original_observation_failure": old,
                "reason": "AppleDouble file signature is metadata,not a scientific JSON/NPZ observation",
                "scientific_design_changed": False,
                "generated_articles_repeated": 0,
                "code_hashes": {str(p.relative_to(ROOT)): J.H.digest(p) for p in files},
            },
        )
    for name, checksum in J.H.load(record)["code_hashes"].items():
        if J.H.digest(ROOT / name) != checksum:
            raise ValueError("Enumeration adapter changed: " + name)
    J.check(out)
    try:
        getattr(J, stage)(out)
    except Exception as error:
        J.status(
            out,
            {"observe": "observation"}.get(stage, stage),
            "FAILED",
            error=f"{type(error).__name__}: {error}",
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("observe", "analyze", "verify"))
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/grain_math_response_v1/joint_persona_style_r2",
    )
    args = parser.parse_args()
    recover(args.output, args.stage)
