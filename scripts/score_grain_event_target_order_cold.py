#!/usr/bin/env python3
"""Cold whole-target analysis of a fixed addressed numeric vector edit."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_event_target_order as M
from scripts.run_grain_event_anchor_field_hm267 import selected


def main():
    """Require explicit numeric inputs without application labels or model calls."""
    p = argparse.ArgumentParser()
    p.add_argument("--source-bundle", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    args = p.parse_args()
    bank = selected(
        args.source_bundle,
        ("source_comments", "codes", "anchors", "variance", "directions"),
    )
    print(
        json.dumps(
            M.cold_record(bank, json.loads(args.input.read_text())), allow_nan=False
        )
    )


if __name__ == "__main__":
    main()
