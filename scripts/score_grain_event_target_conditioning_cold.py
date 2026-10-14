#!/usr/bin/env python3
"""Read an addressed vector edit through a frozen SOURCE conditional instrument."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_event_target_conditioning as M
from scripts.run_grain_event_anchor_field_hm267 import selected


def main():
    """Require explicit numeric input and no model calls or inferred labels."""
    p = argparse.ArgumentParser()
    p.add_argument("--source-bundle", required=True, type=Path)
    p.add_argument("--input", required=True, type=Path)
    a = p.parse_args()
    bank = selected(
        a.source_bundle,
        ("source_comments", "codes", "anchors", "variance", "directions"),
    )
    result = M.cold_record(bank, json.loads(a.input.read_text()))
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
