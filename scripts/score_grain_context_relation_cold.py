#!/usr/bin/env python3
"""SOURCE relation scoring on frozen identified subsets,without outcomes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_context_relation as C


def main():
    """Apply one frozen bundle to strict outcome-free records,once."""
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "input", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    b = C.validate_bundle(json.loads(a.bundle.read_text()))
    with a.input.open() as source, a.output.open("x") as sink:
        for line in source:
            sink.write(
                json.dumps(C.cold_record(b, json.loads(line)), allow_nan=False) + "\n"
            )


if __name__ == "__main__":
    main()
