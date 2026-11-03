#!/usr/bin/env python3
"""Cold numeric claim-law replay,no neural runtime or application labels."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grain_research.construction.math_claim_meaning import cold


def main():
    """Read one qualified numeric record and return its full finite contacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    reply = cold(json.loads(args.input.read_text()))
    reply["neural_modules_loaded"] = any(
        name == "torch" or name.startswith("transformers") for name in sys.modules
    )
    if reply["neural_modules_loaded"]:
        raise RuntimeError(
            "Cold numeric interface unexpectedly imported a neural runtime"
        )
    print(json.dumps(reply, allow_nan=False))


if __name__ == "__main__":
    main()
