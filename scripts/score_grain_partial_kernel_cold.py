#!/usr/bin/env python3
"""Cold partial-observation Dirichlet response law,no neural runtime."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grain_research.construction import math_partial_kernel as M
from scripts.score_grain_author_belief_cold import load_bundle


def main():
    """Score an identified author record with the fixed finite numerical contract."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=29020261014)
    args = parser.parse_args()
    reply = M.cold(
        load_bundle(args.bundle), json.loads(args.input.read_text()), 64, args.seed
    )
    reply["neural_modules_loaded"] = any(
        n == "torch" or n.startswith("transformers") for n in sys.modules
    )
    if reply["neural_modules_loaded"]:
        raise RuntimeError("Cold operator imported a neural runtime")
    print(json.dumps(reply, allow_nan=False))


if __name__ == "__main__":
    main()
