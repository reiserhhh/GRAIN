#!/usr/bin/env python3
"""Cold shared-profile/occasion-mode comparison on two numeric histories."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_profile_episode as M
from scripts.score_grain_author_belief_cold import load_bundle


def main():
    """Return all fixed laws without model calls or current psychological labels."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    result = M.cold(load_bundle(args.bundle), json.loads(args.input.read_text()))
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
