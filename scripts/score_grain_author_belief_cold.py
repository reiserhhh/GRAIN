#!/usr/bin/env python3
"""Score a qualified whole path from SOURCE parameters, without neural imports."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_author_belief as M


def load_bundle(path):
    """Load only the source law, never application outcomes or model answers."""
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def main():
    """Emit the two-episode posterior trajectory under the fixed integrated law."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    a = p.parse_args()
    result = M.cold(load_bundle(a.bundle), json.loads(a.input.read_text()))
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
