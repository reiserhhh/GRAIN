#!/usr/bin/env python3
"""Score one observed publication window with a fixed conditional reference."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grain_research.construction import math_conditional_publication as C


def main():
    """Use no application labels,LLM judgment or current-batch normalization."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--reference", choices=("bank0", "bank1", "pooled"), default="pooled"
    )
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    rule = {
        k: np.array(v)
        for k, v in json.loads(args.bundle.read_text())[args.reference].items()
    }
    result = C.cold(rule, json.loads(args.input.read_text()))
    print(
        json.dumps(
            {
                section: {k: v.tolist() for k, v in values.items()}
                for section, values in result.items()
            },
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
