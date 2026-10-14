#!/usr/bin/env python3
"""Read the numerical predictive-link bundle without any language-model runtime."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grain_research.construction import math_author_predictive_link as M


def score(bundle):
    """Score all targets/modes from supplied sums, without text or target labels."""
    if bundle["version"] != "HM257-v1" or bundle["targets"] != list(M.TARGETS):
        raise ValueError("Unknown probability-observation contract")
    z = np.asarray([r["chosen"] for r in bundle["units"]], float)[..., None]
    a = np.asarray([r["partition"] for r in bundle["units"]], float)[..., None]
    values = M.accounts(z, a)
    rows = []
    for i, unit in enumerate(bundle["units"]):
        rows.append(
            {
                "unit_id": unit["unit_id"],
                "tokens": unit["tokens"],
                "stranger": values["stranger"][i, :, 0].tolist(),
                "order": values["order"][i, :, 0].tolist(),
                "carry": values["carry"][i, :, 0].tolist(),
                "neutral": float(values["stranger_neutral"][i, 0]),
            }
        )
    return {
        "version": "HM257-v1",
        "neural_runtime_loaded": "torch" in sys.modules,
        "rows": rows,
    }


def main():
    """Require an explicit immutable numerical input; never infer a corpus path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(score(json.loads(args.input.read_text())), allow_nan=False))


if __name__ == "__main__":
    main()
