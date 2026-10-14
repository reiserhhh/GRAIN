#!/usr/bin/env python3
"""Score numeric own-evidence/focus ledgers without text, truth or model runtime."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grain_research.construction.math_policy_coupling import score


def main():
    """Read only sealed kernels, typed owner addresses and joint event counts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernels", type=Path, required=True)
    parser.add_argument("--ledgers", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with (
        np.load(args.kernels, allow_pickle=False) as source,
        np.load(args.ledgers, allow_pickle=False) as ledgers,
    ):
        result = score(
            source["kernels"],
            ledgers["joint_counts"],
            ledgers["evidence_owner"],
            ledgers["query_owner"],
            ledgers["origins"],
        )
    if any(name in sys.modules for name in ("torch", "transformers")):
        raise RuntimeError("Cold interface loaded a neural runtime")
    with args.output.open("xb") as stream:
        np.savez_compressed(stream, **result, neural_runtime_loaded=np.array(False))
    print("COLD HM118", result["observed"].shape, "no neural runtime", flush=True)


if __name__ == "__main__":
    main()
