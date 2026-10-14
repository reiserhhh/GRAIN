#!/usr/bin/env python3
"""Reproduce the full numeric item-operation packet without neural imports."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_observer_operation as M


def main():
    """Contract archived final states with the two explicit output vocabulary rows."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--packet", type=Path, required=True)
    a = p.parse_args()
    bank = json.loads((a.packet / "bank.json").read_text())
    with np.load(a.packet / "head.npz", allow_pickle=False) as z:
        weights, bias = z["weights"], z["bias"]
    rows = json.loads((a.packet / "units.json").read_text())
    logits = np.empty((48, 2, 4, 50, 2))
    for i, row in enumerate(rows):
        if row["unit"] >= 0:
            with np.load(
                a.packet / "batches" / f"{i:03d}.npz", allow_pickle=False
            ) as z:
                logits[
                    row["unit"], M.OPERATIONS.index(row["operation"]), row["context"]
                ] = M.head_logits(z["states"], weights, bias)
    account = M.operation_accounts(logits, bank)
    print(
        json.dumps(
            {
                s: {
                    o: np.sqrt(np.mean(v**2, axis=0)).tolist()
                    for o, parts in keys["balanced_key"]["parts"].items()
                    for v in [parts["join"]]
                }
                for s, keys in account.items()
            }
        )
    )


if __name__ == "__main__":
    main()
