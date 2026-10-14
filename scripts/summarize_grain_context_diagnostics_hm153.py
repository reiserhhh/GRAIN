#!/usr/bin/env python3
"""Post-outcome descriptive grouping; not a new prospective experiment."""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_grain_context_diagnostics_hm153 import (
    DEFAULT,
    PARENT,
    check,
    digest,
    read,
    table,
    write,
)


def rows(path):
    """Read the complete finite output, without sample selection."""
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main():
    """Save all task/family groups and link all108 old condition contrasts."""
    out = DEFAULT
    check(out)
    for name, checksum in read(out / "output_seal.json").items():
        if digest(out / name) != checksum:
            raise ValueError("Sealed result changed")
    if (
        read(out / "alternate_audit.json")["status"]
        != "PASS_FINITE_ALTERNATE_NOT_BLIND"
    ):
        raise ValueError("Alternate numerical verification not complete")
    groups = defaultdict(list)
    for r in rows(out / "cross_target_accounts.csv"):
        groups[r["family"], r["situation"]].append(float(r["cosine"]))
    grouped = [
        {
            "family": k[0],
            "situation": k[1],
            "mean_cosine": float(np.mean(v)),
            "median_cosine": float(np.median(v)),
            "negative_units": int(np.sum(np.array(v) < 0)),
            "token_units": len(v),
        }
        for k, v in sorted(groups.items())
    ]
    if len(grouped) != 18 or any(r["token_units"] != 144 for r in grouped):
        raise ValueError("Incomplete task/family support")
    key = lambda r: tuple(r[k] for k in ("axis", "family", "situation", "style"))
    parent = {key(r): r for r in rows(PARENT / "condition_summary.csv")}
    joined = []
    for r in rows(out / "condition_accounts.csv"):
        old = parent[key(r)]
        full, reset = (
            float(old["full_high_minus_low"]),
            float(old["reset_sum_high_minus_low"]),
        )
        shape, calibration, total = [
            float(r[k])
            for k in ("delta_shape_sum", "delta_calibration_sum", "delta_native_k_sum")
        ]
        if not np.isclose(total, full - reset, atol=1e-8, rtol=1e-8):
            raise ArithmeticError("Parent prefix contrast changed")
        joined.append(
            {
                **r,
                "parent_full": full,
                "parent_reset": reset,
                "parent_sign_reversal": full * reset < 0,
                "shape_calibration_opposite": shape * calibration < 0,
                "shape_total_opposite": shape * total < 0,
            }
        )
    table(out / "descriptive_cross_target_groups.csv", grouped)
    table(out / "descriptive_parent_condition_join.csv", joined)
    write(
        out / "interpretation_receipt.json",
        {
            "scope": "Post-outcome descriptive grouping of ALL declared support;no new test,pvalue,fit or selection",
            "parent_reversal_cells": sum(r["parent_sign_reversal"] for r in joined),
            "shape_calibration_opposite_cells": sum(
                r["shape_calibration_opposite"] for r in joined
            ),
            "shape_total_opposite_cells": sum(
                r["shape_total_opposite"] for r in joined
            ),
            "original_output_seal_sha256": digest(out / "output_seal.json"),
            "descriptive_outputs": {
                p: digest(out / p)
                for p in (
                    "descriptive_cross_target_groups.csv",
                    "descriptive_parent_condition_join.csv",
                )
            },
            "code_sha256": digest(Path(__file__)),
            "original_scope_complete": False,
        },
    )
    print(read(out / "interpretation_receipt.json"))


if __name__ == "__main__":
    main()
