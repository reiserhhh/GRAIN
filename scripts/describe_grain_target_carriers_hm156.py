#!/usr/bin/env python3
"""Post-outcome qualified-character descriptions; no fitting or model calls."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_target_carrier_closure_hm156 as R


def describe(out):
    """Retain both actor drivers for both targets, including null operations."""
    R.check(out)
    for name, checksum in R.R.read(out / "output_seal.json").items():
        if R.H.digest(out / name) != checksum:
            raise ValueError("Primary outcome changed: " + name)
    target = out / "post_outcome_description_r1"
    target.mkdir(exist_ok=False)
    with (out / "characters.csv").open() as f:
        coefficients = list(csv.DictReader(f))
    lookup = {
        (int(v["group"]), v["mode"], v["axis"], v["family"], int(v["mask"])): float(
            v["logodds"]
        )
        for v in coefficients
    }
    rows = []
    for gi, group in enumerate(R.R.GROUPS):
        for mode in R.MODES:
            for axis in R.R.M.AXES:
                for family in R.R.M.FAMILIES:
                    for driver, mask in (("e", 5), ("n", 6)):
                        base = lookup[gi, "natural", axis, family, mask]
                        value = lookup[gi, mode, axis, family, mask]
                        reference = (
                            -2 * base
                            if mode.startswith(driver + "_") or mode.startswith("both_")
                            else 0.0
                        )
                        rows.append(
                            {
                                "group": gi,
                                "bank": group[0],
                                "condition": group[1],
                                "order": group[2],
                                "mode": mode,
                                "axis": axis,
                                "family": family,
                                "actor_driver": driver,
                                "base_coefficient": base,
                                "patched_coefficient": value,
                                "effect_coefficient": value - base,
                                "natural_flip_effect_coefficient": reference,
                                "signed_coefficient_transfer": (value - base)
                                / reference
                                if reference != 0
                                else None,
                            }
                        )
    R.H.write_csv(target / "qualified_actions.csv", rows)
    accounts = []
    for mode, order, driver in (("e_other", 0, "e"), ("n_other", 1, "n")):
        for axis in R.R.M.AXES:
            values = [
                r
                for r in rows
                if r["mode"] == mode
                and r["order"] == order
                and r["actor_driver"] == driver
                and r["axis"] == axis
            ]
            accounts.append(
                {
                    "mode": mode,
                    "order": order,
                    "axis": axis,
                    "actor_driver": driver,
                    "cells": len(values),
                    "mean_effect_coefficient": np.mean(
                        [v["effect_coefficient"] for v in values]
                    ),
                    "mean_natural_flip_coefficient": np.mean(
                        [v["natural_flip_effect_coefficient"] for v in values]
                    ),
                    "same_direction": sum(
                        v["effect_coefficient"] * v["natural_flip_effect_coefficient"]
                        > 0
                        for v in values
                    ),
                    "opposite_direction": sum(
                        v["effect_coefficient"] * v["natural_flip_effect_coefficient"]
                        < 0
                        for v in values
                    ),
                    "zero_direction": sum(
                        v["effect_coefficient"] * v["natural_flip_effect_coefficient"]
                        == 0
                        for v in values
                    ),
                }
            )
    with (out / "interactions.csv").open() as f:
        inter = list(csv.DictReader(f))
    summary = {
        "post_outcome_not_prospective": True,
        "qualified_rows": len(rows),
        "active_other_accounts": accounts,
        "encoding_over_reader_rms_range": [
            min(float(v["encoding_rms"]) / float(v["reader_rms"]) for v in inter),
            max(float(v["encoding_rms"]) / float(v["reader_rms"]) for v in inter),
        ],
        "opposing_encoding_reader_inner_products": sum(
            float(v["encoding_reader_inner"]) < 0 for v in inter
        ),
        "interaction_accounts": len(inter),
        "new_model_forwards": 0,
        "note": "Finite actor-swap coefficients, not accuracy/personality correlation or psychological-information shares. Encoding/readout composition measured for the weak ending carrier only.",
    }
    R.write_json(target / "summary.json", summary)
    R.write_json(
        target / "provenance.json",
        {
            "script_sha256": R.H.digest(Path(__file__)),
            "primary_output_seal_sha256": R.H.digest(out / "output_seal.json"),
            "output_hashes": {
                n: R.H.digest(target / n)
                for n in ("summary.json", "qualified_actions.csv")
            },
        },
    )
    print(R.M.json_native(summary))


if __name__ == "__main__":
    describe(R.DEFAULT)
