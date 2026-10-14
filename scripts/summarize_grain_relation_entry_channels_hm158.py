#!/usr/bin/env python3
"""Describe all HM158 qualified groups after outcomes, without new model calls."""

from __future__ import annotations

import csv
import itertools
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_relation_entry_channels_hm158 as C


def describe(out=C.DEFAULT):
    """Keep every condition, target and wording cell in a descriptive companion."""
    C.check(out)
    C.authenticate_results(out)
    with (out / "components.csv").open() as stream:
        records = list(csv.DictReader(stream))
    qualified = [
        r
        for r in records
        if r["active_later"] == "True"
        and int(r["mask"]) == (5 if r["axis"] == "e" else 6)
    ]
    components = (
        "key_entry",
        "value_entry",
        "recursive_interaction",
        "joint_entry",
        "native",
    )
    cells = []
    for condition, axis, target, family in itertools.product(
        ("0", "1"), ("e", "n"), C.R.Q.AXES, C.R.Q.FAMILIES
    ):
        selected = [
            r
            for r in qualified
            if (r["condition"], r["axis"], r["target"], r["family"])
            == (condition, axis, target, family)
        ]
        found = {r["component"]: float(r["coefficient"]) for r in selected}
        if len(selected) != len(components) or set(found) != set(components):
            raise ValueError("Missing or duplicate qualified entry cell")
        cell = {
            "condition": int(condition),
            "axis": axis,
            "target": target,
            "family": family,
            **found,
        }
        cell["value_joint_same_direction"] = (
            found["value_entry"] * found["joint_entry"] > 0
        )
        cell["joint_native_same_direction"] = found["joint_entry"] * found["native"] > 0
        cells.append(cell)
    if len(qualified) != 120 or len(cells) != 24:
        raise ValueError("Incomplete qualified support")
    grouped = []
    for axis, target, component in itertools.product(
        ("e", "n"), C.R.Q.AXES, components[:-1]
    ):
        selected = [c for c in cells if (c["axis"], c["target"]) == (axis, target)]
        xx = math.fsum(c[component] ** 2 for c in selected)
        yy = math.fsum(c["joint_entry"] ** 2 for c in selected)
        grouped.append(
            {
                "axis": axis,
                "target": target,
                "component": component,
                "count": len(selected),
                "rms": math.sqrt(xx / len(selected)),
                "signed_joint_projection": math.fsum(
                    c[component] * c["joint_entry"] for c in selected
                )
                / yy
                if yy
                else None,
                "same_joint_direction": sum(
                    c[component] * c["joint_entry"] > 0 for c in selected
                ),
                "same_native_direction": sum(
                    c[component] * c["native"] > 0 for c in selected
                ),
            }
        )
    companion = out / "post_outcome_description_r1"
    companion.mkdir(exist_ok=False)
    for name, rows in (("qualified_cells.csv", cells), ("entry_groups.csv", grouped)):
        with (companion / name).open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    receipt = {
        "scope": "Post-outcome complete description, not a new hypothesis or experiment",
        "input_output_seal_sha256": C.R.H.digest(out / "output_seal.json"),
        "script_sha256": C.R.H.digest(Path(__file__)),
        "qualified_cells": len(cells),
        "grouped_rows": len(grouped),
        "new_model_forwards": 0,
        "fit_or_selection": False,
        "joint_native_reversals": [
            c for c in cells if not c["joint_native_same_direction"]
        ],
        "outputs": {
            name: C.R.H.digest(companion / name)
            for name in ("qualified_cells.csv", "entry_groups.csv")
        },
    }
    with (companion / "receipt.json").open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"groups": grouped, "receipt": receipt}, indent=2))


if __name__ == "__main__":
    describe()
