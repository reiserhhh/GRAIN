#!/usr/bin/env python3
"""Alternate arithmetic audit of actual source edits and qualified effects."""

from __future__ import annotations

import csv
import itertools
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_carrier_edit_transport_hm157 as R


def verify(out=R.DEFAULT):
    """Rebuild every used memory argument and all scalar character accounts."""
    R.check(out)
    for seal in ("capture_seal.json", "output_seal.json"):
        for name, checksum in R.R.read(out / seal).items():
            if R.H.digest(out / name) != checksum:
                raise ValueError("Sealed artifact changed: " + name)
    rows, design = (R.R.read(out / f"{p}.json") for p in ("rows", "design"))
    source = R.R.arrays(out / "source_operations.npz")
    captures = {
        (entry["row"], entry["mode"]): R.R.arrays(
            out / "captures" / f"{entry['batch']:03d}.npz"
        )
        for entry in design
    }
    lookup = {(r["condition"], r["order"], r["e"], r["n"]): r["id"] for r in rows}
    weights = (
        R.R.OLD.weights()
        .cpu()[[R.R.read(out / "queries.json")["answers"][s][0] for s in ("Yes", "No")]]
        .numpy()
        .astype(float)
    )
    max_native, max_argument, max_formula = 0.0, 0.0, 0.0
    memory_records = []
    for entry in design:
        row, mode = rows[entry["row"]], entry["mode"]
        actual = captures[row["id"], mode]
        expected = actual["states"].astype(float) @ weights.T
        max_native = max(
            max_native, float(np.max(abs(expected - actual["selected_logits"])))
        )
        if not np.allclose(expected, actual["selected_logits"], atol=1e-4, rtol=1e-5):
            raise ValueError("CPU64 native output-head reconstruction failed")
        if mode == "natural":
            continue
        base = captures[row["id"], "natural"]
        if not np.array_equal(actual["prefix_states"], base["prefix_states"]):
            raise ValueError("Prefix changed under suffix intervention")
        saved = R.R.arrays(out / "arguments" / f"{entry['batch']:03d}.npz")
        if mode == "self":
            slots = list(range(len(row["history"])))
        else:
            slots = row["scopes"][
                "neuroticism" if mode.startswith("e_") else "extraversion"
            ]
        if saved["slots"].tolist() != slots:
            raise ValueError("Incorrect carrier support")
        for name in ("history_key", "history_value"):
            expected = base[name].copy()
            if mode != "self":
                axis = 0 if mode.startswith("e_") else 1
                i, j = (row["e"] + 1) // 2, (row["n"] + 1) // 2
                if mode.endswith("native"):
                    donor_id = lookup[
                        row["condition"],
                        row["order"],
                        -row["e"] if axis == 0 else row["e"],
                        -row["n"] if axis == 1 else row["n"],
                    ]
                    delta = captures[donor_id, "natural"][name].astype(float) - base[
                        name
                    ].astype(float)
                else:
                    x = source[f"{row['condition']}_{row['order']}_{name}"].astype(
                        float
                    )
                    if axis == 0:
                        delta = x[1 - i, j] - x[i, j]
                        if mode.endswith("marginal"):
                            delta = (
                                (x[1 - i, 0] - x[i, 0]) + (x[1 - i, 1] - x[i, 1])
                            ) / 2
                    else:
                        delta = x[i, 1 - j] - x[i, j]
                        if mode.endswith("marginal"):
                            delta = (
                                (x[0, 1 - j] - x[0, j]) + (x[1, 1 - j] - x[1, j])
                            ) / 2
                    if mode.endswith("reverse"):
                        delta = -delta
                expected[:, :, slots] = (
                    base[name][:, :, slots].astype(float) + delta[:, :, slots]
                ).astype(base[name].dtype)
            error = float(
                np.max(abs(expected.astype(float) - saved[name].astype(float)))
            )
            max_argument = max(max_argument, error)
            if not np.array_equal(expected, saved[name]):
                raise ValueError("Actual source arithmetic argument failed")
            if mode.startswith(("e_source_", "n_source_")):
                axis = 0 if mode.startswith("e_") else 1
                native_entry = next(
                    item
                    for item in design
                    if item["row"] == row["id"]
                    and item["mode"] == ("e_native" if axis == 0 else "n_native")
                )
                native_argument = R.R.arrays(
                    out / "arguments" / f"{native_entry['batch']:03d}.npz"
                )[name]
                x = (
                    saved[name][:, :, slots].astype(float) - base[name][:, :, slots]
                ).ravel()
                y = (
                    native_argument[:, :, slots].astype(float) - base[name][:, :, slots]
                ).ravel()
                xx, yy, xy = float(x @ x), float(y @ y), float(x @ y)
                memory_records.append(
                    {
                        "row": row["id"],
                        "mode": mode,
                        "primitive": name,
                        "active_later": (axis == 0 and row["order"] == 0)
                        or (axis == 1 and row["order"] == 1),
                        "rms": float(np.sqrt(xx / len(x))),
                        "native_rms": float(np.sqrt(yy / len(y))),
                        "signed_projection": xy / yy if yy else None,
                        "relative_difference": float(
                            np.linalg.norm(x - y) / np.sqrt(yy)
                        )
                        if yy
                        else None,
                    }
                )
    with (out / "characters.csv").open() as f:
        characters = list(csv.DictReader(f))
    for record in characters:
        c, o, mask = (int(record[k]) for k in ("condition", "order", "mask"))
        target, family = (
            R.Q.AXES.index(record["target"]),
            R.Q.FAMILIES.index(record["family"]),
        )
        terms = []
        for e, n, p in itertools.product((-1, 1), repeat=3):
            a = captures[lookup[c, o, e, n], record["mode"]]["selected_logits"]
            index = target * 6 + family * 2 + (p + 1) // 2
            sign = math.prod(s for bit, s in enumerate((e, n, p)) if mask & (1 << bit))
            terms.append(sign * (float(a[index, 0]) - float(a[index, 1])))
        error = abs(math.fsum(terms) / 8 - float(record["coefficient"]))
        max_formula = max(max_formula, error)
        if error > 1e-10:
            raise ValueError("Independent character sum failed")
    with (out / "actions.csv").open() as f:
        actions = list(csv.DictReader(f))
    for record in actions:
        c, o = int(record["condition"]), int(record["order"])
        target, family = (
            R.Q.AXES.index(record["target"]),
            R.Q.FAMILIES.index(record["family"]),
        )
        axis = record["axis"]
        x, y, signs = [], [], []
        for e, n, p in itertools.product((-1, 1), repeat=3):
            row_id, index = lookup[c, o, e, n], target * 6 + family * 2 + (p + 1) // 2

            def score(mode, row_id=row_id, index=index):
                a = captures[row_id, mode]["selected_logits"]
                return float(a[index, 0]) - float(a[index, 1])

            x.append(score(axis + "_" + record["variant"]) - score("natural"))
            y.append(score(axis + "_native") - score("natural"))
            signs.append((e if axis == "e" else n) * p)
        beta_x = math.fsum(a * s for a, s in zip(x, signs, strict=True)) / 8
        beta_y = math.fsum(a * s for a, s in zip(y, signs, strict=True)) / 8
        xx, yy = math.fsum(a * a for a in x), math.fsum(a * a for a in y)
        xy = math.fsum(a * b for a, b in zip(x, y, strict=True))
        expected = {
            "qualified_effect": beta_x,
            "native_qualified_effect": beta_y,
            "rms": math.sqrt(xx / 8),
            "signed_projection": xy / yy if yy else None,
        }
        for key, value in expected.items():
            observed = float(record[key]) if record[key] else None
            if value is None:
                if observed is not None:
                    raise ValueError("Undefined reference was silently scored")
            else:
                error = abs(observed - value)
                max_formula = max(max_formula, error)
                if error > 1e-10:
                    raise ValueError("Independent action account failed")
        if (record["same_direction"] == "True") != (beta_x * beta_y > 0):
            raise ValueError("Qualified action sign count failed")
    report = {
        "status": "PASS_ALTERNATE_FORMULA_NOT_BLIND",
        "batches": len(design),
        "characters": len(characters),
        "actions": len(actions),
        "max_native_head_error": max_native,
        "max_argument_error": max_argument,
        "max_formula_error": max_formula,
        "new_model_forwards": 0,
        "memory_transport": {},
    }
    for variant, primitive in itertools.product(
        ("source_full", "source_marginal", "source_reverse"),
        ("history_key", "history_value"),
    ):
        records = [
            r
            for r in memory_records
            if r["active_later"]
            and r["mode"].endswith(variant)
            and r["primitive"] == primitive
        ]
        report["memory_transport"][variant + "_" + primitive] = {
            "relative_difference_range": [
                min(r["relative_difference"] for r in records),
                max(r["relative_difference"] for r in records),
            ],
            "signed_projection_range": [
                min(r["signed_projection"] for r in records),
                max(r["signed_projection"] for r in records),
            ],
        }
    path = out / "memory_transport_description.csv"
    if not path.exists():
        with path.open("x", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(memory_records[0]))
            writer.writeheader()
            writer.writerows(memory_records)
    report["memory_description_sha256"] = R.H.digest(path)
    R.P.write_json(out / "alternate_audit.json", report)
    print(report)


if __name__ == "__main__":
    verify()
