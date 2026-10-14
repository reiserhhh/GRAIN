#!/usr/bin/env python3
"""Reconstruct actual entry arguments and all coefficients by alternate sums."""

from __future__ import annotations

import csv
import itertools
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_relation_entry_channels_hm158 as C


def verify(out=C.DEFAULT):
    """Check physical donors, CPU64 output and direct sign averages without fitting."""
    start, m = time.monotonic(), C.check(out)
    R = C.R
    C.authenticate_results(out)
    parent, followup = ROOT / m["parent"], ROOT / m["followup"]
    for root in (parent, followup):
        C.authenticate_results(root)
    old = C.inherited_captures(parent, followup)
    rows, queries = (R.R.read(parent / f"{name}.json") for name in ("rows", "queries"))
    source = R.R.arrays(parent / "source_operations.npz")
    weights = (
        R.R.OLD.weights()
        .cpu()[[queries["answers"][s][0] for s in ("Yes", "No")]]
        .numpy()
        .astype(float)
    )
    lookup = {(r["condition"], r["order"], r["e"], r["n"]): r["id"] for r in rows}
    captured = dict(old)
    max_donor, max_head, max_formula, max_control = 0.0, 0.0, 0.0, 0.0
    design = R.R.read(out / "design.json")
    for entry in design:
        if time.monotonic() - start > m["verification_seconds_cap"]:
            raise TimeoutError("Finite verification cap; no automatic renewal")
        row, mode = rows[entry["row"]], entry["mode"]
        a = R.R.arrays(out / "captures" / f"{entry['batch']:03d}.npz")
        captured[row["id"], mode] = a
        b = old[row["id"], "natural"]
        expected_slots = (
            list(range(len(row["history"])))
            if mode == "self"
            else row["scopes"]["neuroticism" if mode[0] == "e" else "extraversion"]
        )
        if a["patched_slots"].tolist() != expected_slots or a[
            "blocks"
        ].tolist() != list(range(28)):
            raise ValueError("Entry support/block count changed")
        if (
            not np.array_equal(a["prefix_states"], b["prefix_states"])
            or float(a["cache_spread"]) != 0
        ):
            raise ValueError("Prefix/history independence failed")
        for channel in ("key", "value"):
            name, saved = "history_" + channel, "donor_" + channel
            expected = b[name].copy()
            if mode != "self" and mode.endswith((channel, "joint_control")):
                x = source[f"{row['condition']}_{row['order']}_{name}"].astype(float)
                sign_weights = np.array([[1, -1], [-1, 1]], dtype=float)
                mixed = (
                    sum(
                        sign_weights[i, j] * x[i, j]
                        for i, j in itertools.product(range(2), repeat=2)
                    )
                    / 4
                )
                slots = expected_slots
                expected[:, :, slots] = (
                    expected[:, :, slots].astype(float)
                    - 2 * row["e"] * row["n"] * mixed[:, :, slots]
                ).astype(expected.dtype)
            error = float(np.max(abs(expected.astype(float) - a[saved].astype(float))))
            max_donor = max(max_donor, error)
            if error > 1e-6:
                raise ValueError(
                    "Actual entry donor differs from direct character construction"
                )
        expected_logits = a["states"].astype(float) @ weights.T
        max_head = max(
            max_head, float(np.max(abs(expected_logits - a["selected_logits"])))
        )
        if not np.allclose(expected_logits, a["selected_logits"], atol=1e-4, rtol=1e-5):
            raise ValueError("CPU64 W.h reconstruction failed")
        if mode == "self" or mode.endswith("joint_control"):
            reference = b if mode == "self" else old[row["id"], mode[0] + "_relation"]
            error = float(
                np.max(abs(a["selected_logits"] - reference["selected_logits"]))
            )
            max_control = max(max_control, error)
            if error > 1e-4:
                raise ValueError("Actual control no longer agrees")
    with (out / "components.csv").open() as f:
        records = list(csv.DictReader(f))
    for record in records:
        c, order, mask = (int(record[name]) for name in ("condition", "order", "mask"))
        axis, component = record["axis"], record["component"]
        target, family = (
            R.Q.AXES.index(record["target"]),
            R.Q.FAMILIES.index(record["family"]),
        )
        terms = []
        for e, n, p in itertools.product((-1, 1), repeat=3):
            row_id = lookup[c, order, e, n]
            index = target * 6 + family * 2 + (p + 1) // 2

            def score(mode, row_id=row_id, index=index):
                a = captured[row_id, mode]["selected_logits"]
                return float(a[index, 0]) - float(a[index, 1])

            base, k, v, joint, native = (
                score(mode)
                for mode in (
                    "natural",
                    axis + "_key",
                    axis + "_value",
                    axis + "_relation",
                    axis + "_native",
                )
            )
            value = {
                "key_entry": k - base,
                "value_entry": v - base,
                "recursive_interaction": joint - k - v + base,
                "joint_entry": joint - base,
                "native": native - base,
            }[component]
            weight = math.prod(
                s for bit, s in enumerate((e, n, p)) if mask & (1 << bit)
            )
            terms.append(weight * value)
        error = abs(math.fsum(terms) / 8 - float(record["coefficient"]))
        max_formula = max(max_formula, error)
        if error > 1e-10:
            raise ValueError("Direct entry coefficient reconstruction failed")
    summary = R.R.read(out / "summary.json")
    for qualified, section in ((True, "qualified"), (False, "all_characters")):
        active = [
            r
            for r in records
            if r["active_later"] == "True"
            and (not qualified or int(r["mask"]) == (5 if r["axis"] == "e" else 6))
        ]
        key = lambda r: (
            r["condition"],
            r["order"],
            r["axis"],
            r["mask"],
            r["target"],
            r["family"],
        )
        maps = {
            name: {
                key(r): float(r["coefficient"])
                for r in active
                if r["component"] == name
            }
            for name in (
                "key_entry",
                "value_entry",
                "recursive_interaction",
                "joint_entry",
                "native",
            )
        }
        for name, x in maps.items():
            if name == "native":
                continue
            for reference, ref_name in (
                ("joint_reference", "joint_entry"),
                ("native_reference", "native"),
            ):
                y = maps[ref_name]
                yy = math.fsum(a * a for a in y.values())
                expected = {
                    "count": len(x),
                    "rms": math.sqrt(math.fsum(a * a for a in x.values()) / len(x)),
                    "reference_rms": math.sqrt(yy / len(x)),
                    "same_reference_direction": sum(x[k] * y[k] > 0 for k in x),
                    "signed_projection": math.fsum(x[k] * y[k] for k in x) / yy
                    if yy
                    else None,
                    "relative_difference": math.sqrt(
                        math.fsum((x[k] - y[k]) ** 2 for k in x) / yy
                    )
                    if yy
                    else None,
                }
                for metric, v in expected.items():
                    actual = summary[section][name][reference][metric]
                    if v is None:
                        if actual is not None:
                            raise ValueError("Null reference was scored")
                    else:
                        max_formula = max(max_formula, abs(actual - v))
                        if abs(actual - v) > 1e-10:
                            raise ValueError("Alternate action summary differs")
    result = {
        "status": "PASS_ALTERNATE_FORMULA_NOT_BLIND",
        "actual_batches": len(design),
        "complete_component_rows": len(records),
        "max_donor_error": max_donor,
        "max_native_head_error": max_head,
        "max_formula_error": max_formula,
        "max_control_error": max_control,
        "new_model_forwards": 0,
        "seconds": time.monotonic() - start,
    }
    R.P.write_json(out / "alternate_audit.json", result)
    print(result)


if __name__ == "__main__":
    verify()
