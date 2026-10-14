#!/usr/bin/env python3
"""Audit relation-only donor arithmetic and complete component contrasts."""

from __future__ import annotations

import csv
import itertools
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_carrier_relation_followup_hm157 as F


def verify(out=F.DEFAULT):
    """Use the direct mixed character, not the collection edit helper."""
    m = F.check(out)
    parent, R = ROOT / m["parent"], F.R
    for seal in ("capture_seal.json", "output_seal.json"):
        for name, checksum in R.R.read(out / seal).items():
            if R.H.digest(out / name) != checksum:
                raise ValueError("Follow-up sealed result changed")
    source = R.R.arrays(parent / "source_operations.npz")
    rows = R.R.read(parent / "rows.json")
    lookup = {(r["condition"], r["order"], r["e"], r["n"]): r["id"] for r in rows}
    captures = {}
    for root in (parent, out):
        for entry in R.R.read(root / "design.json"):
            captures[entry["row"], entry["mode"]] = R.R.arrays(
                root / "captures" / f"{entry['batch']:03d}.npz"
            )
    queries = R.R.read(parent / "queries.json")
    weights = (
        R.R.OLD.weights()
        .cpu()[[queries["answers"][s][0] for s in ("Yes", "No")]]
        .numpy()
        .astype(float)
    )
    max_donor, max_formula, max_native = 0.0, 0.0, 0.0
    for entry in R.R.read(out / "design.json"):
        row, mode = rows[entry["row"]], entry["mode"]
        a, base = captures[row["id"], mode], captures[row["id"], "natural"]
        logits = a["states"].astype(float) @ weights.T
        max_native = max(max_native, float(np.max(abs(logits - a["selected_logits"]))))
        if not np.allclose(logits, a["selected_logits"], atol=1e-4, rtol=1e-5):
            raise ValueError("Follow-up CPU64 output-head reconstruction failed")
        if (
            not np.array_equal(a["prefix_states"], base["prefix_states"])
            or float(a["cache_spread"]) != 0
        ):
            raise ValueError("Follow-up observation guard failed")
        slots = a["patched_slots"].tolist()
        expected_slots = (
            list(range(len(row["history"])))
            if mode == "self"
            else row["scopes"][
                "neuroticism" if mode.startswith("e_") else "extraversion"
            ]
        )
        if slots != expected_slots:
            raise ValueError("Follow-up carrier support changed")
        for name, saved in (
            ("history_key", "donor_key"),
            ("history_value", "donor_value"),
        ):
            expected = base[name].copy()
            if mode != "self":
                x = source[f"{row['condition']}_{row['order']}_{name}"].astype(float)
                mixed = (x[0, 0] - x[0, 1] - x[1, 0] + x[1, 1]) / 4
                delta = -2 * row["e"] * row["n"] * mixed
                expected[:, :, slots] = (
                    expected[:, :, slots].astype(float) + delta[:, :, slots]
                ).astype(expected.dtype)
            err = float(np.max(abs(expected.astype(float) - a[saved])))
            max_donor = max(max_donor, err)
            if err > 1e-6:
                raise ValueError("Direct mixed-character donor check failed")
    with (out / "components.csv").open() as f:
        records = list(csv.DictReader(f))
    for record in records:
        c, o, mask = (int(record[k]) for k in ("condition", "order", "mask"))
        axis, component = record["axis"], record["component"]
        target, family = (
            R.Q.AXES.index(record["target"]),
            R.Q.FAMILIES.index(record["family"]),
        )
        terms = []
        for e, n, p in itertools.product((-1, 1), repeat=3):
            row_id, index = lookup[c, o, e, n], target * 6 + family * 2 + (p + 1) // 2

            def score(mode, row_id=row_id, index=index):
                a = captures[row_id, mode]["selected_logits"]
                return float(a[index, 0]) - float(a[index, 1])

            b = score("natural")
            marginal = score(axis + "_source_marginal") - b
            relation = score(axis + "_relation") - b
            full = score(axis + "_source_full") - b
            native = score(axis + "_native") - b
            v = {
                "marginal": marginal,
                "relation": relation,
                "full": full,
                "native": native,
                "interaction": full - marginal - relation,
            }[component]
            sign = math.prod(s for bit, s in enumerate((e, n, p)) if mask & (1 << bit))
            terms.append(sign * v)
        error = abs(math.fsum(terms) / 8 - float(record["coefficient"]))
        max_formula = max(max_formula, error)
        if error > 1e-10:
            raise ValueError("Direct endpoint component check failed")
    result = {
        "status": "PASS_ALTERNATE_FORMULA_NOT_BLIND",
        "actual_batches": 34,
        "complete_component_rows": len(records),
        "max_donor_error": max_donor,
        "max_formula_error": max_formula,
        "max_native_head_error": max_native,
        "new_model_forwards": 0,
    }
    R.P.write_json(out / "alternate_audit.json", result)
    print(result)


if __name__ == "__main__":
    verify()
