#!/usr/bin/env python3
"""Alternate arithmetic for source updates, defects and complete endpoint tables."""

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
from scripts import run_grain_source_update_support_hm162 as U


def rows_csv(path):
    """Read an immutable numeric result table without changing scientific output."""
    with path.open() as f:
        return list(csv.DictReader(f))


def direct_increment(cube, e, n, axis, operation):
    """Use direct corner differences and signed enumeration, not production code."""
    x = np.asarray(cube, float)
    i, j = (e + 1) // 2, (n + 1) // 2
    target = x[1 - i, j] if axis == "e" else x[i, 1 - j]
    full = target - x[i, j]
    mixed = (
        sum(
            a * b * x[(a + 1) // 2, (b + 1) // 2]
            for a, b in itertools.product((-1, 1), repeat=2)
        )
        / 4
    )
    used = -2 * e * n * mixed if operation.startswith("relation") else full
    return full, used


def direct_slots(row, axis, operation):
    """Compute time support independently from original clause addresses."""
    own = "extraversion" if axis == "e" else "neuroticism"
    other = "neuroticism" if axis == "e" else "extraversion"
    if operation.endswith("other"):
        return row["scopes"][other]
    return list(range(min(row["scopes"][own]), len(row["history"])))


def verify(out=U.DEFAULT):
    """Recompute actual donors, head projections and all finite table entries."""
    start, m = time.monotonic(), U.check(out)
    R, C, E = U.R, U.C, U.E
    E.G.authenticate_results(out)
    parent = ROOT / m["parent"]
    base = U.natural_captures(parent)
    rows, queries, design = (
        R.read(out / f"{name}.json") for name in ("rows", "queries", "design")
    )
    if len(design) != 100 or len(queries["definitions"]) != 16:
        raise ValueError("Registered100batch/16endpoint support missing")
    source = R.arrays(ROOT / m["source_operations"])
    weights = (
        R.OLD.weights()
        .cpu()[[queries["answers"][s][0] for s in ("Yes", "No")]]
        .numpy()
        .astype(float)
    )
    donor_error = head_error = prefix_error = control_error = probability_error = (
        character_error
    ) = 0.0
    observed, scores = {}, {}
    inherited = {}
    for root in (parent, out):
        for d in R.read(root / "design.json"):
            if time.monotonic() - start > m["verification_seconds_cap"]:
                raise TimeoutError("Finite180s audit cap, no renewal")
            if root == parent and d["mode"] == "self":
                continue
            mode = (
                d["mode"] + "_other"
                if root == parent and d["mode"].endswith("_relation")
                else d["mode"]
            )
            a = R.arrays(root / "captures" / f"{d['batch']:03d}.npz")
            observed[d["row"], mode] = d
            score = {key: [] for key in E.M.MEASURES}
            for logits, z in zip(a["selected_logits"], a["normalizer"], strict=True):
                y, n, z = float(logits[0]), float(logits[1]), float(z)
                py, pn = math.exp(y - z), math.exp(n - z)
                for k, value in (
                    ("log_odds", y - n),
                    ("restricted_yes", py / (py + pn)),
                    ("full_yes", py),
                    ("answer_mass", py + pn),
                ):
                    score[k].append(value)
            scores[d["row"], mode] = score
            if root == parent:
                if d["mode"].endswith("_relation"):
                    inherited[d["row"], mode] = a["selected_logits"]
                continue
            row = rows[d["row"]]
            native = base[d["row"]]
            prefix_error = max(
                prefix_error,
                float(np.max(abs(a["prefix_states"] - native["prefix_states"]))),
            )
            projected = a["states"].astype(float) @ weights.T
            head_error = max(
                head_error, float(np.max(abs(projected - a["selected_logits"])))
            )
            if not np.allclose(projected, a["selected_logits"], atol=1e-4, rtol=1e-5):
                raise ValueError("CPU64 selected head reconstruction failed")
            if (
                a["blocks"].tolist() != list(range(28))
                or float(a["cache_spread"]) > 1e-4
            ):
                raise ValueError("Causal support changed across suffixes")
            operation = "relation_other" if mode.endswith("replay") else mode[2:]
            slots = (
                list(range(len(row["history"])))
                if mode == "self"
                else direct_slots(row, mode[0], operation)
            )
            if a["patched_slots"].tolist() != slots:
                raise ValueError(
                    "Actual patch support differs from registered addresses"
                )
            for channel in ("key", "value"):
                name = "history_" + channel
                expected = native[name].copy()
                if mode != "self":
                    _, delta = direct_increment(
                        source[f"{row['condition']}_{row['order']}_{name}"],
                        row["e"],
                        row["n"],
                        mode[0],
                        operation,
                    )
                    expected[:, :, slots] = (
                        expected[:, :, slots].astype(float) + delta[:, :, slots]
                    ).astype(expected.dtype)
                donor_error = max(
                    donor_error,
                    float(
                        np.max(
                            abs(
                                expected.astype(float)
                                - a["donor_" + channel].astype(float)
                            )
                        )
                    ),
                )
            if mode == "self" or mode.endswith("replay"):
                reference = (
                    native["selected_logits"]
                    if mode == "self"
                    else inherited[d["row"], mode[0] + "_relation_other"]
                )
                control_error = max(
                    control_error, float(np.max(abs(a["selected_logits"] - reference)))
                )
    if donor_error > 1e-6 or max(prefix_error, control_error) > 1e-4:
        raise ValueError("Donor or historical-prefix/control audit failed")
    endpoints = rows_csv(out / "endpoints.csv")
    if len(endpoints) != 2368:
        raise ValueError("New1600 plus inherited768 endpoint records required")
    for r in endpoints:
        qi = queries["definitions"].index(
            {"axis": r["axis"], "family": r["family"], "person": int(r["person"])}
        )
        for measure in E.M.MEASURES:
            probability_error = max(
                probability_error,
                abs(float(r[measure]) - scores[int(r["row"]), r["mode"]][measure][qi]),
            )
    indexed = {(r["condition"], r["order"], r["e"], r["n"]): r["id"] for r in rows}
    chars = rows_csv(out / "characters.csv")
    if len(chars) != 12288:
        raise ValueError("All12288 endpoint-character rows required")
    for r in chars:
        c, o, mask = (int(r[k]) for k in ("condition", "order", "mask"))
        axis, operation, measure = (r[k] for k in ("axis", "operation", "measure"))
        terms = []
        for e, n, p in itertools.product((-1, 1), repeat=3):
            rid = indexed[c, o, e, n]
            qi = queries["definitions"].index(
                {"axis": r["target"], "family": r["family"], "person": p}
            )
            b = scores[rid, "natural"][measure][qi]
            if operation == "natural":
                partner = indexed[
                    c, o, -e if axis == "e" else e, -n if axis == "n" else n
                ]
                value = scores[partner, "natural"][measure][qi] - b
            elif operation == "factorial_interaction":
                value = sum(
                    sign * scores[rid, axis + "_" + op][measure][qi]
                    for op, sign in (
                        ("full_closure", 1),
                        ("full_other", -1),
                        ("relation_closure", -1),
                        ("relation_other", 1),
                    )
                )
            else:
                value = scores[rid, axis + "_" + operation][measure][qi] - b
            weight = math.prod(s for j, s in enumerate((e, n, p)) if mask & (1 << j))
            terms.append(weight * value)
        character_error = max(
            character_error, abs(float(r["coefficient"]) - math.fsum(terms) / 8)
        )
    defect_rows = rows_csv(out / "memory_defects.csv")
    if len(defect_rows) != 256:
        raise ValueError("Every256 component/support/channel defect required")
    defect_identity_error = energy_error = source_upstream_error = 0.0
    for r in defect_rows:
        row, axis, op, channel = (
            rows[int(r["row"])],
            r["axis"],
            r["operation"],
            r["channel"],
        )
        name = "history_" + channel
        cube = source[f"{row['condition']}_{row['order']}_{name}"].astype(float)
        full, delta = direct_increment(cube, row["e"], row["n"], axis, op)
        slots = direct_slots(row, axis, op)
        b = base[row["id"]][name].astype(float)
        pid = indexed[
            row["condition"],
            row["order"],
            -row["e"] if axis == "e" else row["e"],
            -row["n"] if axis == "n" else row["n"],
        ]
        target = base[pid][name].astype(float)
        context = b - target + full
        support = -full.copy()
        support[:, :, slots] = 0
        component = np.zeros_like(full)
        component[:, :, slots] = (delta - full)[:, :, slots]
        total = b - target
        total[:, :, slots] += delta[:, :, slots]
        defect_identity_error = max(
            defect_identity_error,
            float(np.max(abs(context + support + component - total))),
        )
        parts = {
            "context": context,
            "omitted_support": support,
            "omitted_component": component,
        }
        expected = {k + "_energy": float(np.sum(v * v)) for k, v in parts.items()}
        expected["total_energy"] = float(np.sum(total * total))
        for first, second in itertools.combinations(parts, 2):
            expected[first + "_x_" + second] = float(
                2 * np.sum(parts[first] * parts[second])
            )
        for k, v in expected.items():
            actual = float(r[k])
            energy_error = max(energy_error, abs(v - actual))
            if not np.isclose(v, actual, rtol=1e-10, atol=1e-8):
                raise ValueError("Defect energy/cross-term reconstruction failed")
        closure = direct_slots(row, axis, "full_closure")
        outside = sorted(set(range(full.shape[2])) - set(closure))
        if outside:
            source_upstream_error = max(
                source_upstream_error, float(np.max(abs(full[:, :, outside])))
            )
    if (
        max(probability_error, character_error, defect_identity_error) > 1e-10
        or source_upstream_error > 1e-6
    ):
        raise ValueError("Finite formula or source-causal support check failed")
    report = {
        "status": "NUMERICAL_AUDIT_PASS_NOT_PSYCHOLOGICAL_VALIDITY",
        "actual_new_batches": 100,
        "endpoint_rows": len(endpoints),
        "character_rows": len(chars),
        "memory_defect_rows": len(defect_rows),
        "max_donor_error": donor_error,
        "max_head_error": head_error,
        "max_prefix_error": prefix_error,
        "max_control_error": control_error,
        "max_probability_error": probability_error,
        "max_character_error": character_error,
        "max_defect_identity_error": defect_identity_error,
        "max_energy_absolute_error": energy_error,
        "max_source_upstream_error": source_upstream_error,
        "seconds": time.monotonic() - start,
        "same_agent_alternate_arithmetic_not_blind": True,
        "original_scope_complete": False,
        "new_decoder_calls": 0,
    }
    if not (out / "alternate_audit.json").exists():
        C.P.write_json(out / "alternate_audit.json", report)
    print(report, flush=True)
    return report


if __name__ == "__main__":
    verify()
