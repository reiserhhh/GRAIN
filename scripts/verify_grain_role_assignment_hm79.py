#!/usr/bin/env python3
"""HM79 independent finite sums, raw bindings and observation consequences."""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import read_json
from scripts.run_grain_math_construct_routing import write
from scripts.run_grain_role_assignment_hm79 import OUT, check, raw_tables, seal_check


def verify(out):
    """Recalculate every coefficient and table consequence by a separate formula."""
    manifest = check(out)
    seal_check(out, out / "03/seal.json")
    tables, raw_checks = raw_tables(out)
    summary = read_json(out / "03/summary.json")
    points = np.asarray(list(itertools.product((-1, 1), repeat=4)), dtype=float)
    masks = list(itertools.product((0, 1), repeat=4))
    old_points = points[points[:, 3] == -points[:, 2]]
    old_masks = list(itertools.product((0, 1), repeat=3))
    error, comparisons, old_replays = 0.0, 0, 0

    def compare(a, b):
        nonlocal error, comparisons
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("Mismatched/nonfinite comparison")
        error = max(error, float(np.max(np.abs(a - b))))
        comparisons += a.size

    independent = {}
    with np.load(out / "03/tables_and_coefficients.npz", allow_pickle=False) as saved:
        for context, (key, table) in enumerate(sorted(tables.items())):
            if list(key) != summary["context_keys"][context]:
                raise ValueError("Context identity mismatch")
            for stage in summary["stages"]:
                values = table[stage]
                compare(values, saved[f"values_{context:02d}_{stage}"])
                coefficients = []
                for mask in masks:
                    sign = np.ones(16)
                    for bit in range(4):
                        if mask[bit]:
                            sign *= points[:, bit]
                    coefficients.append(
                        sum(sign[i] * values[i] for i in range(16)) / 16
                    )
                c = np.asarray(coefficients)
                compare(c, saved[f"coefficients_{context:02d}_{stage}"])
                independent[(context, stage)] = c
                old = values[points[:, 3] == -points[:, 2]]
                for i, mask in enumerate(old_masks):
                    sign = np.ones(8)
                    for bit in range(3):
                        if mask[bit]:
                            sign *= old_points[:, bit]
                    direct = sum(sign[j] * old[j] for j in range(8)) / 8
                    a, b, g = mask
                    index0 = 8 * a + 4 * b + 2 * g
                    index1 = 8 * a + 4 * b + 2 * (1 - g) + 1
                    compare(direct, c[index0] - c[index1])
                    old_replays += direct.size
                t = values.reshape(2, 2, 2, 2, *values.shape[1:])
                mixed = t[1, 1] - t[1, 0] - t[0, 1] + t[0, 0]
                for gi, g in enumerate((-1, 1)):
                    for hi, h in enumerate((-1, 1)):
                        compare(
                            mixed[gi, hi],
                            4 * (c[12] + g * c[14] + h * c[13] + g * h * c[15]),
                        )
                    compare(mixed[gi, 1] - mixed[gi, 0], 8 * (c[13] + g * c[15]))
                for hi, h in enumerate((-1, 1)):
                    compare(mixed[1, hi] - mixed[0, hi], 8 * (c[14] + h * c[15]))
                compare(mixed[1, 0] - mixed[0, 1], 8 * (c[14] - c[13]))
                compare(
                    (c[14] - c[13]) ** 2, c[14] ** 2 + c[13] ** 2 - 2 * c[14] * c[13]
                )
                character = np.prod(points, axis=1)
                epsilon = manifest["spec"]["profile_epsilon"]
                plus, minus = (
                    (1 + epsilon * character) / 16,
                    (1 - epsilon * character) / 16,
                )
                compare(
                    plus @ values.reshape(16, -1) - minus @ values.reshape(16, -1),
                    (2 * epsilon * c[15]).reshape(-1),
                )
    targets = manifest["spec"]["targets"]

    def channel(row):
        return 0 if row["query"] == "query_independent" else targets.index(row["query"])

    for filename in (
        "coefficients",
        "conditional_effects",
        "alias_energy_accounts",
        "author_composition_witnesses",
    ):
        with (out / f"03/{filename}.csv").open(newline="") as stream:
            for row in csv.DictReader(stream):
                context, stage, q = int(row["context"]), row["stage"], channel(row)
                c = independent[(context, stage)]
                key = summary["context_keys"][context]
                if (
                    int(row["family"]) != key[0]
                    or row["focal"] != key[1]
                    or row["companion"] != key[2]
                ):
                    raise ValueError("CSV context attribution error")
                if filename == "coefficients":
                    mask = tuple(int(v in row["term"]) for v in "xygh")
                    index = masks.index(mask) if row["term"] != "1" else 0
                    value = c[index, q]
                    compare(np.asarray(float(row["rms"])), np.sqrt(np.mean(value**2)))
                    if row["scalar"]:
                        compare(np.asarray(float(row["scalar"])), value[0])
                elif filename == "conditional_effects":
                    value = (
                        8 * (c[14, q] + int(row["h"]) * c[15, q])
                        if row["edited_role"] == "companion"
                        else 8 * (c[13, q] + int(row["g"]) * c[15, q])
                    )
                    compare(
                        np.asarray(float(row["gate_rms"])), np.sqrt(np.mean(value**2))
                    )
                    if row["gate_scalar"]:
                        compare(np.asarray(float(row["gate_scalar"])), value[0])
                elif filename == "alias_energy_accounts":
                    values = {
                        "old_energy": np.mean((c[14, q] - c[13, q]) ** 2),
                        "companion_energy": np.mean(c[14, q] ** 2),
                        "anchor_energy": np.mean(c[13, q] ** 2),
                        "signed_cross_term": -2 * np.mean(c[14, q] * c[13, q]),
                        "four_way_energy": np.mean(c[15, q] ** 2),
                    }
                    for field, value in values.items():
                        compare(np.asarray(float(row[field])), value)
                else:
                    value = 2 * manifest["spec"]["profile_epsilon"] * c[15, q]
                    compare(
                        np.asarray(float(row["difference_rms"])),
                        np.sqrt(np.mean(value**2)),
                    )
                    if row["difference_scalar"]:
                        compare(np.asarray(float(row["difference_scalar"])), value[0])
    for row in summary["summary_by_group"]:
        selected = []
        for context, key in enumerate(summary["context_keys"]):
            if key[0] != row["family"]:
                continue
            queries = (
                ["query_independent"] if row["stage"] == "body_endpoint" else targets
            )
            for q, query in enumerate(queries):
                group = (
                    "query_independent"
                    if query == "query_independent"
                    else "focal_query"
                    if query == key[1]
                    else "companion_query"
                    if query == key[2]
                    else "other_queries"
                )
                if row["group"] not in (group, "all_queries"):
                    continue
                c = independent[(context, row["stage"])][:, q]
                selected.append(
                    {
                        "old_gate": 8 * (c[14] - c[13]),
                        "companion_only": 8 * c[14],
                        "anchor_only": 8 * c[13],
                        "joint_modulation": 8 * c[15],
                        "companion_h_low": 8 * (c[14] - c[15]),
                        "companion_h_high": 8 * (c[14] + c[15]),
                        "anchor_g_low": 8 * (c[13] - c[15]),
                        "anchor_g_high": 8 * (c[13] + c[15]),
                        "companion_main_role": 4 * c[6],
                        "anchor_main_role": 4 * c[5],
                    }
                )
        if len(selected) != row["contexts_queries"]:
            raise ValueError("Summary scope mismatch")
        for field in selected[0]:
            v = np.asarray([r[field] for r in selected])
            compare(np.asarray(row[field + "_rms"]), np.sqrt(np.mean(v**2)))
            if row[field + "_mean"] is not None:
                compare(np.asarray(row[field + "_mean"]), np.asarray(v.mean()))
        a, b = (
            np.asarray([r[field] for r in selected])
            for field in ("companion_only", "anchor_only")
        )
        compare(np.asarray(row["signed_cross_term"]), np.asarray(-2 * np.mean(a * b)))
    status = read_json(out / "run_status.json")
    if (
        any(
            status[k] != manifest["spec"]["max_forwards"]
            for k in ("forward_calls", "attempted_forward_calls", "completed_units")
        )
        or status["in_flight"] is not None
    ):
        raise ValueError("Actual collector count incomplete")
    if error > manifest["spec"]["arithmetic_tolerance"]:
        raise ValueError(f"Independent numerical discrepancy {error}")
    result = {
        "status": "VERIFIED_FINITE_HM79_NOT_GLOBAL",
        "actual_new_forwards": status["forward_calls"],
        "reused_actual_readings": manifest["spec"]["expected_reused_readings"],
        "actual_scalar_comparisons": comparisons,
        "old_restriction_checks": old_replays,
        "maximum_alternate_error": error,
        **raw_checks,
        "same_agent_alternate_calculation_not_third_party": True,
        "original_scope_complete": False,
    }
    write(out / "verification.json", result)
    print(result, flush=True)


def main():
    """Verify a completed frozen packet without new model calls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    verify(parser.parse_args().out)


if __name__ == "__main__":
    main()
