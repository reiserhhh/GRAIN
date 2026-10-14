#!/usr/bin/env python3
"""Alternate scalar checks of actual HM48 binding, token bags and loss floors."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_evaluative_routing import load_arrays, write
from scripts.run_grain_math_relational_appraisal import OUT, bag_key, check, cube_key
from scripts.verify_grain_math_evaluative_routing import native_score


def read_csv(path):
    """Read complete result rows for numerical, rather than existence, checks."""
    with path.open() as stream:
        return list(csv.DictReader(stream))


def verify(out):
    """Rebuild all scores, signed coefficients, target contrasts and group loss bounds."""
    manifest = check(out)
    status = read_json(out / "run_status.json")
    if (
        status["status"] != "COLLECTED"
        or status["in_flight"] is not None
        or any(
            status[k] != 1728
            for k in ("forward_calls", "attempted_forward_calls", "completed_units")
        )
    ):
        raise ValueError("Actual complete1728-score collection required")
    for name, sha in read_json(out / "03/artifact_hashes.json").items():
        if digest(out / name) != sha:
            raise ValueError("Changed actual output " + name)
    for name, sha in read_json(out / "01/source_barrier.json")["files"].items():
        if digest(out / name) != sha:
            raise ValueError("Changed sealed source")
    scores, main, absent, bags, token_bags, errors = (
        {},
        {},
        {},
        defaultdict(list),
        {},
        [],
    )
    for row in read_json(out / "00/design.json"):
        meta, _ = load_arrays(out / f"01/units/unit_{row['unit']:04d}.json")
        if meta["unit"] != row["unit"] or meta["role"] != row["role"]:
            raise ValueError("Wrong unit identity")
        y, mass = native_score(meta["native"])
        errors.extend(
            (
                abs(y - meta["native"]["conditional_score"]),
                abs(mass - meta["native"]["candidate_mass"]),
            )
        )
        scores[row["unit"]] = y
        k, b, o, r = (
            cube_key(row),
            row["behaviour"],
            row["order"],
            int(row["person"] == "B"),
        )
        if row["appraisal"] == "none":
            absent.setdefault(k, {})[b, o, r] = y
        else:
            main.setdefault(k, {})[b, int(row["appraisal"] == "B"), o, r] = y
        bk, tokens = bag_key(row), Counter(row["input_ids"])
        if bk in token_bags and token_bags[bk] != tokens:
            raise ValueError("Actual token bag equality does not hold")
        token_bags[bk] = tokens
        bags[bk].append(y)
    if (
        len(main) != 72
        or len(absent) != 72
        or len(bags) != 288
        or any(len(t) != 16 for t in main.values())
        or any(len(t) != 8 for t in absent.values())
    ):
        raise ValueError("Incomplete finite scope")
    for row in read_csv(out / "03/readouts.csv"):
        errors.append(abs(float(row["score"]) - scores[int(row["unit"])]))
    coeffs = {}
    for row in read_csv(out / "03/coefficients.csv"):
        key = row["role"], row["content"], row["query"]
        table = main[key] if row["presence"] == "present" else absent[key]
        mask = tuple(map(int, row["mask"]))
        value = sum(
            y * (-1) ** sum(a * b for a, b in zip(mask, cell, strict=True))
            for cell, y in table.items()
        ) / len(table)
        errors.append(abs(value - float(row["value"])))
        coeffs[key + (row["presence"], row["mask"])] = value
    lookup = {
        "present": {
            "behaviour_bound": "1001",
            "behaviour_order_shared": "1010",
            "query_order": "0011",
            "appraisal_bound": "0101",
            "appraisal_order_shared": "0110",
        },
        "absent": {
            "behaviour_bound": "101",
            "behaviour_order_shared": "110",
            "query_order": "011",
        },
    }
    for filename in ("contrasts.csv", "response_matrices.csv"):
        for row in read_csv(out / "03" / filename):
            mask = lookup[row["presence"]][row["channel"]]
            value = (
                2
                * coeffs[
                    row["role"], row["content"], row["query"], row["presence"], mask
                ]
            )
            errors.append(abs(value - float(row["value"])))
    for row in read_csv(out / "03/token_bag_bounds.csv"):
        y = bags[
            row["role"],
            row["content"],
            row["query"],
            row["person"],
            row["presence"] == "absent",
        ]
        mean = sum(y) / len(y)
        mse = sum((t - mean) ** 2 for t in y) / len(y)
        for field, value in (
            ("oracle_group_mean", mean),
            ("minimum_mse", mse),
            ("minimum_rmse", math.sqrt(mse)),
            ("range", max(y) - min(y)),
        ):
            errors.append(abs(value - float(row[field])))
        for constant in (0, 0.5, 1):
            errors.append(
                abs(
                    sum((t - constant) ** 2 for t in y) / len(y)
                    - mse
                    - (mean - constant) ** 2
                )
            )
    for row in read_csv(out / "03/group_diagnostics.csv"):
        key = row["role"], row["content"], row["query"]
        table = main[key] if row["presence"] == "present" else absent[key]
        for cell, y in table.items():
            value = sum(
                v
                * (-1)
                ** sum(a * b for a, b in zip(tuple(map(int, k[-1])), cell, strict=True))
                for k, v in coeffs.items()
                if k[:3] == key and k[3] == row["presence"]
            )
            errors.append(abs(y - value))
        rename_rms = math.sqrt(
            sum(
                (y - table[tuple(1 - b for b in cell)]) ** 2
                for cell, y in table.items()
            )
            / len(table)
        )
        errors.append(abs(rename_rms - float(row["renaming_rms"])))
    for group in read_json(out / "03/summary.json")["group_summary"]:
        mask = lookup[group["presence"]][group["channel"]]
        matrix = [
            [
                2 * coeffs[group["role"], c, q, group["presence"], mask]
                for q in manifest["spec"]["targets"]
            ]
            for c in manifest["spec"]["targets"]
        ]
        errors.append(
            abs(
                math.sqrt(sum(v * v for row in matrix for v in row) / 36) - group["rms"]
            )
        )
        errors.extend(
            abs(x - y)
            for row, expected in zip(matrix, group["matrix"], strict=True)
            for x, y in zip(row, expected, strict=True)
        )
    expected_counts = {
        "readouts.csv": 1728,
        "coefficients.csv": 1728,
        "token_bag_bounds.csv": 288,
        "group_diagnostics.csv": 144,
        "contrasts.csv": 576,
        "response_matrices.csv": 576,
        "carrier_changes.csv": 3456,
    }
    for name, count in expected_counts.items():
        if len(read_csv(out / "03" / name)) != count:
            raise ValueError("Wrong full output scope " + name)
    if max(errors) > 1e-12:
        raise ValueError("Alternate finite arithmetic failed")
    result = {
        "status": "VERIFIED_FINITE_HM48_NOT_GLOBAL",
        "actual_scores": len(scores),
        "exact_token_bag_classes": len(bags),
        "maximum_alternate_scalar_error": max(errors),
        "checked_rows": expected_counts,
        "source_scores_sealed_before_application": 864,
        "same_agent_alternate_arithmetic_not_neural_replication": True,
        "new_model_calls": 0,
        "original_goal_complete": False,
    }
    write(out / "verification.json", result)
    print(result)


def main():
    """Check one finished packet without a new forward, fitting or dispatch."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    verify(parser.parse_args().out)


if __name__ == "__main__":
    main()
