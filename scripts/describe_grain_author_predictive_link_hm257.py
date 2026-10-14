#!/usr/bin/env python3
"""Describe every frozen target/link direction and token quarter without refit."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_author_predictive_link as M
from scripts import run_grain_author_predictive_link_hm257 as P
from scripts import run_grain_observed_family_hm170 as R


def describe(fields, choice, actions, lengths, token_rows):
    """Return complete signed accounts; fixed quarters are post-outcome summaries."""
    fields, choice, lengths = map(np.asarray, (fields, choice, lengths))
    if (
        fields.shape != (48, 5)
        or choice.shape != fields.shape
        or lengths.shape != (48,)
    ):
        raise ValueError("The complete addressed author family is required")
    directions, denominator = [], []
    for operation in ("stranger", "order", "carry"):
        for window in (0, 1):
            ids = np.arange(24) + 24 * window
            for part, f in (
                ("full", fields),
                ("choice", choice),
                ("within", fields - choice),
            ):
                for scale, y in (
                    ("sum", actions[operation]),
                    ("per_token", actions[operation] / lengths[:, None]),
                ):
                    matrix = M.contact(f[ids], y[ids])
                    for i, source in enumerate(M.TARGETS):
                        for j, target in enumerate(M.TARGETS):
                            directions.append(
                                {
                                    "operation": operation,
                                    "window": window,
                                    "part": part,
                                    "scale": scale,
                                    "source_target": source,
                                    "observer_target": target,
                                    "contact": float(matrix[i, j]),
                                }
                            )
    bands = {}
    terms = {}
    seen = set()
    for row in token_rows:
        unit, t = int(row["unit"]), int(row["token_position"])
        if unit not in range(48) or t not in range(int(lengths[unit])):
            raise ValueError("Unaddressed token record")
        operation, target = row["operation"], row["target"]
        key = (unit, operation, target, t)
        if key in seen:
            raise ValueError("Duplicate token would inflate the evidence")
        seen.add(key)
        quarter = 4 * t // int(lengths[unit])
        base = (unit, operation, target)
        values = np.array([float(row[k]) for k in ("numerator", "normalizer", "nats")])
        terms.setdefault(base, np.zeros(3))[:] += values
        bands.setdefault((*base, quarter), np.zeros(3))[:] += values
    if len(seen) != int(lengths.sum()) * 15:
        raise ValueError("The complete token family must survive description")
    quarter_rows, reconstruction = [], 0.0
    for unit in range(48):
        for operation in ("stranger", "order", "carry"):
            for q, target in enumerate(M.TARGETS):
                base = (unit, operation, target)
                total = terms[base]
                reconstruction = max(
                    reconstruction,
                    abs(total[0] - total[1] - total[2]),
                    abs(total[2] - actions[operation][unit, q]),
                )
                for quarter in range(4):
                    values = bands[(*base, quarter)]
                    quarter_rows.append(
                        {
                            "unit": unit,
                            "window": unit // 24,
                            "operation": operation,
                            "target": target,
                            "quarter": quarter,
                            "numerator": float(values[0]),
                            "normalizer": float(values[1]),
                            "nats": float(values[2]),
                        }
                    )
    for operation in ("stranger", "order", "carry"):
        for window in (0, 1):
            for target in M.TARGETS:
                values = np.array(
                    [
                        terms[u, operation, target]
                        for u in range(24 * window, 24 * (window + 1))
                    ]
                )
                mean = values.mean(0)
                denominator.append(
                    {
                        "operation": operation,
                        "window": window,
                        "target": target,
                        "numerator_mean": float(mean[0]),
                        "normalizer_mean": float(mean[1]),
                        "net_mean": float(mean[2]),
                        "normalizer_changes_mean_sign": bool(mean[0] * mean[2] < 0),
                        "normalizer_changes_unit_signs": int(
                            (values[:, 0] * values[:, 2] < 0).sum()
                        ),
                    }
                )
    return directions, denominator, quarter_rows, reconstruction


def main():
    """Append a separately sealed description; no model or parent edits occur."""
    parent = P.DEFAULT
    R.check(
        parent, ("collected_seal.json", "analyzed_seal.json", "verification_seal.json")
    )
    out = parent / "post_outcome_description_r1"
    out.mkdir(exist_ok=False)
    inputs = (
        "source_observations.npz",
        "actions.npz",
        "cases.json",
        "token_accounts.csv",
    )
    R.write(
        out / "manifest.json",
        {
            "post_outcome_not_preregistered": True,
            "no_fit_selection_or_model_calls": True,
            "code": {
                "scripts/describe_grain_author_predictive_link_hm257.py": R.digest(
                    Path(__file__)
                ),
                "grain_research/construction/math_author_predictive_link.py": R.digest(
                    ROOT / "grain_research/construction/math_author_predictive_link.py"
                ),
            },
            "inputs": {name: R.digest(parent / name) for name in inputs},
            "quarters_fixed_by_native_target_position": True,
        },
    )
    cases = R.read(parent / "cases.json")
    with np.load(parent / "source_observations.npz", allow_pickle=False) as z:
        fields, choice = z["fields"], z["choice"]
    with np.load(parent / "actions.npz", allow_pickle=False) as z:
        actions = {k: z[k] for k in ("stranger", "order", "carry")}
    with (parent / "token_accounts.csv").open() as f:
        rows = list(csv.DictReader(f))
    directions, terms, quarters, error = describe(
        fields, choice, actions, np.array([len(c["target_ids"]) for c in cases]), rows
    )
    if error > 1e-8:
        raise ArithmeticError("The literal frozen accounts did not reconstruct")
    R.table(out / "direction_contacts.csv", directions)
    R.table(out / "normalizer_accounts.csv", terms)
    R.table(out / "token_quarters.csv", quarters)
    summary = {
        "post_outcome_not_primary_evidence": True,
        "all_direction_contacts": len(directions),
        "all_normalizer_accounts": len(terms),
        "all_quarter_accounts": len(quarters),
        "reconstruction_error": error,
        "no_new_model_calls_or_accuracy_estimates": True,
        "stranger_normalizer_changes_mean_sign": sum(
            r["normalizer_changes_mean_sign"]
            for r in terms
            if r["operation"] == "stranger"
        ),
        "stranger_same_target_by_window": [
            [
                next(
                    r["contact"]
                    for r in directions
                    if r["operation"] == "stranger"
                    and r["window"] == w
                    and r["part"] == "full"
                    and r["scale"] == "sum"
                    and r["source_target"] == q
                    and r["observer_target"] == q
                )
                for q in M.TARGETS
            ]
            for w in (0, 1)
        ],
    }
    R.write(out / "summary.json", summary)
    R.seal(
        out,
        "description_seal.json",
        (
            "direction_contacts.csv",
            "normalizer_accounts.csv",
            "token_quarters.csv",
            "summary.json",
        ),
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
