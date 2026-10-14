#!/usr/bin/env python3
"""Build and verify finite pairing accounts from immutable HM118 observations."""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_policy_cycles as M
from scripts import run_grain_policy_coupling_hm118 as R

H = R.H
CODE = (
    "grain_research/construction/math_policy_cycles.py",
    "scripts/readback_grain_policy_cycles_hm118.py",
    "tests/test_grain_math_policy_cycles.py",
    "docs/GRAIN_POLICY_CYCLES_READBACK_PLAN_HM118.md",
)
FIELDS = (
    "reference",
    "independent",
    "marginal",
    "cycle",
    "observed",
    "reconstructed",
    "lower",
    "upper",
)


def load_source(source):
    """Require complete verified source, exact input/output seals and own roles."""
    manifest = R.check(source)
    if (
        H.load(source / "verification.json")["status"]
        != "VERIFIED_FINITE_HM118_NOT_GLOBAL"
    ):
        raise ValueError("Verified actual HM118 source required")
    H.verify_hashes(source, H.load(source / "02/output_seal.json"))
    values = R.read_scores(source)
    with np.load(source / "00/kernels.npz", allow_pickle=False) as stream:
        kernels = stream["kernels"]
    with np.load(source / "00/ledgers_instrument.npz", allow_pickle=False) as stream:
        joint, owners, queried, origins = (
            stream[p]
            for p in ("joint_counts", "evidence_owner", "query_owner", "origins")
        )
    if (
        not np.array_equal(owners, queried)
        or not np.isin(owners, (0, 1)).all()
        or not np.all(origins == "instrument")
        or bool(values["neural_runtime_loaded"])
    ):
        raise ValueError("Own-actor instrument records and cold source required")
    return manifest["spec"], kernels, joint, owners, values


def run(source):
    """Construct all signed accounts and sharp bounds without choosing a model."""
    start = time.monotonic()
    spec, kernels, joint, owners, values = load_source(source)
    out = source / "cycles_readback"
    out.mkdir()
    H.write(
        out / "manifest.json",
        {
            "code": {p: H.sha(ROOT / p) for p in CODE},
            "source": {
                p: H.sha(source / p)
                for p in (
                    "manifest.json",
                    "verification.json",
                    "00/kernels.npz",
                    "00/ledgers_instrument.npz",
                    "00/records.json",
                    "01/instrument_scores.npz",
                    "01/output_seal.json",
                    "02/output_seal.json",
                )
            },
            "timing": "Result-after companion;code and analysis rule frozen before readback",
            "scope": "Finite 2x2 groups;not personality factors,confidence intervals or simultaneous target extrema",
            "new_forwards": 0,
            "fits": 0,
            "arithmetic_tolerance": spec["arithmetic_tolerance"],
        },
    )
    records = H.load(source / "00/records.json")
    grouped = joint.reshape(len(joint), 5, 12, 2, 2)
    coordinates = M.coordinates(grouped)
    gamma = kernels.reshape(2, 5, 12, 2, 2, 6, 4)
    gamma = (
        gamma[:, :, :, 0, 0]
        - gamma[:, :, :, 0, 1]
        - gamma[:, :, :, 1, 0]
        + gamma[:, :, :, 1, 1]
    )
    arrays = {field: np.empty_like(values["observed"]) for field in FIELDS}
    group_shape = (len(joint), 5, 5, 12, 6, 4)
    arrays.update(
        {
            p: np.empty(group_shape)
            for p in ("cycle_by_group", "lower_witness", "upper_witness")
        }
    )
    for h, policy, obj in itertools.product(range(len(joint)), range(5), range(5)):
        result = M.response_account(
            grouped[h, policy], kernels[owners[h], obj].reshape(12, 2, 2, 6, 4)
        )
        for field, output in arrays.items():
            output[h, policy, obj] = result[field]
    R.save(
        out / "accounts.npz",
        **arrays,
        **{f"coordinate_{p}": v for p, v in coordinates.items()},
        gamma=gamma,
    )
    coordinate_rows = []
    for h, policy, group in itertools.product(range(len(joint)), range(5), range(12)):
        family, condition = divmod(group, 2)
        row = {
            **records[h],
            "policy_name": spec["policies"][policy],
            "family": spec["targets"][family],
            "condition": 2 * condition - 1,
            "coverage": bool(coordinates["observed"][h, policy, group]),
            "low_count": float(coordinates["rows"][h, policy, group, 0]),
            "high_count": float(coordinates["rows"][h, policy, group, 1]),
            "focus_B_count": float(coordinates["columns"][h, policy, group, 0]),
            "focus_A_count": float(coordinates["columns"][h, policy, group, 1]),
        }
        row.update(
            {
                p: float(coordinates[p][h, policy, group])
                for p in ("coordinate", "lower_coordinate", "upper_coordinate")
            }
        )
        coordinate_rows.append(row)
    H.table(out / "pairing_coordinates.csv", coordinate_rows)
    response_rows = []
    for owner, obj, group, q, unit in itertools.product(
        range(2), range(5), range(12), range(6), range(4)
    ):
        family, condition = divmod(group, 2)
        response_rows.append(
            {
                "slot": "A" if owner else "B",
                "object": spec["objects"][obj],
                "family": spec["targets"][family],
                "condition": 2 * condition - 1,
                "query": spec["targets"][q],
                "unit": spec["units"][unit],
                "rectangular_response": float(gamma[owner, obj, group, q, unit]),
            }
        )
    H.table(out / "target_response_vectors.csv", response_rows)
    account_rows = []
    for h, policy, obj, q, unit in itertools.product(
        range(len(joint)), range(5), range(5), range(6), range(4)
    ):
        address = (h, policy, obj, q, unit)
        account_rows.append(
            {
                **records[h],
                "policy_name": spec["policies"][policy],
                "object": spec["objects"][obj],
                "query": spec["targets"][q],
                "unit": spec["units"][unit],
                **{field: float(arrays[field][address]) for field in FIELDS},
                "identification_width": float(
                    arrays["upper"][address] - arrays["lower"][address]
                ),
            }
        )
    H.table(out / "reading_accounts.csv", account_rows)
    summaries = []
    for owner, obj, unit in itertools.product(range(2), range(5), range(4)):
        selected = np.flatnonzero(owners == owner)
        width = (arrays["upper"] - arrays["lower"])[selected, 2, obj, :, unit]
        actual = (
            values["observed"][selected, 3, obj, :, unit]
            - values["observed"][selected, 4, obj, :, unit]
        )
        summaries.append(
            {
                "slot": "A" if owner else "B",
                "object": spec["objects"][obj],
                "unit": spec["units"][unit],
                "coupled_pair_RMS": float(np.sqrt(np.mean(actual**2))),
                "fixed_margin_width_RMS": float(np.sqrt(np.mean(width**2))),
                "fixed_margin_width_max": float(width.max()),
                "mean_absolute_cycle": float(
                    np.mean(abs(arrays["cycle"][selected, 3, obj, :, unit]))
                ),
                "cells": int(width.size),
            }
        )
    H.table(out / "identification_summaries.csv", summaries)
    lookup = {(r["slot"], r["recipe"], r["seed"]): r["history"] for r in records}
    comparisons = []
    for slot, seed, pair, obj, q, unit in itertools.product(
        ("A", "B"),
        spec["seeds"],
        ((1, 2), (3, 0), (5, 4)),
        range(5),
        range(6),
        range(4),
    ):
        left, right = lookup[(slot, pair[0], seed)], lookup[(slot, pair[1], seed)]
        index = (2, obj, q, unit)
        lower = float(arrays["lower"][left][index] - arrays["upper"][right][index])
        upper = float(arrays["upper"][left][index] - arrays["lower"][right][index])
        comparisons.append(
            {
                "slot": slot,
                "seed": seed,
                "recipes": f"{pair[0]}-{pair[1]}",
                "object": spec["objects"][obj],
                "query": spec["targets"][q],
                "unit": spec["units"][unit],
                "known_balanced_contrast": float(
                    arrays["observed"][left][index] - arrays["observed"][right][index]
                ),
                "unlogged_pairing_lower": lower,
                "unlogged_pairing_upper": upper,
                "sign_identified_from_margins": bool(
                    lower > spec["arithmetic_tolerance"]
                    or upper < -spec["arithmetic_tolerance"]
                ),
            }
        )
    H.table(out / "contrast_identification.csv", comparisons)
    H.write(
        out / "summary.json",
        {
            "state": "ANALYZED_PENDING_VERIFICATION",
            "seconds": time.monotonic() - start,
            "histories": len(joint),
            "distinct_event_ledgers": len(joint) // 2,
            "coordinates_per_group": 1,
            "groups_per_history": 12,
            "rows": {
                "coordinates": len(coordinate_rows),
                "responses": len(response_rows),
                "accounts": len(account_rows),
                "comparisons": len(comparisons),
            },
            "new_forwards": 0,
            "fits": 0,
            "neural_runtime_loaded": bool(
                {"torch", "transformers"}.intersection(sys.modules)
            ),
            "original_scope_complete": False,
        },
    )
    H.write(out / "output_seal.json", H.seals(out, out))
    print(
        "CONSTRUCTED HM118 cycles",
        len(account_rows),
        "readings;no neural calls/fits",
        flush=True,
    )


def verify(source):
    """Recompute each actual reading and every extremum with independent tables."""
    spec, kernels, joint, owners, values = load_source(source)
    out = source / "cycles_readback"
    manifest = H.load(out / "manifest.json")
    H.verify_hashes(ROOT, manifest["code"])
    H.verify_hashes(source, manifest["source"])
    H.verify_hashes(out, H.load(out / "output_seal.json"))
    with np.load(out / "accounts.npz", allow_pickle=False) as stream:
        arrays = {p: stream[p] for p in stream.files}
    errors = {
        p: 0.0
        for p in (
            "tables",
            "direct_readings",
            "reference",
            "closure",
            "gamma",
            "endpoints",
            "witness_margins",
            "witness_nonnegative",
            "witness_integer",
            "witness_extremal",
            "bound_contains_observed",
            "CSV",
        )
    }
    grouped = joint.reshape(len(joint), 5, 12, 2, 2)
    for h, policy in itertools.product(range(len(joint)), range(5)):
        x = grouped[h, policy]
        reconstructed = (
            arrays["coordinate_independent"][h, policy]
            + arrays["coordinate_coordinate"][h, policy, :, None, None] * M.CYCLE
        )
        errors["tables"] = max(errors["tables"], float(np.max(abs(reconstructed - x))))
        for obj in range(5):
            k = kernels[owners[h], obj].reshape(12, 2, 2, 6, 4)
            actual, reference, cycles = (
                np.zeros((6, 4)),
                np.zeros((6, 4)),
                np.zeros((6, 4)),
            )
            for group, b, r in itertools.product(range(12), range(2), range(2)):
                actual += x[group, b, r] * k[group, b, r] / x.sum()
                reference += x[group, b].sum() * k[group, b, r] / (2 * x.sum())
            for group in range(12):
                g = k[group, 0, 0] + k[group, 1, 1] - k[group, 0, 1] - k[group, 1, 0]
                errors["gamma"] = max(
                    errors["gamma"],
                    float(np.max(abs(g - arrays["gamma"][owners[h], obj, group]))),
                )
                cycles += (
                    arrays["coordinate_coordinate"][h, policy, group] * g / x.sum()
                )
            address = (h, policy, obj)
            errors["direct_readings"] = max(
                errors["direct_readings"],
                float(np.max(abs(actual - arrays["observed"][address]))),
                float(np.max(abs(actual - values["observed"][address]))),
            )
            errors["reference"] = max(
                errors["reference"],
                float(np.max(abs(reference - arrays["reference"][address]))),
                float(np.max(abs(reference - values["reference"][address]))),
            )
            errors["closure"] = max(
                errors["closure"],
                float(
                    np.max(
                        abs(actual - reference - arrays["marginal"][address] - cycles)
                    )
                ),
            )
            errors["bound_contains_observed"] = max(
                errors["bound_contains_observed"],
                float(np.max(np.maximum(arrays["lower"][address] - actual, 0))),
                float(np.max(np.maximum(actual - arrays["upper"][address], 0))),
            )
            for endpoint in ("lower", "upper"):
                scores = np.zeros((6, 4))
                for group, q, unit in itertools.product(range(12), range(6), range(4)):
                    rows, columns = x[group].sum(axis=1), x[group].sum(axis=0)
                    cell = (
                        arrays["coordinate_independent"][h, policy, group, 0, 0]
                        + arrays[f"{endpoint}_witness"][h, policy, obj, group, q, unit]
                    )
                    witness = np.array(
                        [
                            [cell, rows[0] - cell],
                            [columns[0] - cell, rows[1] - columns[0] + cell],
                        ]
                    )
                    rectangle = (
                        k[group, 0, 0, q, unit]
                        + k[group, 1, 1, q, unit]
                        - k[group, 0, 1, q, unit]
                        - k[group, 1, 0, q, unit]
                    )
                    smallest = max(0, rows[0] + columns[0] - rows.sum())
                    largest = min(rows[0], columns[0])
                    use_smallest = (endpoint == "lower") == (rectangle >= 0)
                    extremal_cell = smallest if use_smallest else largest
                    errors["witness_extremal"] = max(
                        errors["witness_extremal"], abs(float(cell - extremal_cell))
                    )
                    errors["witness_margins"] = max(
                        errors["witness_margins"],
                        float(np.max(abs(witness.sum(axis=1) - rows))),
                        float(np.max(abs(witness.sum(axis=0) - columns))),
                    )
                    errors["witness_nonnegative"] = max(
                        errors["witness_nonnegative"], float(max(0, -witness.min()))
                    )
                    errors["witness_integer"] = max(
                        errors["witness_integer"],
                        float(np.max(abs(witness - np.rint(witness)))),
                    )
                    scores[q, unit] += (
                        witness * k[group, :, :, q, unit]
                    ).sum() / x.sum()
                errors["endpoints"] = max(
                    errors["endpoints"],
                    float(np.max(abs(scores - arrays[endpoint][address]))),
                )
    with (out / "reading_accounts.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    cases = itertools.product(range(len(joint)), range(5), range(5), range(6), range(4))
    records = H.load(source / "00/records.json")
    for row, address in zip(rows, cases, strict=True):
        h, policy, obj, q, unit = address
        expected = {
            **records[h],
            "policy_name": spec["policies"][policy],
            "object": spec["objects"][obj],
            "query": spec["targets"][q],
            "unit": spec["units"][unit],
        }
        if any(row[key] != str(value) for key, value in expected.items()):
            raise ValueError("Reading scientific address changed")
        for field in FIELDS:
            errors["CSV"] = max(
                errors["CSV"], abs(float(row[field]) - float(arrays[field][address]))
            )
    summary = H.load(out / "summary.json")
    if summary["neural_runtime_loaded"] or summary["new_forwards"] or summary["fits"]:
        raise ValueError("A numerical readback must not invoke inference/fitting")
    if max(errors.values()) > spec["arithmetic_tolerance"]:
        raise ValueError(f"Cycle construction verification failed: {errors}")
    H.write(
        out / "verification.json",
        {
            "status": "VERIFIED_FINITE_HM118_CYCLES_NOT_GLOBAL",
            "errors": errors,
            "per_output_endpoint_witnesses": len(joint) * 5 * 5 * 6 * 4 * 2,
            "new_forwards": 0,
            "fits": 0,
            "same_agent_not_third_party": True,
            "original_scope_complete": False,
        },
    )
    print("VERIFIED HM118 cycles", errors, flush=True)


def main():
    """Run a single finite companion or verify it, without automatic retries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("run", "verify"))
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "results/grain_math_response_v1/policy_coupling_r1",
    )
    args = parser.parse_args()
    if args.phase == "run":
        run(args.source)
    else:
        verify(args.source)


if __name__ == "__main__":
    main()
