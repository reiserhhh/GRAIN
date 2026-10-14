#!/usr/bin/env python3
"""Describe the finite writer-choice prefix without promoting an incomplete bridge."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_writer_choice_observation as O
from grain_research.construction import math_writer_history_bridge as M
from scripts import run_grain_observed_text_anchor_hm164 as A
from scripts import run_grain_writer_history_bridge_hm165 as R
from scripts.describe_grain_writer_history_partial_hm165 import complete_tables


def main():
    """Keep every complete pair/order and seal all signed link/contrast accounts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=R.DEFAULT)
    args = parser.parse_args()
    R.check(args.out)
    description = args.out / "partial_description_r1"
    for name, expected in A.read(description / "output_seal.json").items():
        if A.digest(description / name) != expected:
            raise ValueError("Partial scientific description changed: " + name)
    if A.read(args.out / "collection_status.json")["status"] != "PARTIAL_BUDGET":
        raise ValueError("No finite-budget terminal stage")
    captures = R.results(args.out)
    tables, missing = complete_tables(A.read(args.out / "requests.json"), captures)
    pairs = {p["pair_id"]: p for p in A.read(args.out / "pairs.json")}
    destination = args.out / "observation_law_description_r1"
    destination.mkdir(exist_ok=False)
    files = (
        Path(__file__),
        ROOT / "grain_research/construction/math_writer_choice_observation.py",
        ROOT / "tests/test_grain_writer_choice_observation.py",
    )
    A.write(
        destination / "manifest.json",
        {
            "stage": "POST_OUTCOME_FINITE_OBSERVATION_LAW_DESCRIPTION",
            "parent_manifest": A.digest(args.out / "manifest.json"),
            "partial_output_seal": A.digest(description / "output_seal.json"),
            "capture_hashes": {
                str(p.relative_to(ROOT)): A.digest(p)
                for p in sorted((args.out / "batches").glob("*.json"))
            },
            "companion_hashes": {str(p.relative_to(ROOT)): A.digest(p) for p in files},
            "new_model_forwards": 0,
            "selection": "all complete registered tables; nonrandom prefix; no new fit",
            "order_odd_is_removed": False,
        },
    )
    start = time.monotonic()
    contrast_rows, link_rows, order_rows = [], [], []
    error = defaultdict(float)
    coordinates = ("constant", "polarity", "nonneutral_vs_neutral")
    grouped = defaultdict(dict)
    for (pair_id, opportunity, order), cells in sorted(tables.items()):
        pair = pairs[pair_id]
        metadata = {
            k: pair[k]
            for k in ("pair_id", "axis", "situation", "style", "seed", "role")
        }
        metadata |= {"opportunity": opportunity, "order": order}
        ell = np.array(
            [[cells[p, h]["positive_log_odds"] for h in (-1, 0, 1)] for p in (-1, 0, 1)]
        )
        q = np.array(
            [[cells[p, h]["q_positive"] for h in (-1, 0, 1)] for p in (-1, 0, 1)]
        )
        grouped[pair_id, opportunity][order] = (metadata, q, ell)
        for measure, values in (("q", q), ("log_odds", ell)):
            coefficients, reconstruction = O.orthogonal_accounts(values)
            error["orthogonal_reconstruction"] = max(
                error["orthogonal_reconstruction"], reconstruction
            )
            error["energy_reconstruction"] = max(
                error["energy_reconstruction"],
                abs(float(np.sum(coefficients**2) - np.sum(values**2))),
            )
            for i, j in np.ndindex((3, 3)):
                contrast_rows.append(
                    metadata
                    | {
                        "measure": measure,
                        "policy_contrast": coordinates[i],
                        "history_contrast": coordinates[j],
                        "coefficient": float(coefficients[i, j]),
                    }
                )
        accounts = O.link_accounts(ell)
        error["native_probability_replay"] = max(
            error["native_probability_replay"], float(np.max(abs(q - accounts["q"])))
        )
        error["signed_link_reconstruction"] = max(
            error["signed_link_reconstruction"], accounts["q_reconstruction_error"]
        )
        for i, j in np.ndindex((3, 3)):
            link_rows.append(
                metadata
                | {
                    "policy": i - 1,
                    "history": j - 1,
                    "actual_q": float(q[i, j]),
                    "actual_log_odds": float(ell[i, j]),
                    **{
                        k: float(accounts[k][i, j])
                        for k in (
                            "separated_log_odds",
                            "log_odds_interaction",
                            "q_interaction",
                            "link_curvature",
                            "response_remainder",
                        )
                    },
                }
            )
    for (pair_id, opportunity), orders in sorted(grouped.items()):
        if set(orders) != {0, 1}:
            continue
        metadata = {k: v for k, v in orders[0][0].items() if k != "order"}
        row = metadata.copy()
        for index, measure in ((1, "q"), (2, "log_odds")):
            split = O.aligned_order_accounts(orders[0][index], orders[1][index])
            error["order_reconstruction"] = max(
                error["order_reconstruction"], split["reconstruction_error"]
            )
            for component in ("even", "odd"):
                for name, value in M.nine_corner_account(split[component]).items():
                    row[measure + "_" + component + "_" + name] = value
            for order in (0, 1):
                row[measure + "_order" + str(order) + "_history_effect"] = (
                    M.nine_corner_account(orders[order][index])["history_effect"]
                )
            row[measure + "_history_order_sign_reversal"] = int(
                row[measure + "_order0_history_effect"]
                * row[measure + "_order1_history_effect"]
                < 0
            )
        order_rows.append(row)
    A.table(destination / "contrast_coordinates.csv", contrast_rows)
    A.table(destination / "link_accounts.csv", link_rows)
    A.table(destination / "order_accounts.csv", order_rows)
    overview = []
    for opportunity in sorted({r["opportunity"] for r in order_rows}):
        selected = [r for r in order_rows if r["opportunity"] == opportunity]
        row = {"opportunity": opportunity, "complete_pair_two_orders": len(selected)}
        for measure in ("q", "log_odds"):
            row[measure + "_history_order_sign_reversals"] = sum(
                r[measure + "_history_order_sign_reversal"] for r in selected
            )
            for component in ("even", "odd"):
                for effect in (
                    "policy_effect",
                    "history_effect",
                    "interaction_effect",
                    "natural_effect",
                ):
                    key = measure + "_" + component + "_" + effect
                    values = np.array([r[key] for r in selected])
                    row[key + "_mean"] = float(values.mean())
                    row[key + "_rms"] = float(np.sqrt(np.mean(values**2)))
        corners = [
            r
            for r in link_rows
            if r["opportunity"] == opportunity
            and r["policy"] == r["history"]
            and r["policy"] != 0
        ]
        by_table = defaultdict(dict)
        for r in corners:
            by_table[r["pair_id"], r["order"]][r["policy"]] = r
        for component in ("q_interaction", "link_curvature", "response_remainder"):
            values = np.array(
                [v[1][component] - v[-1][component] for v in by_table.values()]
            )
            row["diagonal_" + component + "_mean"] = float(values.mean())
            row["diagonal_" + component + "_rms"] = float(np.sqrt(np.mean(values**2)))
        overview.append(row)
    A.table(destination / "opportunity_summary.csv", overview)
    summary = {
        "status": "FINITE_PREFIX_OBSERVATION_MECHANISMS_DESCRIBED_FULL_BRIDGE_PENDING",
        "captured_unique": len(captures),
        "complete_tables": len(tables),
        "missing_tables": len(missing),
        "paired_order_accounts": len(order_rows),
        "covered_axes": sorted({r["axis"] for r in order_rows}),
        "prefix_is_balanced_population_sample": False,
        "source_program_compiled": False,
        "cold_application_complete": False,
        "new_forwards": 0,
        "fitted_parameters": 0,
        "audit": dict(error),
        "cpu_seconds": time.monotonic() - start,
        "original_scope_complete": False,
        "same_agent_not_blind": True,
    }
    A.write(destination / "summary.json", summary)
    A.write(
        destination / "output_seal.json",
        {
            name: A.digest(destination / name)
            for name in (
                "summary.json",
                "contrast_coordinates.csv",
                "link_accounts.csv",
                "order_accounts.csv",
                "opportunity_summary.csv",
            )
        },
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
