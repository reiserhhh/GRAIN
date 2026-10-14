#!/usr/bin/env python3
"""Describe every complete actual response table after HM165's finite budget stop."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_writer_history_bridge as M
from scripts import run_grain_observed_text_anchor_hm164 as A
from scripts import run_grain_writer_history_bridge_hm165 as R


def complete_tables(requests, captured):
    """Require all nine real corners; preserve missing keys instead of filling zeros."""
    tables = {}
    for request in requests:
        key = (request["pair_id"], request["opportunity"], request["order"])
        cells = tables.setdefault(key, {})
        if request["unit_index"] not in captured:
            continue
        item = captured[request["unit_index"]]
        cell = (request["policy"], request["history"])
        if cell in cells:
            raise ValueError("Duplicate addressed corner")
        cells[cell] = M.choice_account(
            item["ab_logits"], item["log_normalizer"], request["order"]
        )
    complete = {k: v for k, v in tables.items() if len(v) == 9}
    missing = [
        {
            "pair_id": k[0],
            "opportunity": k[1],
            "order": k[2],
            "captured_corners": len(v),
        }
        for k, v in sorted(tables.items())
        if len(v) < 9
    ]
    return complete, missing


def main():
    """Seal a post-outcome description, not a completed source/application experiment."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=R.DEFAULT)
    args = parser.parse_args()
    manifest = R.check(args.out)
    state = A.read(args.out / "collection_status.json")
    if state["status"] != "PARTIAL_BUDGET":
        raise ValueError(
            "This companion describes a confirmed finite-budget terminal stage"
        )
    captured = R.results(args.out)
    requests = A.read(args.out / "requests.json")
    tables, missing = complete_tables(requests, captured)
    pairs = {r["pair_id"]: r for r in A.read(args.out / "pairs.json")}
    destination = args.out / "partial_description_r1"
    destination.mkdir(exist_ok=False)
    sources = {
        str(p.relative_to(ROOT)): A.digest(p)
        for p in sorted((args.out / "batches").glob("*.json"))
    }
    A.write(
        destination / "manifest.json",
        {
            "stage": "POST_OUTCOME_PARTIAL_DESCRIPTION_NOT_PROMOTION_OR_PREREGISTRATION",
            "parent_manifest_sha256": A.digest(args.out / "manifest.json"),
            "capture_hashes": sources,
            "script_sha256": A.digest(Path(__file__)),
            "helper_sha256": A.digest(
                ROOT / "grain_research/construction/math_writer_history_bridge.py"
            ),
            "selection": "all complete registered nine-corner tables;capture prefix not random sample",
            "science_retries": 0,
            "additional_forwards": 0,
        },
    )
    with np.load(args.out / "features.npz", allow_pickle=False) as data:
        old = data["old_readings"]
    rows, errors = [], []
    for (pair_id, opportunity, order), cells in sorted(tables.items()):
        pair = pairs[pair_id]
        q = np.array(
            [[cells[p, h]["q_positive"] for h in (-1, 0, 1)] for p in (-1, 0, 1)]
        )
        odds = np.array(
            [[cells[p, h]["positive_log_odds"] for h in (-1, 0, 1)] for p in (-1, 0, 1)]
        )
        row = {
            k: pair[k]
            for k in ("pair_id", "axis", "situation", "style", "seed", "role")
        }
        row |= {"opportunity": opportunity, "order": order} | M.nine_corner_account(q)
        row |= {"log_odds_" + k: v for k, v in M.nine_corner_account(odds).items()}
        row["option_mass_mean"] = float(
            np.mean([v["unrestricted_option_mass"] for v in cells.values()])
        )
        row["option_mass_min"] = min(
            v["unrestricted_option_mass"] for v in cells.values()
        )
        target = ("extraversion", "neuroticism").index(pair["axis"])
        for j, arm in enumerate(("full", "single", "source_relinked")):
            row["old_persona_" + arm + "_difference"] = float(
                old[pair["high_article"], j, target]
                - old[pair["low_article"], j, target]
            )
        errors.append(
            abs(
                float(q[2, 2] - q[0, 0])
                - row["policy_effect"]
                - row["history_effect"]
                - row["interaction_effect"]
            )
        )
        rows.append(row)
    if not rows:
        raise ValueError(
            "No complete table is available; do not construct a scientific headline"
        )
    A.table(destination / "complete_response_accounts.csv", rows)
    A.table(destination / "missing_tables.csv", missing)
    groups = []
    keys = sorted(
        {(r["axis"], r["situation"], r["style"], r["opportunity"]) for r in rows}
    )
    for axis, situation, style, opportunity in keys:
        group = [
            r
            for r in rows
            if (r["axis"], r["situation"], r["style"], r["opportunity"])
            == (axis, situation, style, opportunity)
        ]
        result = {
            "axis": axis,
            "situation": situation,
            "style": style,
            "opportunity": opportunity,
            "accounts": len(group),
        }
        for k in (
            "policy_effect",
            "history_effect",
            "interaction_effect",
            "natural_effect",
            "joint_policy_history",
            "option_mass_mean",
        ):
            result[k + "_mean"] = float(np.mean([r[k] for r in group]))
            result[k + "_min"] = min(r[k] for r in group)
            result[k + "_max"] = max(r[k] for r in group)
        for k in ("history_effect", "natural_effect"):
            response = np.array([r[k] for r in group])
            reading = np.array([r["old_persona_full_difference"] for r in group])
            result[k + "_old_full_same"] = int((response * reading > 0).sum())
            result[k + "_old_full_opposite"] = int((response * reading < 0).sum())
        groups.append(result)
    A.table(destination / "condition_summary.csv", groups)
    probability_error = 0.0
    for item in captured.values():
        aa, bb = item["ab_logits"]
        q0 = M.choice_account(item["ab_logits"], item["log_normalizer"], 0)[
            "q_positive"
        ]
        probability_error = max(
            probability_error, abs(q0 - float(1 / (1 + np.exp(bb - aa))))
        )
    summary = {
        "status": "PARTIAL_BUDGET_MECHANISM_DESCRIPTION_NOT_FULL_BRIDGE_COMPLETE",
        "logical_requests_planned": manifest["logical_requests"],
        "unique_histories_planned": manifest["unique_histories"],
        "unique_histories_captured": len(captured),
        "model_seconds_completed_batches": sum(
            r["model_seconds_share"] for r in captured.values()
        ),
        "complete_nine_corner_tables": len(rows),
        "missing_tables": len(missing),
        "planned_tables": len(rows) + len(missing),
        "covered_axes": sorted({r["axis"] for r in rows}),
        "covered_tasks": sorted({r["situation"] for r in rows}),
        "covered_styles": sorted({r["style"] for r in rows}),
        "complete_pair_all8tables": sum(
            sum(r["pair_id"] == key for r in rows) == 8 for key in pairs
        ),
        "source_program_compiled": False,
        "cold_application_test_complete": False,
        "unobserved_are_zero_filled": False,
        "original_scope_complete": False,
        "max_natural_account_error": max(errors),
        "max_sigmoid_replay_error": probability_error,
        "same_agent_audit_not_blind": True,
        "additional_model_forwards": 0,
        "all_draws": "one per exact input;duplicate logical controls share the same draw",
    }
    A.write(destination / "summary.json", summary)
    A.write(
        destination / "output_seal.json",
        {
            name: A.digest(destination / name)
            for name in (
                "summary.json",
                "complete_response_accounts.csv",
                "missing_tables.csv",
                "condition_summary.csv",
            )
        },
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
