#!/usr/bin/env python3
"""Result-after terminal table replay and target stratification; no new inference."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_condition_order as M
from scripts.run_grain_author_contrast_hm82 import load, seal, write
from scripts.run_grain_condition_order_hm83 import OUT, check


def main():
    """Rebuild numerical report rows from actual cold scores, retaining all arms."""
    check(OUT)
    if load(OUT / "verification.json")["status"] != "VERIFIED_FINITE_HM83_NOT_GLOBAL":
        raise ValueError("primary numerical verification required")
    dest = OUT / "04_terminal_interpretation_r2"
    dest.mkdir(exist_ok=False)
    events = load(OUT / "00/application.json")
    outputs = [
        json.loads(s) for s in (OUT / "03/scores.jsonl").read_text().splitlines()
    ][:120]
    with np.load(OUT / "03/terminal_arrays.npz") as a:
        truth = a["truth"]
    summary = list(csv.DictReader((OUT / "03/new_text_function_summary.csv").open()))
    actions = list(csv.DictReader((OUT / "03/new_text_signed_actions.csv").open()))
    errors, checks, strata = [], 0, []
    for arm in M.ARMS:
        values = np.array([s["arms"][arm]["scores"] for s in outputs])
        for j, target in enumerate(M.TARGETS):
            row = next(r for r in summary if r["arm"] == arm and r["target"] == target)
            alternate = [
                np.sqrt(
                    sum(
                        (v - t) ** 2
                        for v, t in zip(values[:, j], truth[:, j], strict=True)
                    )
                    / 120
                ),
                sum(values[:, j] * truth[:, j]) / 120,
                sum(values[:, j]) / 120,
                np.sqrt(sum(values[:, j] ** 2) / 120),
            ]
            fields = (
                "new_text_RMSE_design_only",
                "signed_target_action",
                "new_text_score_mean",
                "new_text_score_RMS",
            )
            errors.extend(
                abs(float(row[k]) - v) for k, v in zip(fields, alternate, strict=True)
            )
            checks += 4
        by_target = {}
        for k in range(30):
            event = events[4 * k]
            local = values[4 * k : 4 * k + 4]
            for j, target in enumerate(M.TARGETS):
                row = next(
                    r
                    for r in actions
                    if r["arm"] == arm
                    and r["focal"] == event["focal"]
                    and r["companion"] == event["companion"]
                    and r["target_function"] == target
                )
                coefficients = [
                    sum(local[:, j] * truth[4 * k : 4 * k + 4, ix]) / 4
                    for ix in (1, 0, 2)
                ] + [sum(local[:, j]) / 4]
                errors.extend(
                    abs(float(row[f]) - v)
                    for f, v in zip(
                        ("x_action", "c_action", "cx_action", "offset"),
                        coefficients,
                        strict=True,
                    )
                )
                checks += 4
                if target == "focal_polarity":
                    by_target.setdefault(event["focal"], []).append(coefficients[0])
        strata.extend(
            {
                "arm": arm,
                "focal_material_target": name,
                "mean_x_action": float(sum(v) / len(v)),
                "positive_pairs": int(sum(a > 0 for a in v)),
                "pairs": len(v),
                "not_questionnaire_r_or_real_authors": True,
            }
            for name, v in by_target.items()
        )
    conditional = np.array(
        [s["arms"]["conditional_joint99"]["scores"] for s in outputs]
    )
    gated_identity = float(
        np.max(abs(conditional[:, 2] - truth[:, 0] * conditional[:, 1]))
    )
    if max(errors) > 1e-10 or gated_identity > 1e-10:
        raise ArithmeticError("terminal arithmetic/gating identity mismatch")
    result = {
        "status": "VERIFIED_RESULT_AFTER_TERMINAL_TABLES_HM83",
        "scalar_checks": checks,
        "max_error": max(errors),
        "conditional_cx_equals_c_times_x_max_error": gated_identity,
        "identity_is_constructive_control_not_two_independent_successes": True,
        "target_strata": strata,
        "new_model_calls": 0,
        "predecessor_failure": "04_terminal_interpretation retained: int64 JSON serialization failed after arithmetic checks; no model calls or primary scientific changes",
        "source_functions_modified": False,
        "role_joint_confounded_with_cx_in_fixed_other_actor_application": True,
        "same_agent_alternate_not_independent_rebuilder": True,
        "original_scope_complete": False,
    }
    write(dest / "terminal_verification.json", result)
    write(
        dest / "seal.json",
        {
            "files": seal(OUT, [dest / "terminal_verification.json"]),
            "code_sha256": seal(ROOT, [Path(__file__)]),
        },
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
