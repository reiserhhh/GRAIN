#!/usr/bin/env python3
"""Reconstruct HM77 actions with explicit columns, including anchor odd leakage."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_grain_context_anchored_operation import (
    OUT,
    all_groups,
    check,
    digest,
    models,
    read,
    write,
)


def main():
    """Recompute actual paired-state accounts without the primary math helpers."""
    manifest = check()
    for name, sha in read(OUT / "03/output_seal.json").items():
        if digest(OUT / name) != sha:
            raise ValueError("Output seal drift: " + name)
    fitted = models()
    groups, used = all_groups()
    if used != read(OUT / "03/input_seal.json"):
        raise ValueError("Consumed object set changed")
    max_error, max_cross, max_null, checks = 0.0, 0.0, 0.0, 0
    source_max, group_max = 0.0, 0.0
    for summary in read(OUT / "03/summary.json")["summaries"]:
        method = summary["arm"].split("_", 1)[1]
        conditioned = summary["arm"].startswith("anchored")
        model = fitted[summary["query"], method]
        base, total, context, missing, contamination, crosses = [], [], [], [], [], []
        for key, value in sorted(groups.items()):
            if key[:2] != (summary["bank"], summary["query"]):
                continue
            bp, qp = value[1]
            bm, qm = value[-1]
            project = lambda v, model=model: model["left"] @ (model["right"] @ v)
            anchor = lambda b, model=model: (
                model["anchor_left"]
                @ (model["anchor_right"] @ (b - model["body_origin"]))
            )
            ap, am = (
                (anchor(bp), anchor(bm))
                if conditioned
                else (np.zeros_like(qp), np.zeros_like(qm))
            )
            tp = qp - 2 * project(qp - model["origin"] - ap)
            tm = qm - 2 * project(qm - model["origin"] - am)
            total.append(((tp - qm) @ (tp - qm) + (tm - qp) @ (tm - qp)) / 2)
            base.append((qp - qm) @ (qp - qm))
            l = project((qp + qm) / 2 - model["origin"] - (ap + am) / 2)
            d = (qp - qm) / 2
            miss = d - project(d)
            contam = project((ap - am) / 2)
            context.append(l @ l)
            missing.append(miss @ miss)
            contamination.append(contam @ contam)
            crosses.append(2 * (miss @ contam))
            max_error = max(
                max_error,
                abs(
                    total[-1]
                    - 4 * (context[-1] + missing[-1] + contamination[-1] + crosses[-1])
                ),
            )
            if method == "orthogonal":
                max_cross = max(max_cross, abs(crosses[-1]))
            if conditioned and method == "oblique":
                max_null = max(
                    max_null,
                    float(np.max(abs(tp - (qp - 2 * project(qp - model["origin"]))))),
                )
            tb = bp - 2 * model["body_left"] @ (
                model["body_right"] @ (bp - model["body_origin"])
            )
            second_q = tp - 2 * project(
                tp - model["origin"] - (anchor(tb) if conditioned else 0)
            )
            group_max = max(group_max, float(np.max(abs(second_q - qp))))
            if summary["bank"] == "source0" and conditioned:
                source_max = max(
                    source_max, float(np.max(abs(tp - qm))), float(np.max(abs(tb - bm)))
                )
            checks += 2 * len(qp)
        expected = {
            "relative_error": float(np.sqrt(np.mean(total) / np.mean(base))),
            "operator_error_rms": float(np.sqrt(np.mean(total))),
            "identity_error_rms": float(np.sqrt(np.mean(base))),
            "context_sq_mean": float(np.mean(context)),
            "missing_sq_mean": float(np.mean(missing)),
            "contamination_sq_mean": float(np.mean(contamination)),
            "odd_cross_mean": float(np.mean(crosses)),
        }
        for field, value in expected.items():
            max_error = max(max_error, abs(summary[field] - value))
        if len(total) != summary["pairs"]:
            raise ValueError("Pair count changed")
    status = read(OUT / "run_status.json")
    if (
        status["status"] != "COLLECTED"
        or status["completed"] != manifest["spec"]["max_calls"]
        or status["attempts"] != status["completed"]
        or status["in_flight"] is not None
    ):
        raise ValueError("Fresh acquisition incomplete or uncertain")
    if max(max_error, max_cross, max_null, source_max, group_max) > 1e-8:
        raise ValueError("Conditional operation/account controls failed")
    result = {
        "status": "VERIFIED_FINITE_HM77",
        "fresh_actual_forwards": status["attempts"],
        "alternate_scalar_checks": checks,
        "max_account_error": max_error,
        "orthogonal_cross_max_error": max_cross,
        "oblique_null_max_error": max_null,
        "source_exchange_max_error": source_max,
        "joint_group_max_error": group_max,
        "zero_new_model_calls": True,
        "same_agent_not_third_party": True,
        "original_scope_complete": False,
    }
    write(OUT / "verification.json", result)
    write(
        OUT / "checkpoint.json",
        {
            "status": "VERIFIED",
            "active_processes": [],
            "original_scope_complete": False,
        },
    )
    print(result, flush=True)


if __name__ == "__main__":
    main()
