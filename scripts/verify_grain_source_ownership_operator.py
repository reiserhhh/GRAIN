#!/usr/bin/env python3
"""Check real raw inputs and alternate single-vector action arithmetic for HM76."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_grain_source_ownership_operator import (
    COLD,
    OUT,
    SOURCE,
    check,
    digest,
    load_groups,
    models,
    read,
    write,
)


def main():
    """Recompute from sealed endpoint pairs without new model calls or fitting."""
    check()
    for name, sha in read(OUT / "03/output_seal.json").items():
        if digest(OUT / name) != sha:
            raise ValueError("Primary output drift: " + name)
    source_models = models()
    groups, used = load_groups(SOURCE, lambda _r: True)
    for root in (COLD, OUT):
        part, inputs = load_groups(root, lambda _r: True)
        groups.update(part)
        used.update(inputs)
    if used != read(OUT / "03/input_seal.json"):
        raise ValueError("Consumed raw object set mismatch")
    max_account, max_group = 0.0, 0.0
    checks = 0
    for row in read(OUT / "03/summary.json")["summaries"]:
        model = source_models[row["stage"], row["method"]]
        observed = [
            value
            for key, value in sorted(groups.items())
            if key[:2] == (row["bank"], row["stage"])
        ]
        total, base, leak, missing = [], [], [], []
        for value in observed:
            a, b = value[1], value[-1]
            # Explicit column covectors, independent of pair_accounts/operate.
            p = model["left"] @ (model["right"] @ (a - model["origin"]))
            q = model["left"] @ (model["right"] @ (b - model["origin"]))
            ra, rb = a - 2 * p - b, b - 2 * q - a
            total.append((ra @ ra + rb @ rb) / 2)
            base.append((a - b) @ (a - b))
            pe = (p + q) / 2
            dd = (a - b) / 2 - (p - q) / 2
            leak.append(4 * (pe @ pe))
            missing.append(4 * (dd @ dd))
            max_account = max(max_account, abs(total[-1] - leak[-1] - missing[-1]))
            pp = model["left"] @ (model["right"] @ p)
            max_group = max(max_group, float(np.max(abs(pp - p))))
            checks += 2 * len(a)
        expected = {
            "identity_error_rms": float(np.sqrt(np.mean(base))),
            "operator_error_rms": float(np.sqrt(np.mean(total))),
            "relative_error": float(np.sqrt(np.mean(total) / np.mean(base))),
            "even_leakage_error_rms": float(np.sqrt(np.mean(leak))),
            "odd_missing_error_rms": float(np.sqrt(np.mean(missing))),
        }
        for field, val in expected.items():
            max_account = max(max_account, abs(val - row[field]))
        if len(observed) != row["pairs"]:
            raise ValueError("Pair count mismatch")
    status = read(OUT / "run_status.json")
    if (
        status["attempts"] != 576
        or status["completed"] != 576
        or status["in_flight"] is not None
    ):
        raise ValueError("Actual bounded acquisition incomplete")
    if max_account > 1e-8 or max_group > 1e-8:
        raise ValueError("Independent operation account/group law mismatch")
    result = {
        "status": "VERIFIED_FINITE_HM76_NOT_GLOBAL",
        "fresh_actual_forwards": 576,
        "upper_reader_calls": 0,
        "alternate_scalar_checks": checks,
        "max_account_error": max_account,
        "max_group_error": max_group,
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
