#!/usr/bin/env python3
"""Result-guided exact mean-account witnesses; no new neural call or target fit."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_same_evidence_hm137 as H
from grain_research.construction import math_conditional_mean_fiber as M

CODE = (
    "docs/GRAIN_SAME_EVIDENCE_MEAN_FIBER_ANNEX_HM137.md",
    "grain_research/construction/math_conditional_mean_fiber.py",
    "scripts/analyze_grain_same_evidence_mean_fibers.py",
    "tests/test_grain_conditional_mean_fiber.py",
)


def main():
    """Freeze the annex before computing all declared summaries and targets."""
    parent = ROOT / H.read(H.SPEC)["out"]
    if (
        H.read(parent / "verification.json")["status"]
        != "VERIFIED_FINITE_HM137_NOT_GLOBAL"
    ):
        raise ValueError("Verified preceding comparison required")
    out = parent / "03_mean_fiber_annex"
    out.mkdir(exist_ok=False)
    H.write(
        out / "manifest.json",
        {
            "post_result_diagnostic": True,
            "code": {p: H.sha(ROOT / p) for p in CODE},
            "parent_verification": H.sha(parent / "verification.json"),
            "relative_SVD_cutoff": 1e-10,
            "numerical_tolerance": 1e-10,
            "max_seconds": 60,
            "new_neural_calls": 0,
            "new_predictor_fits": 0,
        },
    )
    start = time.monotonic()
    with threadpool_limits(4):
        execute(parent, out, start)


def execute(parent, out, started):
    """Check every positive law against original rows and alternate integration."""
    manifest, embed, grain, native, _ = H.arrays(parent)
    g, e = grain.reshape(96, 15), embed.reshape(96, 1024)
    # The source cubes already use exact stratum/b/c ordering, not encounter order.
    d = M.conditional_contrasts(24)
    y = native[..., :2].reshape(96, 56)
    views = {
        "E5_full_mean": e,
        "GRAIN15_mean": g,
        "GRAIN15_mean_and_second_moments": M.summary_columns(g, True),
    }
    ranks, rows, saved = [], [], {}
    errors = {
        "summary_preservation": 0.0,
        "all_stratum_margins": 0.0,
        "probability_normalization": 0.0,
        "direct_reading_identity": 0.0,
    }
    for view, x in views.items():
        null = M.conditional_null_space(x, d)
        ranks.append(
            {
                "view": view,
                "columns": x.shape[1],
                "rank": null["rank"],
                "conditional_law_nullity": null["nullity"],
                "cutoff": float(null["cutoff"]),
                "singular_values": null["singular_values"].tolist(),
            }
        )
        saved[view + "__basis"] = null["basis"]
        for j, (r, p, q, u) in enumerate(np.ndindex(2, 2, 7, 2)):
            if time.monotonic() - started > 60:
                raise TimeoutError("Original annex cap reached")
            witness = M.positive_witness(y[:, j], d, null["basis"])
            row = {
                "view": view,
                "program": ("H", "full")[r],
                "actor": ("A", "B")[p],
                "unit": ("score", "logodds")[u],
                "target": manifest["previous_targets"][q] if q < 6 else "action_fact",
                "status": witness["status"],
                "residual_norm": witness["residual_norm"],
                "account_difference": witness["difference"],
                "minimum_probability": None,
                "mean_preservation_error": None,
                "second_moment_difference_max": None,
            }
            if witness["plus"] is not None:
                plus, minus = witness["plus"], witness["minus"]
                change = plus - minus
                row["minimum_probability"] = float(min(plus.min(), minus.min()))
                err = float(np.max(abs(change @ x)))
                row["mean_preservation_error"] = err
                errors["summary_preservation"] = max(
                    errors["summary_preservation"], err
                )
                margins = change.reshape(24, 2, 2)
                errors["all_stratum_margins"] = max(
                    errors["all_stratum_margins"],
                    float(np.max(abs(margins.sum(1)))),
                    float(np.max(abs(margins.sum(2)))),
                )
                errors["probability_normalization"] = max(
                    errors["probability_normalization"],
                    abs(float(plus.sum()) - 1),
                    abs(float(minus.sum()) - 1),
                )
                direct = sum(float(change[i] * y[i, j]) for i in range(96))
                alternate = (
                    2 * witness["epsilon"] * float(witness["alpha"] @ (d.T @ y[:, j]))
                )
                errors["direct_reading_identity"] = max(
                    errors["direct_reading_identity"],
                    abs(direct - witness["difference"]),
                    abs(direct - alternate),
                )
                row["second_moment_difference_max"] = float(
                    np.max(
                        abs(change @ views["GRAIN15_mean_and_second_moments"][:, 15:])
                    )
                )
                if row["minimum_probability"] <= 0:
                    raise ValueError("Law is not strictly positive")
                saved[f"{view}__q{j}__plus"] = plus
                saved[f"{view}__q{j}__minus"] = minus
            rows.append(row)
    if len(rows) != 168 or max(errors.values()) > 1e-10:
        raise ValueError("Incomplete or non-preserving witness battery")
    H.table(out / "all_targets.csv", rows)
    np.savez(out / "witness_laws.npz", **saved)
    result = {
        "status": "VERIFIED_POST_RESULT_FINITE_MEAN_FIBERS_NOT_FULL_GRAIN_FAILURE",
        "ranks": ranks,
        "rows": len(rows),
        "errors": errors,
        "elapsed_seconds": time.monotonic() - started,
        "witness_counts": {
            view: sum(
                r["view"] == view and r["account_difference"] is not None for r in rows
            )
            for view in views
        },
        "original_scope_complete": False,
        "post_result_diagnostic": True,
    }
    H.write(out / "summary.json", result)
    H.write(out / "seal.json", {str(p.resolve()): H.sha(p) for p in out.iterdir()})
    print(result, flush=True)


if __name__ == "__main__":
    main()
