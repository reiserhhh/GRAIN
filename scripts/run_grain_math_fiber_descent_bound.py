#!/usr/bin/env python3
"""HM31 result-after mathematical bound on identical-H action compilation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_position_support import check, now
from scripts.run_grain_math_response import write_csv, write_json
from scripts.verify_grain_math_crossed_field import table

PARENT = ROOT / "results/grain_math_response_v1/context_fiber_recombination_r1"
VERIFIER = (
    ROOT / "results/grain_math_response_v1/context_fiber_verification_r1/summary.json"
)
OUT = ROOT / "results/grain_math_response_v1/context_fiber_descent_bound_r1"


def bound(values):
    """Return the exact best shared-per-donor L2 error for two receivers."""
    x = np.asarray(values, float)
    if x.shape[:2] != (2, 2) or not np.isfinite(x).all():
        raise ValueError("finite complete two-by-two values required")
    difference = x[:, 0] - x[:, 1]
    dimensions = tuple(range(1, difference.ndim))
    squared = difference * difference
    if dimensions:
        squared = squared.sum(axis=dimensions)
    mse = float(squared.mean() / 4)
    mu = x.mean(axis=1)
    residual = x - mu[:, None]
    total = float(np.sum(x * x) / 4)
    return {
        "h_only_mse_lower_bound": mse,
        "h_only_rmse_lower_bound": float(np.sqrt(mse)),
        "observed_rms_norm": float(np.sqrt(total)),
        "bound_relative_to_observed_rms": float(np.sqrt(mse / total))
        if total
        else None,
        "oracle_identity_error": abs(mse - float(np.sum(residual * residual) / 4)),
    }


def run(parent, out):
    """Bound all roles/queries/arms without training a held-data predictor."""
    if out.exists():
        raise FileExistsError("preserve prior diagnostic")
    m = check(parent)
    if read_json(VERIFIER)["status"] != "VERIFIED_FINITE_HM31_NOT_GLOBAL":
        raise ValueError("verified primary required")
    ledger = read_json(parent / "02/artifact_hashes.json")
    for name, sha in ledger.items():
        if digest(parent / name) != sha:
            raise ValueError("primary changed " + name)
    design = read_json(parent / "00/design.json")
    cells, rows, summaries = [], [], []
    decompositions = table(parent / "02/corner_decomposition.csv")
    with np.load(parent / "02/corner_arrays.npz", allow_pickle=False) as arrays:
        for d in decompositions:
            space = d["space"]
            if space not in ("raw_action", "unit_action", "gradient", "task_gradient"):
                continue
            p, w = int(d["pair"]), int(d["wording"])
            key = f"p{p}_w{w}_{d['query']}_{space}_{d['arm']}"
            result = bound(arrays[key])
            projection = float(d["receiver_energy"]) + float(d["interaction_energy"])
            if abs(result["h_only_mse_lower_bound"] - projection) > 1e-12:
                raise ValueError("conditional variance/projection identity failed")
            rows.append(
                {k: d[k] for k in ("role", "pair", "wording", "query", "space", "arm")}
                | {**result, "receiver_plus_interaction_energy": projection}
            )
            by = {
                (r["donor_side"], r["receiver_side"]): r
                for r in design
                if (r["pair"], r["wording"], r["query"]) == (p, w, d["query"])
            }
            for i in range(2):
                h = []
                for j in range(2):
                    r = by[i, j]
                    root = ROOT / m["spec"]["parent"] if r["diagonal"] else parent
                    unit = r["donor_unit"] if r["diagonal"] else r["unit"]
                    with np.load(
                        root / "01/baselines" / f"unit_{unit:03d}.npz",
                        allow_pickle=False,
                    ) as data:
                        h.append(data["h"].copy())
                if not np.array_equal(*h):
                    raise ValueError("bound requires identical observed H")
                if space in ("raw_action", "unit_action"):
                    a, b = arrays[key][i]
                    cells.append(
                        {
                            k: d[k]
                            for k in (
                                "role",
                                "pair",
                                "wording",
                                "query",
                                "space",
                                "arm",
                            )
                        }
                        | {
                            "donor_side": i,
                            "receiver0_action": float(a),
                            "receiver1_action": float(b),
                            "difference": float(a - b),
                            "best_shared_oracle_action": float((a + b) / 2),
                            "minimum_mse": float((a - b) ** 2 / 4),
                            "label": "result-after optimistic bound, not a held-fitted model",
                        }
                    )
    for role in ("source", "validation", "application"):
        for q in m["spec"]["queries"]:
            for space in ("raw_action", "unit_action", "gradient", "task_gradient"):
                for arm in m["spec"]["arms"] if "action" in space else ("none",):
                    selected = [
                        r
                        for r in rows
                        if (r["role"], r["query"], r["space"], r["arm"])
                        == (role, q, space, arm)
                    ]
                    mse = float(
                        np.mean([r["h_only_mse_lower_bound"] for r in selected])
                    )
                    observed = float(
                        np.mean([r["observed_rms_norm"] ** 2 for r in selected])
                    )
                    summaries.append(
                        {
                            "role": role,
                            "query": q,
                            "space": space,
                            "arm": arm,
                            "four_corner_units": len(selected),
                            "fixed_h_pairs": 2 * len(selected),
                            "h_only_mse_lower_bound": mse,
                            "h_only_rmse_lower_bound": float(np.sqrt(mse)),
                            "observed_rms_norm": float(np.sqrt(observed)),
                            "bound_relative_to_observed_rms": float(
                                np.sqrt(mse / observed)
                            )
                            if observed
                            else None,
                        }
                    )
    out.mkdir(parents=True)
    write_json(
        out / "manifest.json",
        {
            "analysis": "result-after optimistic oracle lower bound; no fitted predictor",
            "primary_manifest_sha256": digest(parent / "00/manifest.json"),
            "primary_ledger_sha256": digest(parent / "02/artifact_hashes.json"),
            "primary_verifier_sha256": digest(VERIFIER),
            "files": {
                "scripts/run_grain_math_fiber_descent_bound.py": digest(Path(__file__)),
                "tests/test_grain_math_fiber_descent_bound.py": digest(
                    ROOT / "tests/test_grain_math_fiber_descent_bound.py"
                ),
            },
            "frozen_before_diagnostic_outputs": now(),
            "new_fits": 0,
            "new_model_calls": 0,
        },
    )
    for name, values in (
        ("bounds_by_corner", rows),
        ("bounds_by_role", summaries),
        ("fixed_h_pairs", cells),
    ):
        write_csv(out / (name + ".csv"), values)
    write_json(
        out / "summary.json",
        {
            "status": "FINITE_H_ONLY_DESCENT_BOUND_VERIFIED_NOT_GLOBAL",
            "corner_rows": len(rows),
            "role_rows": len(summaries),
            "action_pair_rows": len(cells),
            "maximum_oracle_identity_error": max(
                r["oracle_identity_error"] for r in rows
            ),
            "interpretation": "a best possible shared action at fixed H cannot fit both receivers; empirical conditional variance is an optimistic lower bound, not a new predictor or personality validity",
            "new_fits": 0,
            "new_model_calls": 0,
        },
    )
    write_json(
        out / "artifact_hashes.json",
        {str(p.relative_to(out)): digest(p) for p in out.iterdir() if p.is_file()},
    )
    print(read_json(out / "summary.json"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=PARENT)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    run(args.parent, args.out)
