#!/usr/bin/env python3
"""Result-after exact account of HM31 diagonal aliasing inside verified HM32."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_position_support import check, now
from scripts.run_grain_math_response import write_csv, write_json
from scripts.verify_grain_math_crossed_field import table

PARENT = ROOT / "results/grain_math_response_v1/query_context_cube_r1"
VERIFICATION = (
    ROOT / "results/grain_math_response_v1/query_context_cube_verification_r2"
)
OUT = ROOT / "results/grain_math_response_v1/query_context_alias_account_r1"


def account(x, coefficients):
    """Collapse Q=R, retaining signed cross terms rather than comparing unlike laws."""
    x, c = np.asarray(x, float), np.asarray(coefficients, float)
    if (
        x.shape[:3] != (2, 2, 2)
        or c.shape != (8,) + x.shape[3:]
        or not np.isfinite(x).all()
        or not np.isfinite(c).all()
    ):
        raise ValueError("complete finite cube and eight coefficients required")
    diagonal = np.array([[x[i, j, j] for j in range(2)] for i in range(2)])
    collapsed = np.array([c[0] + c[3], c[1] + c[2], c[4] + c[7], c[5] + c[6]])
    direct = np.array(
        [
            sum(
                ((-1) ** ((mask >> 1) * i + (mask & 1) * j)) * diagonal[i, j]
                for i in range(2)
                for j in range(2)
            )
            / 4
            for mask in range(4)
        ]
    )
    square = lambda a: float(np.sum(a * a))
    cross_rq = float(2 * np.sum(c[1] * c[2]))
    cross_hrhq = float(2 * np.sum(c[5] * c[6]))
    diagonal_bound = float(
        np.sum((diagonal - diagonal.mean(axis=1, keepdims=True)) ** 2) / 4
    )
    separate = sum(square(c[k]) for k in (1, 2, 5, 6))
    full_bound = float(np.sum((x - x.mean(axis=(1, 2), keepdims=True)) ** 2) / 8)
    rq_higher = square(c[3]) + square(c[7])
    stats = {
        "diagonal_H_only_mse_bound": diagonal_bound,
        "full_cube_H_only_mse_bound": full_bound,
        "separate_R_Q_HR_HQ_energy": separate,
        "R_Q_signed_cross": cross_rq,
        "HR_HQ_signed_cross": cross_hrhq,
        "RQ_HRQ_energy": rq_higher,
        "diagonal_mean_square_energy": float(np.sum(diagonal * diagonal) / 4),
        "full_cube_mean_square_energy": float(np.sum(x * x) / 8),
        "alias_coefficient_error": float(np.max(abs(collapsed - direct))),
        "diagonal_bound_identity_error": abs(
            diagonal_bound - separate - cross_rq - cross_hrhq
        ),
        "cube_diagonal_difference_error": abs(
            full_bound - diagonal_bound - rq_higher + cross_rq + cross_hrhq
        ),
    }
    return diagonal, collapsed, stats


def run():
    """Verify every inherited array/part before recording finite alias accounting."""
    if OUT.exists():
        raise FileExistsError("preserve previous alias diagnostic")
    m = check(PARENT)
    if (
        read_json(VERIFICATION / "summary.json")["status"]
        != "VERIFIED_FINITE_HM32_NOT_GLOBAL"
    ):
        raise ValueError("verified HM32 required")
    old = ROOT / m["spec"]["parent"]
    for owner, name in (
        (PARENT, "02/artifact_hashes.json"),
        (VERIFICATION, "artifact_hashes.json"),
        (old, "02/artifact_hashes.json"),
    ):
        for path, sha in read_json(owner / name).items():
            if digest(owner / path) != sha:
                raise ValueError("frozen dependency changed " + path)
    for name, sha in read_json(VERIFICATION / "csv_adapter_manifest.json")[
        "files"
    ].items():
        if digest(ROOT / name) != sha:
            raise ValueError("companion verifier code changed " + name)
    manifest = {
        "analysis": "result-after exact diagonal alias accounting, no fitted predictor",
        "primary_manifest_sha256": digest(PARENT / "00/manifest.json"),
        "primary_ledger_sha256": digest(PARENT / "02/artifact_hashes.json"),
        "verification_sha256": digest(VERIFICATION / "summary.json"),
        "files": {
            name: digest(ROOT / name)
            for name in (
                "scripts/run_grain_math_query_alias_account.py",
                "tests/test_grain_math_query_alias_account.py",
            )
        },
        "frozen_before_diagnostic_outputs": now(),
        "new_fits": 0,
        "new_model_calls": 0,
    }
    OUT.mkdir(parents=True)
    write_json(OUT / "manifest.json", manifest)
    rows, error = [], 0.0
    with (
        np.load(PARENT / "02/cube_arrays.npz", allow_pickle=False) as full,
        np.load(old / "02/corner_arrays.npz", allow_pickle=False) as inherited,
    ):
        for row in table(PARENT / "02/cube_decomposition.csv"):
            key = f"p{row['pair']}_w{row['wording']}_{row['query']}_{row['space']}_{row['arm']}"
            diagonal, c, stats = account(full[key], full[key + "_coefficients"])
            parts = {
                "mean": c[0],
                "donor": np.array([c[2], -c[2]]),
                "receiver": np.array([c[1], -c[1]]),
                "interaction": np.array([[c[3], -c[3]], [-c[3], c[3]]]),
            }
            diff = float(np.max(abs(diagonal - inherited[key])))
            for name, part in parts.items():
                diff = max(diff, float(np.max(abs(part - inherited[key + "_" + name]))))
            error = max(error, diff, *(stats[k] for k in stats if k.endswith("_error")))
            rows.append(
                {
                    k: row[k]
                    for k in ("pair", "wording", "query", "role", "space", "arm")
                }
                | {"inherited_diagonal_array_part_error": diff, **stats}
            )
    if len(rows) != 576 or error > 1e-10:
        raise ValueError("complete exact diagonal replay failed " + str(error))
    pooled = []
    for role, query, space, arm in dict.fromkeys(
        (r["role"], r["query"], r["space"], r["arm"]) for r in rows
    ):
        selected = [
            r
            for r in rows
            if (r["role"], r["query"], r["space"], r["arm"])
            == (role, query, space, arm)
        ]
        keys = [
            k for k in selected[0] if k.endswith(("_energy", "_mse_bound", "_cross"))
        ]
        values = {k: float(np.mean([r[k] for r in selected])) for k in keys}
        for law in ("diagonal", "full_cube"):
            energy = values[law + "_mean_square_energy"]
            values[law + "_relative_H_only_rmse_bound"] = (
                float(np.sqrt(values[law + "_H_only_mse_bound"] / energy))
                if energy
                else None
            )
        base = values["diagonal_H_only_mse_bound"]
        values["signed_cross_fraction_of_diagonal_bound"] = (
            (values["R_Q_signed_cross"] + values["HR_HQ_signed_cross"]) / base
            if base
            else None
        )
        pooled.append(
            {
                "role": role,
                "query": query,
                "space": space,
                "arm": arm,
                "cubes": len(selected),
                **values,
            }
        )
    write_csv(OUT / "alias_account.csv", rows)
    write_csv(OUT / "pooled_alias_account.csv", pooled)
    write_json(
        OUT / "summary.json",
        {
            "status": "VERIFIED_FINITE_HM32_DIAGONAL_ALIAS_ACCOUNT",
            "rows": len(rows),
            "pooled_rows": len(pooled),
            "maximum_array_and_identity_error": error,
            "new_model_calls": 0,
            "new_fits": 0,
            "scope": "Q=R restricted design differs from full product cube; signed cross terms are finite model-action accounting, not author correlations",
        },
    )
    write_json(
        OUT / "artifact_hashes.json",
        {
            str(p.relative_to(OUT)): digest(p)
            for p in OUT.iterdir()
            if p.is_file()
            and not p.name.startswith("._")
            and p.name != "artifact_hashes.json"
        },
    )
    print(read_json(OUT / "summary.json"), flush=True)


if __name__ == "__main__":
    run()
