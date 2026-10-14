#!/usr/bin/env python3
"""HM78 alternate cofactor/cell verification of author-comparison evidence."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_score_fiber as M
from scripts.run_grain_score_fiber_hm78 import (
    SPEC,
    check,
    digest,
    load_sources,
    read,
    write,
)


def table(path):
    """Read a full finite audit table rather than trusting headline summaries."""
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def verify(out):
    """Recompute all endpoints, bounds, comparisons and dual inequalities."""
    spec = check(out)["spec"]
    for name, sha in read(out / "03/output_seal.json").items():
        if digest(out / name) != sha:
            raise ValueError("Frozen output drift:" + name)
    summary = read(out / "03/summary.json")
    if read(out / "checkpoint.json")["status"] not in (
        "COLLECTED_HM78",
        "VERIFIED_HM78",
    ):
        raise ValueError("Full terminal computation required")
    pairs, sources, profiles, cold = load_sources(spec)
    errors, checks = [], 0

    def compare(a, b, tolerance=1e-8):
        nonlocal checks
        delta = np.asarray(a) - np.asarray(b)
        error = float(np.max(abs(delta))) if delta.size else 0.0
        if not np.isfinite(error) or error > tolerance:
            raise ArithmeticError(f"Alternate check failed:{error}")
        errors.append(error)
        checks += max(delta.size, 1)

    with np.load(out / "03/fibers_and_certificates.npz", allow_pickle=False) as data:
        compare(len(data["nulls"]), 60)
        for pair_id, _ in enumerate(pairs):
            for family in (0, 1):
                key = f"p{pair_id:02d}_f{family}"
                p = data[key + "_profiles"]
                compare(p[0], np.full(8, 1 / 8))
                compare(p[1:], profiles[pair_id].reshape(16, 8))
                a0 = data[key + "_design_scores_only"]
                h = M.cofactor_null(a0)
                saved_h = data[key + "_null"]
                compare(np.outer(h, h), np.outer(saved_h, saved_h))
                compare(a0 @ h, np.zeros(7))
                for scheme in spec["schemes"]:
                    a, end = (
                        data[key + "_design_" + scheme],
                        data[key + "_endpoints_" + scheme],
                    )
                    compare(a, M.observation_design(sources[pair_id, family], scheme))
                    compare((end - p[:, None]) @ a.T, np.zeros((17, 2, len(a))))
                    if end.min() < -1e-10:
                        raise ArithmeticError("Negative fiber endpoint")
                    compare(end.sum(-1), np.ones((17, 2)))
                    if M.row_chart(a)["rank"] == 7:
                        for pv, endpoints in zip(p, end):
                            lo = max(-pv[i] / h[i] for i in range(8) if h[i] > 1e-13)
                            hi = min(-pv[i] / h[i] for i in range(8) if h[i] < -1e-13)
                            alternative = np.asarray([pv + lo * h, pv + hi * h])
                            direct_error = np.max(abs(alternative - endpoints))
                            reversed_error = np.max(abs(alternative[::-1] - endpoints))
                            compare(min(direct_error, reversed_error), 0)
                    else:
                        compare(end, np.repeat(p[:, None], 2, axis=1))
                    for bank in spec["application_banks"]:
                        v = (
                            sources[pair_id, 1 - family]
                            if bank == "other_source_family"
                            else cold[bank][pair_id]
                        )
                        compare(data[key + "_app_" + bank], v)
                        products = np.asarray(
                            [[e @ v for e in endpoints] for endpoints in end]
                        )
                        independent = np.stack(
                            (products.min(1), products.max(1)), axis=1
                        )
                        compare(
                            data[key + "_interval_" + scheme + "_" + bank], independent
                        )
        bounds = table(out / "03/target_bounds.csv")
        if len(bounds) != 110160:
            raise ValueError("Missing target bounds")
        for row in bounds:
            key = f"p{int(row['pair_id']):02d}_f{row['family']}"
            qi = spec["targets"].index(row["query"])
            profile_id = int(row["profile_id"])
            val = data[key + "_interval_" + row["scheme"] + "_" + row["bank"]][
                profile_id, :, qi
            ]
            compare(val, [float(row["lower"]), float(row["upper"])])
            actual = (
                data[key + "_profiles"][profile_id]
                @ data[key + "_app_" + row["bank"]][:, qi]
            )
            compare(actual, float(row["actual_profile_value"]))
            if actual < val[0] - 1e-8 or actual > val[1] + 1e-8:
                raise ArithmeticError("Actual profile outside its identification set")
        comparisons = table(out / "03/author_comparisons.csv")
        if len(comparisons) != 51840:
            raise ValueError("Missing author comparisons")
        for row in comparisons:
            key = f"p{int(row['pair_id']):02d}_f{row['family']}"
            case_id, qi = (
                spec["cases"].index(row["case"]),
                spec["targets"].index(row["query"]),
            )
            i, j = 1 + 2 * case_id, 2 + 2 * case_id
            p = data[key + "_profiles"]
            e = data[key + "_endpoints_" + row["scheme"]]
            v = data[key + "_app_" + row["bank"]][:, qi]
            four = np.asarray([(a - b) @ v for a in e[i] for b in e[j]])
            compare(
                [four.min(), four.max()], [float(row["lower"]), float(row["upper"])]
            )
            compare((p[i] - p[j]) @ v, float(row["actual_comparison"]))
            reversal = (
                four.min() < -spec["tolerance"] and four.max() > spec["tolerance"]
            )
            if reversal != (row["both_rankings_attainable"] == "True"):
                raise ArithmeticError("Incorrect author ranking claim")
        lp_rows = table(out / "03/lp_certificates.csv")
        if len(lp_rows) != 1080 or summary["linear_programs"] != 2160:
            raise ValueError("Missing primal/dual certificates")
        for row in lp_rows:
            k = int(row["row_id"])
            key = f"p{int(row['pair_id']):02d}_f{row['family']}"
            a = data[key + "_design_scores_only"]
            p = data[key + "_profiles"][int(row["profile_id"])]
            v = data[key + "_app_" + row["bank"]][
                :, spec["targets"].index(row["query"])
            ]
            for side, sign, field in ((0, 1, "lower"), (1, -1, "upper")):
                witness, lam = (
                    data["lp_primal"][k, side],
                    data["lp_original_dual"][k, side],
                )
                compare(a @ witness, a @ p)
                compare(witness @ v, float(row[field]))
                compare((a @ p) @ lam, float(row[field]))
                if witness.min() < -1e-9 or np.min(sign * (v - a.T @ lam)) < -1e-8:
                    raise ArithmeticError(
                        "Invalid original-unit primal/dual inequality"
                    )
                checks += 16
        eigenvalues = np.linalg.eigvalsh(data["nulls"].T @ data["nulls"] / 60)[::-1]
        compare(summary["null_average_projector_eigenvalues"], eigenvalues)
    write(
        out / "verification.json",
        {
            "status": "VERIFIED_FINITE_HM78_NOT_GLOBAL",
            "alternate_scalar_checks": checks,
            "maximum_error": max(errors),
            "checks": "cofactor nulls,all simplex endpoints,bounds,pairwise attainable contrasts,original-unit LP duals",
            "same_agent_alternate_not_third_party": True,
            "original_scope_complete": False,
        },
    )
    state = read(out / "checkpoint.json")
    state.update(status="VERIFIED_HM78", active_processes=[])
    write(out / "checkpoint.json", state)
    print(read(out / "verification.json"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / read(SPEC)["output"])
    args = parser.parse_args()
    verify(args.out)
