#!/usr/bin/env python3
"""Alternate centered IV/OLS and literal cold/moment re-sums, same agent."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_natural_proxy_hm253 as P
from scripts import run_grain_observed_family_hm170 as R


def main():
    """Verify source equations by a centered route without refitting science."""
    out = P.DEFAULT
    P.check(out, ("compiled_seal.json", "cold_seal.json", "analyzed_seal.json"))
    d = P.arrays(out / "observations.npz")
    programs = R.read(out / "bundle.json")["programs"]
    scores = P.arrays(out / "cold_scores.npz")["scores"]
    max_score, max_equation, max_relative_coefficient = 0.0, 0.0, 0.0
    accounts = 0
    for p in programs:
        train = np.where(d["folds"] == p["source_fold"])[0]
        held = np.where(d["folds"] != p["source_fold"])[0]
        a = d["features"][train, p["w"], 1:]
        b = d["features"][train, p["z"], 1:]
        y = d["labels"][train]
        ac, bc, yc = a - a.mean(0), b - b.mean(0), y - y.mean(0)
        slope = np.linalg.solve(bc.T @ ac, bc.T @ yc)
        iv = np.vstack([y.mean(0) - a.mean(0) @ slope, slope])
        oslope = np.linalg.lstsq(ac, yc, rcond=1e-12)[0]
        ordinary = np.vstack([y.mean(0) - a.mean(0) @ oslope, oslope])
        for mi, (method, alternative) in enumerate(
            (("bridge", iv), ("ordinary", ordinary))
        ):
            beta = np.asarray(p[method])
            max_relative_coefficient = max(
                max_relative_coefficient,
                float(np.linalg.norm(beta - alternative) / (1 + np.linalg.norm(beta))),
            )
            for vi, view in enumerate((p["w"], p["t"])):
                prediction = np.array(
                    [
                        [
                            sum(
                                float(v) * float(coef)
                                for v, coef in zip(row, beta[:, q])
                            )
                            for q in range(5)
                        ]
                        for row in d["features"][held, view]
                    ]
                )
                max_score = max(
                    max_score,
                    float(np.max(np.abs(prediction - scores[mi, p["w"], vi, held]))),
                )
                accounts += prediction.size
        k, m = np.asarray(p["kernel"]), np.asarray(p["moment"])
        max_equation = max(max_equation, float(np.max(np.abs(k @ iv - m))))
    if max_score > 1e-9 or max_equation > 1e-8 or max_relative_coefficient > 1e-7:
        raise ValueError("Alternate source or cold accounts disagree")
    result = {
        "PASS": True,
        "cold_scalar_accounts": accounts,
        "max_literal_cold_error": max_score,
        "max_centered_equation_error": max_equation,
        "max_relative_coefficient_error": max_relative_coefficient,
        "same_agent_not_blind": True,
        "no_scientific_rerun": True,
    }
    R.write(out / "verification.json", result)
    R.seal(out, "verification_seal.json", ("verification.json",))
    print(result)


if __name__ == "__main__":
    main()
