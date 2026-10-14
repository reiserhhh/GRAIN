#!/usr/bin/env python3
"""Result-after HM62: does a full hidden-vector mean retain higher moments?"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_author_moment_measurement as M
from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_author_moment_measurement import OUT, check
from scripts.run_grain_math_construct_routing import write
from scripts.run_grain_math_response import write_csv


def analyze_map(coefficients):
    """Test finite mean-embedding injectivity using a fixed numerical convention.

    Full row rank of the nonconstant character map means distinct distributions
    have distinct full vector means on this finite support. It says nothing
    about unobserved text support or psychological construct meaning.
    """
    c = np.asarray(coefficients, float)
    if (
        c.ndim != 2
        or c.shape[0] < 2
        or c.shape[1] == 0
        or not np.isfinite(c).all()
    ):
        raise ValueError("finite character-by-vector map required")
    u, s, vt = np.linalg.svd(c[1:], full_matrices=False)
    tol = 100 * np.finfo(float).eps * max(c[1:].shape) * s[0]
    kept = s > tol
    right_inverse = (vt[kept].T / s[kept]) @ u[:, kept].T
    return {
        "rank": int(kept.sum()),
        "possible_nonconstant_rank": len(c) - 1,
        "largest_singular": float(s[0]),
        "smallest_singular": float(s[-1]),
        "fixed_machine_tolerance": float(tol),
        "full_rank": bool(kept.sum() == len(c) - 1),
        "retained_condition": float(s[0] / s[kept][-1]) if kept.any() else None,
        "inverse_norm": float(1 / s[kept][-1]) if kept.any() else None,
        "right_inverse": right_inverse,
    }


def main():
    """Bind an exploratory supplement without changing the frozen HM62 packet."""
    m = check(OUT)
    if (
        read_json(OUT / "verification.json")["status"]
        != "VERIFIED_FINITE_HM62_NOT_GLOBAL"
    ):
        raise ValueError("Verified parent required")
    dest = OUT / "04_mean_embedding"
    if dest.exists():
        raise FileExistsError("Preserve result-after mean audit")
    dest.mkdir()
    write(
        dest / "manifest.json",
        {
            "result_after": True,
            "parent_verification": digest(OUT / "verification.json"),
            "source_coefficients": digest(
                ROOT / m["spec"]["source"] / "03/coefficients.npz"
            ),
            "code_sha256": digest(Path(__file__)),
            "machine_multiplier": 100,
            "new_model_calls_fits": [0, 0],
            "unseen_text_identifiability": False,
        },
    )
    _, masks, w = M.cube()
    profiles, summaries = [], []
    with np.load(
        ROOT / m["spec"]["source"] / "03/coefficients.npz", allow_pickle=False
    ) as old:
        for family in (0, 1):
            c = old[f"family{family}_body_endpoint_coefficients"]
            h = w @ c
            result = analyze_map(c)
            inv = result.pop("right_inverse")
            gram = c[1:] @ c[1:].T
            eigen = np.linalg.eigvalsh(gram)
            spectrum_error = float(
                np.max(
                    np.abs(eigen - np.linalg.svd(c[1:], compute_uv=False)[::-1] ** 2)
                )
            )
            errors = []
            for j in range(1, 64):
                pair = M.parity_profiles(masks[j], m["spec"]["parity_strength"])
                for side, p in zip((1, -1), pair):
                    observed_mean = p @ h
                    recovered = (observed_mean - c[0]) @ inv
                    truth = w.T @ p
                    pmf = w @ np.r_[1.0, recovered] / 64
                    error = float(np.max(np.abs(recovered - truth[1:])))
                    pmf_error = float(np.max(np.abs(pmf - p)))
                    errors.extend((error, pmf_error))
                    profiles.append(
                        {
                            "family": family,
                            "subset": j,
                            "order": int(masks[j].sum()),
                            "side": side,
                            "moment_recovery_error": error,
                            "distribution_recovery_error": pmf_error,
                            "mean_vector_shift_norm": float(
                                np.linalg.norm(observed_mean - c[0])
                            ),
                        }
                    )
            result.update(
                family=family,
                gram_vs_svd_squared_error=spectrum_error,
                maximum_recovery_error=max(errors),
            )
            summaries.append(result)
    if any(
        r["maximum_recovery_error"] > 1e-8 or r["gram_vs_svd_squared_error"] > 1e-8
        for r in summaries
    ):
        raise ArithmeticError("Full mean-embedding numerical audit failed")
    write_csv(dest / "profiles.csv", profiles)
    write(
        dest / "summary.json",
        {
            "families": summaries,
            "finite_support_only": True,
            "full_vector_mean_is_not_the_same_as_six_behavior_means": True,
            "rank_is_not_psychological_factor_count": True,
            "right_inverse_is_not_a_new_text_predictor": True,
            "source_model_encoding_still_required_for_new_text": True,
            "original_scope_complete": False,
        },
    )
    write(
        dest / "verification.json",
        {
            "status": "VERIFIED_RESULT_AFTER_HM62_MEAN_EMBEDDING",
            "same_agent_alternate_gram_arithmetic": True,
            "maximum_recovery_error": max(
                r["maximum_recovery_error"] for r in summaries
            ),
            "files": {
                name: digest(dest / name)
                for name in ("manifest.json", "summary.json", "profiles.csv")
            },
        },
    )
    print(summaries)


if __name__ == "__main__":
    main()
