#!/usr/bin/env python3
"""One separate post-outcome qualification of author-local, not common-axis, response."""

import json
import signal
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_criterion_components as C
from grain_research.construction import math_output_observability as M
from grain_research.construction import math_source_anchor_exchange as E
from scripts import run_grain_observed_family_hm170 as R
from scripts.score_grain_author_belief_cold import load_bundle

PARENT = ROOT / "results/grain_math_response_v1/output_observability_r1"
DEFAULT = PARENT / "author_local_followup_r1"
SCIENCE = (
    "docs/GRAIN_OUTPUT_AUTHOR_LOCAL_FOLLOWUP_HM284.md",
    "scripts/analyze_grain_output_author_local_hm284.py",
    "tests/test_grain_output_author_local.py",
    "grain_research/construction/math_output_observability.py",
)


def diagonal_mean(left_weights, right_weights, covariance, diagonal):
    """Exact shared-assignment quadratic mean, not a zero null or Gaussian z."""
    a, b = np.asarray(left_weights, float), np.asarray(right_weights, float)
    a, b = a - a.mean(0), b - b.mean(0)
    return float(np.mean(np.diag(a @ covariance @ b.T) * diagonal))


def expired(_signal, _frame):
    """Stop the finite analysis without a reference-count/metric rescue."""
    raise TimeoutError("Original30-second author-local analysis cap")


def main():
    """Bind old numeric observations and new question before evaluating all references."""
    out = DEFAULT
    if out.exists():
        raise FileExistsError("No post-outcome analysis retry")
    R.check(
        PARENT, ("computed_seal.json", "analyzed_seal.json", "verification_seal.json")
    )
    out.mkdir(parents=True)
    R.write(
        out / "manifest.json",
        {
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": {
                str((PARENT / p).relative_to(ROOT)): R.digest(PARENT / p)
                for p in (
                    "source.npz",
                    "geometry.npz",
                    "manifest.json",
                    "prepared_seal.json",
                    "computed_seal.json",
                    "analyzed_seal.json",
                    "verification_seal.json",
                )
            },
            "post_outcome": True,
            "primary": "GRAIN15/pooled/usual_author/within/full/same-target diagonal,all5",
            "old_reference_draws_per_target": 199,
            "seconds_cap": 30,
            "original_scope_complete": False,
        },
    )
    R.write(
        out / "registered_question.json",
        {
            "post_outcome_not_primary_replacement": True,
            "no_new_output_geometry_or_draws": True,
        },
    )
    R.seal(out, "prepared_seal.json", ("registered_question.json",))
    source, geom = (
        load_bundle(PARENT / "source.npz"),
        load_bundle(PARENT / "geometry.npz"),
    )
    left, right = np.arange(0, 48, 2), np.arange(1, 48, 2)
    e = source["labels"] - source["means"]
    started, rows, errors, references = time.monotonic(), [], 0.0, {}
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(30)
    try:
        for sp, space in enumerate(("E5", "GRAIN15")):
            d = source["weights"][sp, 2].mean(1)
            for q, target in enumerate(("E", "N", "A", "C", "O")):
                a, b = (
                    M.centered_scores(d[left] @ e[:, q]),
                    M.centered_scores(d[right] @ e[:, q]),
                )
                en = e[source["assignments"][q], q]
                an, bn = (
                    M.centered_scores(en @ d[left].T),
                    M.centered_scores(en @ d[right].T),
                )
                covariance = C.orbit_covariance(
                    source["labels"][:, q], source["groups"][:, q]
                )
                for op, operation in enumerate(("usual_author", "sample_expression")):
                    i0, i1 = left * 10 + op * 5 + q, right * 10 + op * 5 + q
                    diagonal = geom["grams"][0, i0, i1]
                    observed = float(np.mean(a * b * diagonal))
                    null = np.mean(an * bn * diagonal, axis=1)
                    expected = diagonal_mean(d[left], d[right], covariance, diagonal)
                    for index in range(199):
                        direct = (
                            sum(
                                float(an[index, u] * bn[index, u] * diagonal[u])
                                for u in range(24)
                            )
                            / 24
                        )
                        errors = max(errors, abs(direct - null[index]))
                    rows.append(
                        {
                            "space": space,
                            "operation": operation,
                            "target": target,
                            "same_author_product": observed,
                            "exact_assignment_mean": expected,
                            "old_reference_mean": float(null.mean()),
                            "upper_conditional_rank": float(
                                (1 + np.sum(null >= observed)) / 200
                            ),
                            "post_outcome": True,
                        }
                    )
                    references[f"{space}/{operation}/{target}"] = null
        primary = [
            r.copy()
            for r in rows
            if r["space"] == "GRAIN15" and r["operation"] == "usual_author"
        ]
        for row, adjusted in zip(
            primary, E.holm([r["upper_conditional_rank"] for r in primary]), strict=True
        ):
            row["Holm5_rank"] = float(adjusted)
        if errors > 1e-20:
            raise ArithmeticError("Different author sums do not reproduce")
        R.table(out / "all_author_local.csv", rows)
        R.table(out / "primary.csv", primary)
        R.archive(out / "old_references.npz", **references)
        R.write(
            out / "summary.json",
            {
                "primary": primary,
                "seconds": time.monotonic() - started,
                "literal_reference_error": errors,
                "same_agent_not_blind": True,
                "post_outcome_not_original_primary": True,
                "new_transformer_API_private_training": 0,
                "original_scope_complete": False,
            },
        )
        R.seal(
            out,
            "completed_seal.json",
            (
                "all_author_local.csv",
                "primary.csv",
                "old_references.npz",
                "summary.json",
            ),
        )
        print(json.dumps(R.read(out / "summary.json")))
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
