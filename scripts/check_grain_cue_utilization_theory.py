#!/usr/bin/env python3
"""Finite numerical witnesses for standard cue-judgment decompositions, no corpus/model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def centered(x):
    """Validate one finite, nonempty scalar observation vector and center it."""
    a = np.asarray(x, dtype=float)
    if a.ndim != 1 or len(a) < 4 or not np.isfinite(a).all():
        raise ValueError("Require at least four finite scalar observations")
    return a - a.mean()


def four_terms(y, judgment, predicted_y, predicted_judgment):
    """Exact covariance accounting, including cross terms for arbitrary predictors."""
    y, s, my, ms = map(centered, (y, judgment, predicted_y, predicted_judgment))
    if any(a.shape != y.shape for a in (s, my, ms)):
        raise ValueError("All four vectors must contain the same paired observations")
    denominator = np.linalg.norm(y) * np.linalg.norm(s)
    if denominator <= 1e-14:
        raise ValueError(
            "Correlation is undefined for a constant criterion or judgment"
        )
    ey, es = y - my, s - ms
    terms = {
        "modeled_alignment": float(my @ ms / denominator),
        "unmodeled_alignment": float(ey @ es / denominator),
        "modeled_y_unmodeled_s": float(my @ es / denominator),
        "unmodeled_y_modeled_s": float(ey @ ms / denominator),
    }
    actual = float(y @ s / denominator)
    return {
        **terms,
        "actual_correlation": actual,
        "sum_four_terms": sum(terms.values()),
        "closure_absolute_error": abs(actual - sum(terms.values())),
        "two_term_omission_error": abs(
            actual - terms["modeled_alignment"] - terms["unmodeled_alignment"]
        ),
    }


def linear_lens(y, judgment, cues):
    """Same-sample orthogonal projection only; not out-of-sample performance estimation."""
    y, s = centered(y), centered(judgment)
    z = np.asarray(cues, dtype=float)
    if z.ndim != 2 or z.shape[0] != len(y) or not np.isfinite(z).all():
        raise ValueError("Require a finite paired cue matrix")
    z = z - z.mean(axis=0)
    my = z @ np.linalg.lstsq(z, y, rcond=None)[0]
    ms = z @ np.linalg.lstsq(z, s, rcond=None)[0]
    accounting = four_terms(y, s, my, ms)
    ny, ns = np.linalg.norm(y), np.linalg.norm(s)
    nmy, nms = np.linalg.norm(my), np.linalg.norm(ms)
    ey, es = y - my, s - ms
    ney, nes = np.linalg.norm(ey), np.linalg.norm(es)
    return {
        **accounting,
        "R_y": float(nmy / ny),
        "R_s": float(nms / ns),
        "G": float(my @ ms / (nmy * nms)) if nmy * nms > 1e-14 else None,
        "C": float(ey @ es / (ney * nes)) if ney * nes > 1e-14 else None,
        "cue_rank": int(np.linalg.matrix_rank(z)),
    }


def finite_worlds():
    """Construct deterministic algebra checks; results are not learned human properties."""
    rng = np.random.default_rng(731)
    z = rng.normal(size=(128, 5))
    z -= z.mean(axis=0)
    q, _ = np.linalg.qr(z)
    x1, x2, u, v, _ = q.T
    y = 0.8 * x1 + 0.6 * u
    cases = {}
    for name, angle in (("aligned", 0), ("orthogonal", np.pi / 2), ("reversed", np.pi)):
        s = 0.8 * (np.cos(angle) * x1 + np.sin(angle) * x2) + 0.6 * v
        cases[name] = linear_lens(y, s, q[:, :2])
    s = 0.8 * x1 + 0.6 * v
    cases["invertible_cue_coordinates"] = linear_lens(
        y, s, q[:, :2] @ np.array([[1.0, 3.0], [2.0, 5.0]])
    )
    shared = 0.2 * x1 + np.sqrt(0.96) * u
    cases["omitted_shared_cue"] = linear_lens(shared, shared, x1[:, None])
    cases["shared_cue_included"] = linear_lens(shared, shared, np.column_stack([x1, u]))
    cases["shrinkage_requires_cross_terms"] = four_terms(x1, x1, 0.5 * x1, 0.5 * x1)
    # Two equally common event groups with known Bernoulli probabilities.
    event_rate = np.array([0.2, 0.8])
    forecast = np.array([0.1, 0.4])
    prevalence = float(event_rate.mean())
    uncertainty = prevalence * (1 - prevalence)
    resolution = float(np.mean((event_rate - prevalence) ** 2))
    reliability_error = float(np.mean((forecast - event_rate) ** 2))
    expected_brier = float(
        np.mean(event_rate * (1 - forecast) ** 2 + (1 - event_rate) * forecast**2)
    )
    positive_weights = event_rate / (2 * prevalence)
    negative_weights = (1 - event_rate) / (2 * (1 - prevalence))
    comparisons = (forecast[:, None] > forecast[None, :]).astype(float)
    comparisons += 0.5 * (forecast[:, None] == forecast[None, :])
    auc = float(
        np.sum(positive_weights[:, None] * negative_weights[None, :] * comparisons)
    )
    balanced = float(
        (
            positive_weights[forecast >= 0.5].sum()
            + negative_weights[forecast < 0.5].sum()
        )
        / 2
    )
    cases["binary_resolution_without_threshold_crossing"] = {
        "uncertainty": uncertainty,
        "resolution": resolution,
        "reliability_error": reliability_error,
        "expected_brier": expected_brier,
        "brier_decomposition_error": abs(
            expected_brier - (uncertainty - resolution + reliability_error)
        ),
        "balanced_accuracy_at_half": balanced,
        "population_auc_with_ties": auc,
        "scope": "Known two-cell probability world; not estimated from a corpus",
    }
    cases["duplicate_evidence"] = {
        "posterior_after_one_observation": 0.8,
        "posterior_after_perfect_duplicate_correctly_conditioned": 0.8,
        "posterior_if_duplicate_is_wrongly_independent": float(4**2 / (1 + 4**2)),
        "scope": "Equal prior and one likelihood ratio4; exact duplicate has conditional likelihood ratio1",
    }
    return cases


def main():
    """Write a small synthetic-only proof-check packet without accessing project data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    cases = finite_worlds()
    error = max(c.get("closure_absolute_error", 0) for c in cases.values())
    if (
        error > 1e-12
        or cases["binary_resolution_without_threshold_crossing"][
            "brier_decomposition_error"
        ]
        > 1e-12
    ):
        raise ArithmeticError("Algebraic accounting failed")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    out = args.output_dir / "synthetic_witnesses.json"
    out.write_text(
        json.dumps(cases, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    summary = {
        "status": "SYNTHETIC_IDENTITIES_VERIFIED",
        "cases": len(cases),
        "max_covariance_closure_error": error,
        "source_labels_read": False,
        "bridge_labels_read": False,
        "neural_forwards": 0,
        "new_empirical_personality_claim": False,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "outputs_sha256": {out.name: hashlib.sha256(out.read_bytes()).hexdigest()},
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
