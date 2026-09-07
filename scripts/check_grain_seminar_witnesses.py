#!/usr/bin/env python3
"""Finite counterexamples for the Codex/Claude seminar; no text, models or labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def evidence_scores(events):
    """Compare explicit toy aggregators on (source_id, log_likelihood_ratio) pairs.

    Distinct IDs are independent observations by construction; repeated IDs are
    exact copies, not new evidence. The position rule is one fixed counterexample,
    not a fitted or universally assumed transformer attention rule.
    """
    if not events:
        return dict.fromkeys(
            ("content_mean", "position_mean", "naive_lr_sum", "source_aware_lr_sum"),
            0.0,
        )
    unique = {}
    for key, value in events:
        if not np.isfinite(value):
            raise ValueError("Evidence must be finite")
        if key in unique and unique[key] != value:
            raise ValueError("Same-source copies must agree in this finite world")
        unique[key] = value
    values = np.array([v for _, v in events], dtype=float)
    distance = np.arange(len(values) - 1, -1, -1, dtype=float)
    weights = np.exp(-0.2 * distance**2)
    weights /= weights.sum()
    return {
        "content_mean": float(values.mean()),
        "position_mean": float(weights @ values),
        "naive_lr_sum": float(values.sum()),
        "source_aware_lr_sum": float(sum(unique.values())),
    }


def witnesses():
    """Return exact, small constructed worlds without making empirical claims."""
    x = np.array([-1.0, -1.0, 1.0, 1.0])
    u = np.array([-1.0, 1.0, -1.0, 1.0])
    y, s = 3 * x + u, x + u
    denominator = float(y.std() * s.std())
    before = float(np.mean(y * s) / denominator)
    after = float(np.corrcoef(y - 3 * x, s - x)[0, 1])
    remaining = float(np.mean(u * u) / denominator)
    modeled = float(np.mean(3 * x * x) / denominator)
    a = np.array([[0.0, 1.0], [1.0, 0.0]])
    c = np.array([[1.0, 0.0]])
    h = np.array([[1.0, 1.0], [1.0, -1.0]]).T
    base = [("a", 0.8), ("b", 0.4)]
    cases = {
        "base": base,
        "all_copied_same_source": base * 4,
        "same_values_independent_sources": [
            (f"{key}_{i}", value) for i in range(4) for key, value in base
        ],
        "only_a_copied": [("a", 0.8)] * 4 + [("b", 0.4)],
        "reversed_order": list(reversed(base)),
        "independent_opposing": [("a", 0.8), ("c", -0.8)],
    }
    interactions = {}
    for name, left, right in (
        ("independent_positive_halves", [base[0]], [base[1]]),
        ("same_source_overlapping_halves", [base[0]], [base[0]]),
    ):
        full, lscore, rscore = map(evidence_scores, (left + right, left, right))
        interactions[name] = {
            key: full[key] - lscore[key] - rscore[key] for key in full
        }
    return {
        "scope": "Constructed finite worlds only; no LLM or human mechanism identified",
        "residual_correlation": {
            "before": before,
            "after_removing_x": after,
            "fixed_denominator_remaining": remaining,
            "fixed_denominator_modeled": modeled,
            "closure_error": abs(before - remaining - modeled),
        },
        "observability": {
            "current_outputs": (c @ h).ravel().tolist(),
            "next_outputs": (c @ a @ h).ravel().tolist(),
            "rank_one_observation": int(np.linalg.matrix_rank(c)),
            "rank_two_observations": int(np.linalg.matrix_rank(np.vstack([c, c @ a]))),
            "minimum_next_mse_from_current_scalar": float(np.var(c @ a @ h)),
        },
        "evidence_arithmetic": {k: evidence_scores(v) for k, v in cases.items()},
        "empty_baseline": evidence_scores([]),
        "margin_interactions": interactions,
        "normalization_competition": [
            {
                "copy_count": k,
                "content_only_mean": 1.0,
                "mean_with_fixed_null_slot": k / (1 + k),
                "additive_state_l2_normalized_readout": k / np.sqrt(1 + k**2),
                "same_source_conditional_lr": 1.0,
                "independent_lr": float(k),
            }
            for k in (1, 2, 4, 8)
        ],
        "notes": [
            "All margins use a declared zero empty-history baseline.",
            "Mean margins and LR sums define competing numerical rules, not fitted LLM scores.",
            "Source-aware LR here assumes independent unique IDs and exact duplicate IDs.",
            "Order invariance of this LR world does not imply all Bayesian updates are order invariant.",
            "Null-slot and L2 readout are different mechanisms; saturation alone does not identify either.",
        ],
    }


def main():
    """Write a fresh reproducible numerical packet, never replacing previous results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = witnesses()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / "witnesses.json"
    output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    summary = {
        "scope": result["scope"],
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "witnesses_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "neural_forwards": 0,
        "private_data_read": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
