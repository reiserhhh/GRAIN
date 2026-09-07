#!/usr/bin/env python3
"""Small constructed counterexamples for mechanism design, not LLM findings."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np


def posterior_from_likelihoods(high: float, low: float, prior: float) -> float:
    """Return a binary posterior for a declared positive-probability toy event."""
    if not (0 < high <= 1 and 0 < low <= 1 and 0 < prior < 1):
        raise ValueError("Likelihoods must be in (0, 1], prior in (0, 1)")
    return high * prior / (high * prior + low * (1 - prior))


def factorial_coefficient(values: np.ndarray, signs: np.ndarray) -> float:
    """Extract a contrast on a complete balanced sign design, not a causal effect."""
    return float(np.mean(values * np.prod(signs, axis=1)))


def order_cycle(score, a: float, b: float, neutral: float = 0.0) -> float:
    """Return a three-object order cycle on a declared output scale."""

    def comm(x, y):
        return score(x, y) - score(y, x)

    return float(comm(a, b) - comm(a, neutral) + comm(b, neutral))


def seminar_challenges() -> dict:
    """Compute counterexamples to three interpretations proposed in round one."""
    cells = np.array(list(product((-1.0, 1.0), repeat=2)))
    q, code = cells.T
    table = 0.15 + q + 0.3 * code + 0.2 * q * code
    contexts = np.array([-1.0, 0.0, 1.0])
    additive = lambda a, b: a + 2 * b
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    gradients = np.array([[1.0, 2.0, 0.0], [1.0, 2.0, 0.0], [-2.0, 1.0, 3.0]])
    probes = np.array(list(product((-1.0, 1.0), repeat=3))) / np.sqrt(3)
    responses = probes @ gradients.T
    response_kernel = responses.T @ responses / len(probes)
    exact_kernel = gradients @ gradients.T / 3
    patch_model = lambda edge, other: edge * other
    base, edge_only, other_only, joint = (
        patch_model(0, 0),
        patch_model(1, 0),
        patch_model(0, 1),
        patch_model(1, 1),
    )
    gain_model = lambda evidence, context: (1 + context) * evidence
    eps = 0.01
    return {
        "factorial_mean_not_unique_semantics": {
            "four_cell_mean": float(table.mean()),
            "semantic_world_component": 0.15,
            "nuisance_world_semantic_component": 0.0,
            "nuisance_world_invariant_offset": 0.15,
            "observable_max_difference": 0.0,
            "note": "Both worlds define the identical response table; invariance cannot name the component.",
        },
        "order_cycle_not_unique_updating": {
            "linear_margin": order_cycle(additive, 1, 2),
            "sigmoid_probability": order_cycle(
                lambda a, b: sigmoid(additive(a, b)), 1, 2
            ),
            "static_cubic_margin": order_cycle(lambda a, b: additive(a, b) ** 3, 1, 2),
            "note": "The cubic is a static nonlinear readout, with no learned evidence-update rule.",
        },
        "negative_context_slope_without_reference_comparison": {
            "contexts": contexts.tolist(),
            "outputs": (1 / (2 + np.exp(contexts))).tolist(),
            "derivative_at_zero": -1 / 9,
            "cancelling_assimilation_plus_contrast": (contexts - contexts).tolist(),
            "note": "Target value is 1; other values are 0. Context only changes non-target mass.",
        },
        "literal_budget_arithmetic": {
            "two_sites_four_directions_three_doses_two_signs": 96 * 2 * 4 * 3 * 2,
            "plus_baselines_and_secondary_and_direction_examples": 96 * 2 * 4 * 3 * 2
            + 96
            + 700
            + 24,
            "note": "Logical evaluations are not necessarily forward calls; multi-token candidates add cost.",
        },
        "coded_probe_functional_kernel": {
            "directions": len(probes),
            "dimension": 3,
            "kernel": response_kernel.tolist(),
            "identity_max_error": float(np.max(np.abs(response_kernel - exact_kernel))),
            "same_function_signature_max_error": float(
                np.max(np.abs(responses[:, 0] - responses[:, 1]))
            ),
            "note": "Exact linear toy with every Rademacher sign; finite samples and nonlinear models need error checks.",
        },
        "patch_subtraction_is_not_mediation": {
            "function": "output = edge_state * other_state",
            "base": base,
            "edge_only": edge_only,
            "other_only": other_only,
            "joint": joint,
            "joint_minus_edge": joint - edge_only,
            "interaction": joint - edge_only - other_only + base,
            "note": "Joint-minus-edge cannot be assigned to the other path without interaction assumptions.",
        },
        "context_can_change_sensitivity_at_equal_baseline": {
            "function": "margin = (1 + context) * evidence",
            "contexts": [0, 1],
            "baseline_at_zero_evidence": [gain_model(0, c) for c in (0, 1)],
            "evidence_sensitivities": [
                (gain_model(eps, c) - gain_model(-eps, c)) / (2 * eps) for c in (0, 1)
            ],
            "note": "Matching endpoints cannot eliminate contextual gain differences in intervention responses.",
        },
    }


def witnesses() -> dict:
    """Expose five finite possibilities and their limitations without private data."""
    cells = np.array(list(product((-1.0, 1.0), repeat=3)))
    evidence, role, query = cells.T
    # An answer offset masks a three-way binding interaction in all main effects.
    margin = 4 + 3 * query + 2 * evidence * role * query
    angles = 2 * np.pi * np.arange(8) / 8
    vectors = np.column_stack([np.cos(angles), np.sin(angles)])
    global_direction = vectors.mean(axis=0)
    pairing = np.sum(vectors * vectors, axis=1)
    rotated = np.roll(vectors, 2, axis=0)
    rho = 0.5
    first_positive = rho * 1.0 - 1.0
    first_negative = rho * -1.0 + 1.0
    synergy_cells = np.array(list(product((-1.0, 1.0), repeat=2)))
    x, y = synergy_cells.T
    combined = x * y
    voluntary = posterior_from_likelihoods(0.8, 0.2, 0.5)
    compulsory = posterior_from_likelihoods(0.9, 0.9, 0.5)
    rare_prior = posterior_from_likelihoods(0.8, 0.2, 0.1)
    return {
        "scope": "Exact constructed worlds, not empirical GRAIN/LLM/personality results",
        "query_role_interaction": {
            "cells": len(cells),
            "main_evidence": factorial_coefficient(margin, cells[:, [0]]),
            "main_role": factorial_coefficient(margin, cells[:, [1]]),
            "main_query": factorial_coefficient(margin, cells[:, [2]]),
            "triple_interaction": factorial_coefficient(margin, cells),
            "note": "Complete factorial contrasts recover specified algebra, not an internal circuit.",
        },
        "paired_action_without_global_axis": {
            "mean_vector_norm": float(np.linalg.norm(global_direction)),
            "mean_local_action": float(pairing.mean()),
            "mean_global_action": float(np.mean(vectors @ global_direction)),
            "quarter_turn_mismatched_action": float(
                np.sum(vectors * rotated, axis=1).mean()
            ),
            "note": "Rotations and pairings are given, not learned from held-out authors.",
        },
        "order_without_reasoning": {
            "rule": "state_next = 0.5 * state + event; state_initial = 0",
            "positive_then_negative": first_positive,
            "negative_then_positive": first_negative,
            "order_difference": first_positive - first_negative,
            "note": "A linear recency rule already gives noncommuting affine updates.",
        },
        "synergy_without_marginal_signal": {
            "mean_target": float(combined.mean()),
            "covariance_with_x": float(np.mean(combined * x)),
            "covariance_with_y": float(np.mean(combined * y)),
            "product_readout_mse": float(np.mean((x * y - combined) ** 2)),
            "additive_least_squares_mse": float(
                np.mean(
                    (
                        np.column_stack([np.ones(4), x, y])
                        @ np.linalg.lstsq(
                            np.column_stack([np.ones(4), x, y]), combined, rcond=None
                        )[0]
                        - combined
                    )
                    ** 2
                )
            ),
            "note": "The product is stipulated; this does not show real text needs this kernel.",
        },
        "condition_and_reference": {
            "voluntary_same_action_posterior": voluntary,
            "compulsory_same_action_posterior": compulsory,
            "voluntary_log_likelihood_ratio": float(np.log(0.8 / 0.2)),
            "compulsory_log_likelihood_ratio": 0.0,
            "voluntary_rare_group_prior_posterior": rare_prior,
            "note": "Group prior alone can shift judgments with an unchanged evidence likelihood ratio.",
        },
        "round_one_challenges": seminar_challenges(),
    }


def main() -> None:
    """Save a new reproducible toy packet without changing any experiment inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    data = witnesses()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).read_bytes()
    (args.output_dir / "source_snapshot.py").write_bytes(source)
    output = args.output_dir / "witnesses.json"
    output.write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    receipt = {
        "scope": data["scope"],
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "neural_forwards": 0,
        "private_data_read": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
