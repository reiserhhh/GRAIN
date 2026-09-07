#!/usr/bin/env python3
"""Finite countermodels for the second seminar, without human data or model calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def softmax(z):
    """Compute probabilities without overflow in the finite witness."""
    v = np.exp(z - np.max(z))
    return v / v.sum()


def fisher_witness():
    """Compare local output-KL cost at identical linear target displacement."""
    a = np.array([[1.0, 0.0], [0.0, 2.0], [-1.0, -1.0]])
    h = np.array([0.2, -0.3])
    p = softmax(a @ h)
    metric = a.T @ (np.diag(p) - np.outer(p, p)) @ a
    target, rho = np.array([0.8, 1.0]), 1e-3
    inverse_target = np.linalg.solve(metric, target)
    optimal = rho * inverse_target / (target @ inverse_target)
    euclidean = rho * target / (target @ target)
    costs = [float(0.5 * d @ metric @ d) for d in (optimal, euclidean)]
    actual = float(np.sum(p * np.log(p / softmax(a @ (h + optimal)))))
    return {
        "target_changes": [float(target @ d) for d in (optimal, euclidean)],
        "quadratic_cost_optimal_euclidean": costs,
        "euclidean_to_optimal_cost_ratio": costs[1] / costs[0],
        "actual_kl": actual,
        "relative_quadratic_error": abs(actual - costs[0]) / costs[0],
    }


def query_null_witness():
    """A direction invisible to two readouts can be visible to a third."""
    j = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    d = np.array([0.0, 0.0, 1.0])
    expanded = np.vstack([j, [0.0, 0.0, 1.0]])
    return {
        "two_query_response": (j @ d).tolist(),
        "three_query_response": (expanded @ d).tolist(),
        "direction_norm": float(np.linalg.norm(d)),
    }


def symmetry_witness():
    """Averaging a role-changing action can erase the author-related observable."""
    x, swapped = np.array([1.0, -1.0]), np.array([-1.0, 1.0])
    role = np.array([1.0, 0.0])
    return {
        "fixed_role_scores": [float(role @ x), float(role @ swapped)],
        "role_swap_group_average": float(role @ (x + swapped) / 2),
        "coordinate_and_dual_role_transformed_score": float(role[::-1] @ swapped),
    }


def entropy(values):
    """Exact base-two entropy of a finite uniform empirical probability table."""
    _, counts = np.unique(np.asarray(values), axis=0, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p)))


def synergy_witness():
    """Joint information and edit interaction need not require noncommuting edits."""
    x = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    y = x[:, 0] * x[:, 1]
    marginal_information = [
        entropy(x[:, k]) + entropy(y) - entropy(np.column_stack([x[:, k], y]))
        for k in (0, 1)
    ]
    joint_information = entropy(x) + entropy(y) - entropy(np.column_stack([x, y]))
    origin, da, db = np.zeros(2), np.array([1.0, 0.0]), np.array([0.0, 1.0])
    interaction = np.prod(origin + da + db) - np.prod(origin + da)
    interaction -= np.prod(origin + db) - np.prod(origin)
    return {
        "marginal_correlations": [float(np.corrcoef(x[:, k], y)[0, 1]) for k in (0, 1)],
        "marginal_information_bits": marginal_information,
        "joint_information_bits": joint_information,
        "mixed_edit_difference": float(interaction),
        "input_edit_commutator_norm": float(
            np.linalg.norm((origin + da) + db - ((origin + db) + da))
        ),
    }


def distribution_witness():
    """Different comment distributions can have equal ordinary mean embeddings."""
    p, q = np.array([-1.0, 1.0]), np.array([0.0, 0.0])

    def kernel(x, y):
        return np.exp(-((x[:, None] - y[None, :]) ** 2) / 2)

    return {
        "ordinary_means": [float(p.mean()), float(q.mean())],
        "gaussian_kernel_mmd_squared": float(
            kernel(p, p).mean() + kernel(q, q).mean() - 2 * kernel(p, q).mean()
        ),
        "kernel_bandwidth": 1.0,
    }


def observational_equivalence_witness():
    """Equal observational scores need not imply equal responses to new edits."""
    t = np.array([-2.0, -1.0, 1.0, 2.0])
    points = np.column_stack([t, t])
    base, changed = np.array([1.0, 1.0]), np.array([2.0, 1.0])
    return {
        "observational_mse": float(np.mean((points[:, 0] - points[:, 1]) ** 2)),
        "same_edit_model_responses": (changed - base).tolist(),
    }


def inverse_witness():
    """A fixed class-conditional text likelihood does not fix the inverse posterior."""
    likelihood_true, likelihood_false = 0.8, 0.2
    priors = [0.5, 0.1]
    posterior = [
        likelihood_true
        * prior
        / (likelihood_true * prior + likelihood_false * (1 - prior))
        for prior in priors
    ]
    return {"priors": priors, "posteriors_for_same_text_event": posterior}


def shared_aggregator_witness():
    """One shared vector sum with different readouts can have rank-two attribution."""
    comment_features = np.eye(2)
    readouts = np.eye(2)
    attribution = comment_features @ readouts.T
    return {
        "query_independent_aggregate": comment_features.sum(axis=0).tolist(),
        "attribution_matrix": attribution.tolist(),
        "attribution_rank": int(np.linalg.matrix_rank(attribution)),
    }


def repeated_representation_witness():
    """Repeated irreducible components admit non-scalar intertwining maps."""
    action = np.diag([-1.0, -1.0, 1.0])
    transport = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 3.0]])
    return {
        "commutator_norm": float(
            np.linalg.norm(action @ transport - transport @ action)
        ),
        "repeated_component_off_diagonal": float(transport[0, 1]),
        "transport_determinant": float(np.linalg.det(transport)),
    }


def finite_nonlinearity_witness():
    """A zero local gradient does not imply zero response to a finite edit."""
    base, edit = np.zeros(2), np.ones(2)
    gradient = base[::-1]
    return {
        "local_gradient": gradient.tolist(),
        "linear_prediction": float(gradient @ edit),
        "finite_response": float(np.prod(base + edit) - np.prod(base)),
    }


def main():
    """Write only synthetic aggregate quantities with explicit provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "status": "MATHEMATICAL_COUNTERMODELS_ONLY",
        "human_data_used": False,
        "neural_forwards": 0,
        "fisher": fisher_witness(),
        "query_null": query_null_witness(),
        "symmetry": symmetry_witness(),
        "synergy": synergy_witness(),
        "distribution": distribution_witness(),
        "observational_equivalence": observational_equivalence_witness(),
        "inverse": inverse_witness(),
        "shared_aggregator": shared_aggregator_witness(),
        "repeated_representation": repeated_representation_witness(),
        "finite_nonlinearity": finite_nonlinearity_witness(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
