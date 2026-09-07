#!/usr/bin/env python3
"""Finite mathematical witnesses for the seminar; no empirical personality claims."""

from __future__ import annotations

import argparse
import json
from itertools import pairwise
from pathlib import Path

import numpy as np


def cosine(a, b):
    """Euclidean direction similarity in the declared coordinate system."""
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def paired_loss(d, gradients, projection):
    """Preserve the actual same-case displacement/sensitivity pairing."""
    omitted = d - d @ projection.T
    return float(np.mean(np.einsum("ij,ij->i", gradients, omitted) ** 2))


def variance_witness():
    """A rank-one maximum-variance projection can lose all functional signal."""
    d = np.array([[10.0, 0.1], [10.0, -0.1], [-10.0, 0.1], [-10.0, -0.1]])
    g = np.tile([0.0, 10.0], (4, 1))
    pca = np.diag([1.0, 0.0])
    functional = np.diag([0.0, 1.0])
    cd = d.T @ d / len(d)
    return {
        "pca_retained_variance_fraction": float(cd[0, 0] / np.trace(cd)),
        "pca_response_mse": paired_loss(d, g, pca),
        "functional_response_mse": paired_loss(d, g, functional),
    }


def pairing_witness():
    """Identical marginal second moments need not imply identical response effects."""
    d = np.eye(2)
    same, crossed = d.copy(), d[::-1].copy()
    return {
        "gradient_second_moment_difference": float(
            np.max(abs(same.T @ same - crossed.T @ crossed))
        ),
        "displacement_second_moment_difference": 0.0,
        "paired_same_response": np.einsum("ij,ij->i", d, same).tolist(),
        "paired_crossed_response": np.einsum("ij,ij->i", d, crossed).tolist(),
    }


def coordinate_witness():
    """Dual gradient transport preserves effects and covariance-product spectrum."""
    d = np.array([[1.0, 2.0], [-2.0, 1.0], [2.0, -1.0], [-1.0, -2.0]])
    g = np.array([[2.0, 1.0], [1.0, 3.0], [-1.0, 2.0], [-3.0, 1.0]])
    t = np.diag([10.0, 0.1])
    dt, gt = d @ t.T, g @ np.linalg.inv(t)
    cd, cg = d.T @ d / len(d), g.T @ g / len(g)
    cdt, cgt = dt.T @ dt / len(dt), gt.T @ gt / len(gt)
    a, b = np.array([1.0, 1.0]), np.array([1.0, -1.0])
    return {
        "paired_effect_max_error": float(
            np.max(abs(np.sum(d * g, 1) - np.sum(dt * gt, 1)))
        ),
        "product_spectrum_max_error": float(
            np.max(
                abs(
                    np.sort(np.linalg.eigvals(cd @ cg))
                    - np.sort(np.linalg.eigvals(cdt @ cgt))
                )
            )
        ),
        "euclidean_functional_cosine_original": cosine(a, b),
        "euclidean_functional_cosine_rescaled": cosine(
            np.linalg.solve(t.T, a), np.linalg.solve(t.T, b)
        ),
    }


def path_witness():
    """Path order differs while a fixed scalar gradient remains conservative."""
    paths = (
        np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]),
        np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
    )
    areas, integrals = [], []
    for path in paths:
        areas.append(
            float(0.5 * np.sum(path[:-1, 0] * path[1:, 1] - path[:-1, 1] * path[1:, 0]))
        )
        # grad(0.5*||x||^2)=x; trapezoidal integration is exact on linear segments.
        integrals.append(
            float(np.sum((path[1:] + path[:-1]) * 0.5 * (path[1:] - path[:-1])))
        )
    return {
        "same_endpoint": bool(np.array_equal(paths[0][-1], paths[1][-1])),
        "signed_areas": areas,
        "potential_integrals": integrals,
        "closed_gradient_integral": integrals[0] - integrals[1],
    }


def conditional_expectation_witness():
    """Nested information recovers an interaction without deleting observations."""
    bits = np.array(
        [[a, b, c] for a in (-1.0, 1.0) for b in (-1.0, 1.0) for c in (-1.0, 1.0)]
    )
    a, b, c = bits.T
    y = a + 2 * a * b + 0.5 * c
    # Uniform independent signs make these exact population conditional means.
    means = [np.zeros_like(y), a, a + 2 * a * b, y]
    errors = np.array([np.mean((y - m) ** 2) for m in means])
    increments = np.array(
        [np.mean((right - left) ** 2) for left, right in pairwise(means)]
    )
    return {
        "population_squared_errors": errors.tolist(),
        "explained_increment_energies": increments.tolist(),
        "pythagorean_identity_max_error": float(
            np.max(abs(errors[:-1] - errors[1:] - increments))
        ),
        "zero_terminal_error_reason": "All target-generating variables are observed in this finite toy only.",
    }


def main():
    """Save aggregate toy quantities with an explicit non-empirical designation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    results = {
        "status": "COMPLETE_MATHEMATICAL_WITNESSES_ONLY",
        "neural_forwards": 0,
        "human_data_used": False,
        "variance": variance_witness(),
        "pairing": pairing_witness(),
        "coordinates": coordinate_witness(),
        "paths": path_witness(),
        "nested_information": conditional_expectation_witness(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(results, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
