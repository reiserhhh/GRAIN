#!/usr/bin/env python3
"""F01-01 exact counterexamples for the two object classes without an executed
F02-F16 counterexample: reflective latent vs network interaction, and reflective
latent vs formative index. Pure linear algebra; no data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    """Write object_counterexamples.json."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    # 1. Reflective one-factor vs recursive network (same covariance).
    lam = np.array([0.8, 0.7, 0.6, 0.5])
    psi = 1 - lam**2
    sigma = np.outer(lam, lam) + np.diag(psi)
    chol = np.linalg.cholesky(sigma)
    d = np.diag(np.diag(chol))
    lower = chol @ np.linalg.inv(d)  # unit lower triangular: X = B X + eps
    b = np.eye(4) - np.linalg.inv(lower)
    omega = d @ d
    sigma_net = np.linalg.inv(np.eye(4) - b) @ omega @ np.linalg.inv(np.eye(4) - b).T
    # intervention do(X1 = 1): reflective -> indicators 2..4 unchanged in mean;
    # network -> mean shift of (I-B_sub)^{-1} B[:,0] for downstream nodes.
    b_mut = b.copy()
    b_mut[0, :] = 0
    shift_net = np.linalg.solve(np.eye(4) - b_mut, np.eye(4)[:, 0] * 1.0)
    reflective_shift = np.array([1.0, 0.0, 0.0, 0.0])
    # 2. Reflective one-factor vs formative index: tetrad constraints.
    lam2 = np.array([[0.8, 0.0], [0.7, 0.0], [0.0, 0.6], [0.0, 0.5]])
    phi = np.array([[1.0, 0.3], [0.3, 1.0]])
    sigma2 = lam2 @ phi @ lam2.T + np.diag(1 - np.sum((lam2 @ phi) * lam2, axis=1))

    def tetrads(s):
        out = {}
        for i, j, k, l in [(0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2)]:
            out[f"t{i}{j}{k}{l}"] = float(s[i, j] * s[k, l] - s[i, k] * s[j, l])
        return out

    weights = np.array([0.25, 0.25, 0.25, 0.25])
    report = {
        "reflective_vs_network": {
            "sigma_max_abs_diff": float(np.abs(sigma - sigma_net).max()),
            "network_B_nonzero_edges": int((np.abs(b) > 1e-12).sum()),
            "intervention_do_X1_mean_shift_reflective": reflective_shift.tolist(),
            "intervention_do_X1_mean_shift_network": shift_net.round(12).tolist(),
            "breaking_observation": "intervene on one indicator: a reflective latent does not propagate to other indicators; a network does",
            "scope": "linear Gaussian; recursive network built from the Cholesky factor of the one-factor covariance",
        },
        "reflective_vs_formative": {
            "one_factor_tetrads": tetrads(sigma),
            "two_factor_tetrads": tetrads(sigma2),
            "formative_index_weights": weights.tolist(),
            "formative_index_variance_two_factor": float(weights @ sigma2 @ weights),
            "breaking_observation": "vanishing tetrads are implied by a single reflective latent; a non-vanishing tetrad rejects it while a formative index (a declared composite) stays defined but carries no causal-latent claim",
        },
    }
    (args.output / "object_counterexamples.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
