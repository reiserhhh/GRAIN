#!/usr/bin/env python3
"""Fisher-metric accessibility of a fixed attention value readout."""

from __future__ import annotations

import numpy as np

from grain_research.vector_mechanisms.query_tilt import tilted_readout, validate_arrays


def readout_geometry(keys, values, query, beta, rtol=1e-10):
    """Solve a local Fisher-constrained query response via weighted projection.

    The returned variance is over token values under attention weights, not
    authors or labels. Numerical eigenvalue truncation defines the supported
    query directions. A zero key metric yields a zero first-order response.
    """
    k, v, q = validate_arrays(keys, values, query)
    beta = np.asarray(beta, float)
    if beta.shape != (v.shape[1],) or not np.isfinite(beta).all() or not 0 < rtol < 1:
        raise ValueError("Finite value readout and relative rank tolerance required")
    _, p = tilted_readout(k, v, q)
    scalar = v @ beta
    # Anchor before centering so constant keys remain exactly zero at finite precision.
    kd, sd = k - k[0], scalar - scalar[0]
    kc, sc = kd - p @ kd, sd - p @ sd
    fisher = (kc * p[:, None]).T @ kc
    gradient = kc.T @ (p * sc)
    eigenvalues, vectors = np.linalg.eigh(fisher)
    maximum = max(float(eigenvalues[-1]), 0.0)
    keep = eigenvalues > maximum * rtol if maximum else np.zeros_like(eigenvalues, bool)
    basis, spectrum = vectors[:, keep], eigenvalues[keep]
    solution = (
        basis @ ((basis.T @ gradient) / spectrum) if keep.any() else np.zeros_like(q)
    )
    accessible = max(float(gradient @ solution), 0.0)
    variance = max(float(p @ (sc**2)), 0.0)
    direction = solution / np.sqrt(accessible) if accessible > 0 else np.zeros_like(q)
    residual = sc - kc @ solution
    return {
        "weights": p,
        "fisher": fisher,
        "gradient": gradient,
        "natural_direction": solution,
        "unit_fisher_direction": direction,
        "accessible_variance": accessible,
        "value_variance": variance,
        "residual_variance": float(p @ (residual**2)),
        "accessibility_fraction": accessible / variance if variance > 0 else None,
        "rank": int(keep.sum()),
        "projection_identity_error": abs(variance - accessible - p @ (residual**2)),
    }
