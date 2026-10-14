"""Endpoint-exchange observation geometry and balanced finite-group actions."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from grain_research.construction.math_midpoint_polarization import gram_blocks


def exchange(x, context, kappa):
    """Exchange endpoints in their original, separately normalized units."""
    x, c = np.asarray(x, float), np.asarray(context, float)
    if x.ndim != 2 or x.shape != c.shape or not np.isfinite(kappa):
        raise ValueError("matching endpoint coordinates and finite kappa required")
    if not np.isfinite(x).all() or not np.isfinite(c).all():
        raise ValueError("finite endpoints required")
    return -x, c + kappa * x


def project_actions(forward, reverse):
    """Return even/odd scalar actions without claiming a reverse observation."""
    f, r = np.asarray(forward, float), np.asarray(reverse, float)
    if f.shape != r.shape or not np.isfinite(f).all() or not np.isfinite(r).all():
        raise ValueError("paired finite actions required")
    return (f + r) / 2, (f - r) / 2


def orbit_grams(x, context, kappa):
    """Construct receiver and exchanged symmetric dyad Grams, not new data."""
    b = gram_blocks(x, context, kappa)
    u, v, uv = b["U"], b["V"], b["UV"]
    cross = -u - 2 * uv
    reverse = u + 2 * uv + 2 * uv.T + 4 * v
    return {
        **b,
        "U_reverse": reverse,
        "U_U_reverse": cross,
        "stack_U_V": np.block([[u, uv], [uv.T, v]]),
        "orbit_U_U_reverse": np.block([[u, cross], [cross.T, reverse]]),
    }


def balanced_cross(even, odd, weights):
    """Compare original and balanced-orbit cross moments in function space."""
    e, o, w = (np.asarray(v, float) for v in (even, odd, weights))
    if e.shape != o.shape or e.shape != w.shape or e.ndim != 1:
        raise ValueError("matched scalar actions and probability weights required")
    if np.any(w <= 0) or not np.isclose(w.sum(), 1, atol=1e-12, rtol=0):
        raise ValueError("positive probability measure required")
    if not all(np.isfinite(v).all() for v in (e, o, w)):
        raise ValueError("finite action measure required")
    orbit_e, orbit_o, orbit_w = np.tile(e, 2), np.r_[o, -o], np.tile(w / 2, 2)
    return float(w @ (e * o)), float(orbit_w @ (orbit_e * orbit_o))


def lifted_metric():
    """Average the Euclidean metric for the abstract (U,V) pair representation."""
    rho = np.array([[-1.0, -2.0], [0.0, 1.0]])
    return rho, (np.eye(2) + rho.T @ rho) / 2


def inverse_inventory(descriptors):
    """Find independently addressed inverse edges in identical full contexts."""
    index = defaultdict(list)
    for row in descriptors:
        index[
            row["context_sha256"], row["reference_sha256"], row["target_sha256"]
        ].append(row["row"])
    result = []
    for row in descriptors:
        noop = row["reference_sha256"] == row["target_sha256"]
        matches = index.get(
            (row["context_sha256"], row["target_sha256"], row["reference_sha256"]), []
        )
        result.append(
            {
                "row": row["row"],
                "exact_input_noop": noop,
                "inverse_rows": [i for i in matches if i != row["row"]],
                "nontrivial_observed_inverse": bool(matches) and not noop,
            }
        )
    return result
