#!/usr/bin/env python3
"""Source-fold-bound borrowing and coordinate-free input-subspace geometry."""

from __future__ import annotations

import numpy as np

from grain_research.vector_mechanisms.query_residual_input import fit_extension


def check_roles(recipient, donor, fitting_ids):
    """Reject donor leakage or a different source fold before fitting any target."""
    ids = list(fitting_ids)
    for role in (recipient, donor):
        src = list(role["source"])
        if len(src) != len(set(src)) or role["held"] in src:
            raise ValueError("Source roles overlap the held author")
        if set(ids) != set(src) or len(ids) != len(src):
            raise ValueError("Fitting authors differ from frozen source roles")
    if recipient["held"] != donor["held"]:
        raise ValueError("Donor and recipient must have the same held author")


def subspace_geometry(a, b):
    """Compute principal cosines/angles and normalized projector overlap."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.ndim != 2 or b.shape != a.shape or not len(a):
        raise ValueError("Equal nonzero rank row bases required")
    for u in (a, b):
        if not np.isfinite(u).all() or not np.allclose(
            u @ u.T, np.eye(len(u)), atol=1e-10
        ):
            raise ValueError("Orthonormal finite rows required")
    cosines = np.clip(np.linalg.svd(a @ b.T, compute_uv=False), 0, 1)
    angles = np.degrees(np.arccos(cosines))
    return {
        "rank": len(a),
        "projector_overlap": float(np.sum(cosines**2) / len(a)),
        "angle_min_degrees": float(angles.min()),
        "angle_mean_degrees": float(angles.mean()),
        "angle_max_degrees": float(angles.max()),
        **{f"cosine_{k + 1}": float(c) for k, c in enumerate(cosines)},
    }


def fit_borrowed(
    hidden,
    grain,
    gradients,
    base,
    donor,
    recipient_role,
    donor_role,
    fitting_ids,
    alpha=0.01,
):
    """Borrow only an input basis; recipient target and output geometry stay fixed."""
    check_roles(recipient_role, donor_role, fitting_ids)
    if len(hidden) != len(fitting_ids):
        raise ValueError("Source context and role count differ")
    for key in ("h_center", "h_basis"):
        if not np.array_equal(base[key], donor[key]):
            raise ValueError("Donor was built in a different source input space")
    u = donor["extra_basis"]
    subspace_geometry(u, u)
    if np.max(abs(u @ base["h_basis"].T)) > 1e-10:
        raise ValueError("Donor direction is not in the declared PCA complement")
    model, stats = fit_extension(
        hidden,
        grain,
        gradients,
        base,
        {
            "extra_basis": u.copy(),
            "extra_singular": donor["extra_singular"].copy(),
        },
        alpha=alpha,
    )
    model["source_indices"] = np.asarray(fitting_ids, dtype=int)
    model["held_index"] = np.asarray(recipient_role["held"], dtype=int)
    return model, stats
