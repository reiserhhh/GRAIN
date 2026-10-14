#!/usr/bin/env python3
"""Finite material-common event functions and their joint author moments."""

from __future__ import annotations

import itertools

import numpy as np

EPS_FACTOR = 64.0


def finite_matrix(value):
    """Validate a complete nonempty finite observation matrix."""
    a = np.asarray(value, float)
    if a.ndim != 2 or min(a.shape) < 1 or not np.isfinite(a).all():
        raise ValueError("Nonempty finite two-dimensional observation required")
    return a


def equilibrate(a):
    """Rescale columns without changing their exact event-function span."""
    a = finite_matrix(a)
    scales = np.linalg.norm(a, axis=0)
    scales[scales == 0] = 1.0
    return a / scales, scales


def column_space(a):
    """Return full left vectors and a fixed machine-relative numerical rank."""
    a, _ = equilibrate(a)
    u, s, _ = np.linalg.svd(a, full_matrices=a.shape[0] > a.shape[1])
    tolerance = EPS_FACTOR * np.finfo(float).eps * max(a.shape) * s[0]
    rank = int(np.sum(s > tolerance))
    return u, rank, s


def walsh_anchors():
    """Provide seven ordered nonconstant Boolean characters, not named traits."""
    cells = np.asarray(list(itertools.product((-1, 1), repeat=3)), float)
    return np.column_stack(
        [
            np.prod(cells[:, [j for j in range(3) if mask & (1 << j)]], axis=1)
            / np.sqrt(8)
            for mask in range(1, 8)
        ]
    )


def canonical_basis(projector, dimension):
    """Orient a subspace with fixed anchors; the projector remains the invariant."""
    n = projector.shape[0]
    anchors = walsh_anchors() if n == 8 else np.eye(n)
    columns = []
    tolerance = EPS_FACTOR * np.finfo(float).eps * max(n, 1) * 8
    for anchor in anchors.T:
        v = projector @ anchor
        for _ in range(2):
            for q in columns:
                v -= q * (q @ v)
        norm = np.linalg.norm(v)
        if norm > tolerance:
            q = v / norm
            if q @ anchor < 0:
                q = -q
            columns.append(q)
        if len(columns) == dimension:
            break
    if len(columns) != dimension:
        raise ValueError("Fixed anchors failed to span the common observable space")
    return np.column_stack(columns) if dimension else np.empty((n, 0))


def quadratic_functions(z):
    """Keep event products, not products of separately aggregated author means."""
    z = np.asarray(z, float)
    if z.ndim != 2 or z.shape[0] < 1 or not np.isfinite(z).all():
        raise ValueError("Finite aligned common event functions required")
    pairs = tuple(itertools.combinations_with_replacement(range(z.shape[1]), 2))
    products = (
        np.column_stack([z[:, i] * z[:, j] for i, j in pairs])
        if pairs
        else np.empty((len(z), 0))
    )
    return np.column_stack((np.ones(len(z)), z, products)), pairs


def chart(f0, f1, *, joint=True):
    """Build the complete material-common contrast chart and fixed degree-two lift."""
    f0, f1 = finite_matrix(f0), finite_matrix(f1)
    if f0.shape[0] != f1.shape[0]:
        raise ValueError("Declared event support must be aligned across materials")
    n = len(f0)
    designs = [np.column_stack((np.ones(n), f)) for f in (f0, f1)]
    spaces = [column_space(a) for a in designs]
    constraints = np.vstack(
        [*(u[:, rank:].T for u, rank, _ in spaces), np.ones((1, n)) / np.sqrt(n)]
    )
    _, s, vh = np.linalg.svd(constraints, full_matrices=True)
    tolerance = EPS_FACTOR * np.finfo(float).eps * max(constraints.shape) * s[0]
    rank = int(np.sum(s > tolerance))
    q = vh[rank:].T
    projector = q @ q.T
    z = canonical_basis(projector, n - rank)
    calibration, errors = [], []
    for a in designs:
        balanced, scales = equilibrate(a)
        coefficient = (
            np.linalg.lstsq(
                balanced, z, rcond=EPS_FACTOR * np.finfo(float).eps * max(a.shape)
            )[0]
            / scales[:, None]
        )
        calibration.append(coefficient)
        errors.append(float(np.max(abs(a @ coefficient - z))) if z.size else 0.0)
    h, products = (
        quadratic_functions(z) if joint else (np.column_stack((np.ones(n), z)), ())
    )
    _, joint_rank, joint_s = column_space(h)
    inverse = np.linalg.pinv(h, rcond=EPS_FACTOR * np.finfo(float).eps * max(h.shape))
    return {
        "basis": z,
        "projector": projector,
        "calibration0": calibration[0],
        "calibration1": calibration[1],
        "material_ranks": np.asarray([rank for _, rank, _ in spaces]),
        "constraint_singular_values": s,
        "joint_singular_values": joint_s,
        "quadratic": h,
        "inverse": inverse,
        "products": np.asarray(products, int).reshape(-1, 2),
        "contrast_dimension": np.asarray(z.shape[1]),
        "joint_rank": np.asarray(joint_rank),
        "degree": np.asarray(2 if joint else 1),
        "calibration_errors": np.asarray(errors),
        "minimum_event_separation": np.asarray(
            min(np.linalg.norm(z[i] - z[j]) for i in range(n) for j in range(i))
            if n > 1
            else 0.0
        ),
    }


def profile_moments(probabilities, common_chart):
    """Observe valid profiles, retaining common means and within-event second moments."""
    p = finite_matrix(probabilities)
    z, h = common_chart["basis"], common_chart["quadratic"]
    if p.shape[1] != len(z) or np.any(p < 0) or np.max(abs(p.sum(1) - 1)) > 1e-12:
        raise ValueError(
            "Normalized probabilities on the declared event support required"
        )
    means, moments = p @ z, p @ h
    pair_indices = common_chart["products"]
    separate_products = (
        np.column_stack([means[:, i] * means[:, j] for i, j in pair_indices])
        if len(pair_indices)
        else np.empty((len(p), 0))
    )
    return {
        "means": means,
        "joint_moments": moments,
        "product_of_means": separate_products,
        "reconstructed": moments @ common_chart["inverse"],
    }
