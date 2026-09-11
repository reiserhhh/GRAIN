#!/usr/bin/env python3
"""Matched RKHS reader fits with source-only functional observation losses."""

import numpy as np

from . import material_reader_field as base

VARIANTS = ("vector_control", "observable_teaching")
LOSSES = ("vector", "scalar", "action")


def coordinates(model):
    """Retain numerical source rank while preserving the old RKHS penalty."""
    x, v, kinds, y = [model[k] for k in ("x", "v", "kinds", "target")]
    k = base.gram(x, v, kinds, x, v, kinds)
    eig, vec = np.linalg.eigh(k)
    tol = max(k.shape) * np.finfo(float).eps * max(float(eig[-1]), 1.0)
    if eig[0] < -tol * 10:
        raise ValueError("Source observation kernel is not PSD")
    keep = eig > tol
    if not keep.any():
        raise ValueError("Empty source observation range")
    e, vv = eig[keep], vec[:, keep]
    a, transform = vv * np.sqrt(e), vv / np.sqrt(e)
    _, s, vt = np.linalg.svd(y, full_matrices=False)
    rt = max(y.shape) * np.finfo(float).eps * s[0]
    u = vt[s > rt].T
    if not u.shape[1]:
        raise ValueError("Empty source output range")
    return {
        "a": a,
        "transform": transform,
        "output_basis": u,
        "kernel_reconstruction_relative": np.linalg.norm(k - a @ a.T)
        / np.linalg.norm(k),
        "target_outside_relative": np.linalg.norm(y - y @ u @ u.T) / np.linalg.norm(y),
        "kernel_rank_tolerance": tol,
        "output_rank_tolerance": rt,
    }


def functional_rows(a, u, target, rulers, groups):
    """Build source-normalized scalar/action design rows for row-major Theta."""
    r = np.asarray(rulers, float)
    if r.ndim != 3 or r.shape[0] != len(a) or r.shape[2] != target.shape[1]:
        raise ValueError("One fixed set of output rulers per observation required")
    if not np.isfinite(r).all():
        raise ValueError("Finite source rulers required")
    projection = r @ u
    y = np.einsum("ih,ifh->if", target, r)
    scales, zero = np.zeros(len(a)), []
    for gi in range(3):
        use = groups == gi
        energy = float(np.sum(y[use] ** 2))
        zero.append(energy == 0)
        if energy:
            scales[use] = 1 / np.sqrt(energy)
    design = np.einsum("ia,ifb->ifab", a, projection)
    design *= scales[:, None, None, None]
    return (
        design.reshape(-1, a.shape[1] * u.shape[1]),
        (y * scales[:, None]).ravel(),
        np.asarray(zero),
    )


def losses(error, target, rulers, groups):
    """Report all three source losses, retaining zero-energy flags separately."""
    values, zero = np.zeros((3, 3)), np.zeros((3, 3), bool)
    for li in range(3):
        er, yr = error, target
        if li:
            r = rulers[:, :1] if li == 1 else rulers[:, 1:]
            er = np.einsum("ih,ifh->if", error, r)
            yr = np.einsum("ih,ifh->if", target, r)
        for gi in range(3):
            use = groups == gi
            energy = float(np.sum(yr[use] ** 2))
            zero[li, gi] = energy == 0
            values[li, gi] = np.sum(er[use] ** 2) / energy if energy else 0.0
    return values, zero


def fit(model, rulers, include_observables):
    """Solve one fixed SPD coefficient system, without recipient observations."""
    c = coordinates(model)
    a, u = c["a"], c["output_basis"]
    y, w, groups = [model[k] for k in ("target", "weights", "groups")]
    lam = float(model["regularizer"])
    if lam <= 0 or not np.isfinite(lam):
        raise ValueError("Positive frozen regularizer required")
    aw = a * w[:, None]
    normal = np.kron(aw.T @ aw, np.eye(u.shape[1]))
    normal += lam * np.eye(len(normal))
    rhs = (aw.T @ (y * w[:, None]) @ u).ravel()
    omitted = []
    for r in (rulers[:, :1], rulers[:, 1:]):
        z, b, zero = functional_rows(a, u, y, r, groups)
        omitted.append(zero)
        if include_observables:
            normal += z.T @ z
            rhs += z.T @ b
    theta = np.linalg.solve(normal, rhs).reshape(a.shape[1], u.shape[1])
    residual = np.linalg.norm(normal @ theta.ravel() - rhs) / max(
        np.linalg.norm(rhs), np.finfo(float).tiny
    )
    if residual > 1e-9 or not np.isfinite(theta).all():
        raise ValueError("Source coefficient solve did not converge")
    beta = c["transform"] @ theta @ u.T
    fitted = a @ theta @ u.T
    ls, zero = losses(fitted - y, y, rulers, groups)
    prior_difference = np.linalg.norm(beta - model["beta"]) / np.linalg.norm(
        model["beta"]
    )
    if not include_observables and prior_difference > 1e-7:
        raise ValueError("Matched vector control differs from NEXT82")
    return (
        model
        | c
        | {
            "beta": beta,
            "theta": theta,
            "source_rulers": rulers,
            "source_losses": ls,
            "source_zero_energy": zero,
            "omitted_functional_groups": np.asarray(omitted),
            "coefficient_count": np.asarray(theta.size),
            "normal_residual_relative": np.asarray(residual),
            "prior_beta_relative": np.asarray(prior_difference),
            "regularization_penalty": np.asarray(lam * np.sum(theta**2)),
        }
    )


def split_error(predicted, target, origin, basis):
    """Split error for a source-range-constrained field; reject off-range fits."""
    np.testing.assert_allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=1e-10)
    change = predicted - origin
    np.testing.assert_allclose(change, change @ basis @ basis.T, atol=1e-9)
    e = predicted - target
    inside = e @ basis @ basis.T
    t = target - origin
    outside = -(t - t @ basis @ basis.T)
    np.testing.assert_allclose(e, inside + outside, atol=1e-9)
    return inside, outside


def error_account(inside, outside):
    """Keep signed cross-energy after projection; do not call each part a share."""
    a, b = np.asarray(inside, float), np.asarray(outside, float)
    ea, eb = float(np.sum(a * a)), float(np.sum(b * b))
    cross, total = float(2 * np.sum(a * b)), float(np.sum((a + b) ** 2))
    return {
        "inside_error2": ea,
        "outside_error2": eb,
        "signed_cross": cross,
        "total_error2": total,
        "closure_error": abs(total - ea - eb - cross),
    }
