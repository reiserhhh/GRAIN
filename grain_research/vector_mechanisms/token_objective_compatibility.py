#!/usr/bin/env python3
"""Source-only column-space, value-nullspace and regularization accounting."""

from __future__ import annotations

import numpy as np

RCOND = 1e-10


def reconstruct_design(bodies, lasts, gradients, scores, model):
    """Independently rebuild normalized scalar/derivative constraints from arrays."""
    u, mu, scale, centers = [
        np.asarray(model[n], float) for n in ("basis", "center", "scale", "landmarks")
    ]
    arm = str(model["arm"])
    if arm not in ("mean_then_kernel", "kernel_then_mean"):
        raise ValueError("Frozen pooling order required")
    n, rank = len(bodies), u.shape[1]
    y = np.asarray(scores, float)
    if not n or any(len(x) != n for x in (lasts, gradients, y)):
        raise ValueError("Aligned source authors required")
    projected = [np.asarray(h, float) @ u for h in bodies]
    expected_mu = sum(x.mean(0) for x in projected) / n
    expected_sd = np.sqrt(
        sum(np.mean((x - expected_mu) ** 2, axis=0) for x in projected) / n
    )
    expected_sd[expected_sd < 1e-8] = 1
    ks = [
        len(h) * (np.asarray(g, float) @ u) * scale for h, g in zip(bodies, gradients)
    ]
    sl = max(float(y.std()), 1e-6)
    sk = max(float(np.sqrt(sum(np.mean(k * k) for k in ks) / n)), 1e-6)
    check = max(
        float(np.max(abs(mu - expected_mu))),
        float(np.max(abs(scale - expected_sd))),
        abs(sl - float(model["score_scale"])),
        abs(sk - float(model["density_scale"])),
    )
    if check > 1e-10:
        raise ValueError("Source normalization changed")
    values, density, targets = [], [], []
    width = 1 + 2 * rank + len(centers)
    bandwidth = float(model["bandwidth"])
    for h, last, hu, k in zip(bodies, lasts, projected, ks, strict=True):
        count = len(h)
        if k.shape != (count, rank):
            raise ValueError("Native gradient/token alignment differs")
        z = (hu - mu) / scale
        lastz = (np.asarray(last, float) @ u - mu) / scale
        x = 2 * (np.arange(count) + 0.5) / count - 1
        p = np.column_stack((x, (3 * x * x - 1) / 2, (5 * x * x * x - 3 * x) / 2))
        desc = np.column_stack((z, p))
        if arm == "mean_then_kernel":
            desc = desc.mean(0, keepdims=True)
        diff = desc[:, None] - centers[None]
        kernels = np.exp(-np.einsum("tmr,tmr->tm", diff, diff) / (2 * bandwidth**2))
        phi = np.r_[1, z.mean(0), lastz, kernels.mean(0)]
        d = np.zeros((count, rank, width))
        for r in range(rank):
            d[:, r, 1 + r] = 1
            d[:, r, 1 + 2 * rank :] = -kernels * diff[:, :, r] / bandwidth**2
        values.append(phi / (np.sqrt(n) * sl))
        density.append(d.reshape(-1, width) / (np.sqrt(n * count * rank) * sk))
        targets.append(k.ravel() / (np.sqrt(n * count * rank) * sk))
    v, d, ell, k = (
        np.asarray(values),
        np.concatenate(density),
        y / (np.sqrt(n) * sl),
        np.concatenate(targets),
    )
    gram = v.T @ v + d.T @ d
    rhs = v.T @ ell + d.T @ k
    errors = {
        "normalization": check,
        "gram": float(np.max(abs(gram - model["gram"]))),
        "rhs": float(np.max(abs(rhs - model["rhs"]))),
        "unit_density_target": abs(float(k @ k) - 1),
    }
    if max(errors.values()) > 1e-8:
        raise ValueError("Independent source design differs: " + str(errors))
    anchor = np.r_[0, np.arange(1 + rank, 1 + 2 * rank)]
    return v, d, ell, k, anchor, errors


def svd_operator(a):
    """Return a fixed-cutoff thin pseudoinverse representation, not a fitted model."""
    a = np.asarray(a, float)
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError("Finite matrix required")
    if not min(a.shape):
        return (
            np.empty((a.shape[0], 0)),
            np.empty(0),
            np.empty((0, a.shape[1])),
            np.empty(0),
        )
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    keep = s > (s[0] * RCOND if len(s) and s[0] else 0)
    return u[:, keep], s[keep], vt[keep], s


def project(op, b):
    """Compute min-norm coefficients and the represented target without refitting."""
    u, s, vt, _ = op
    b = np.asarray(b, float)
    coef = vt.T @ ((u.T @ b) / s) if len(s) else np.zeros(vt.shape[1])
    return coef, u @ (u.T @ b)


def value_nullspace(v):
    """Use the parameter nullspace of the fixed scalar observations, not test data."""
    _, s, vt = np.linalg.svd(v, full_matrices=True)
    rank = int(np.sum(s > (s[0] * RCOND if len(s) and s[0] else 0)))
    return vt[rank:].T, rank


def objective(v, d, ell, k, theta, alpha):
    """Return scalar, density, ridge and complete losses in the original units."""
    score = float(np.sum((v @ theta - ell) ** 2))
    field = float(np.sum((d @ theta - k) ** 2))
    penalty = float(alpha * np.sum(theta[1:] ** 2))
    return np.array([score, field, penalty, score + field + penalty])


def diagnose(v, d, ell, k, theta, alpha, anchor_columns):
    """Separate source representability, feasible improvements and objective forces."""
    v, d, ell, k, theta = (np.asarray(x, float) for x in (v, d, ell, k, theta))
    if (
        v.shape != (len(ell), len(theta))
        or d.shape != (len(k), len(theta))
        or not all(np.isfinite(x).all() for x in (v, d, ell, k, theta))
    ):
        raise ValueError("Aligned finite source system required")
    anchor_columns = np.asarray(anchor_columns, int)
    if np.any(d[:, anchor_columns] != 0):
        raise ValueError("Anchor columns must have exactly zero BODY derivative")
    original = objective(v, d, ell, k, theta, alpha)
    field_op = svd_operator(d)
    field_theta, represented = project(field_op, k)
    null, v_rank = value_nullspace(v)
    null_op = svd_operator(d @ null)
    e = k - d @ theta
    null_coef, feasible_response = project(null_op, e)
    delta = null @ null_coef
    preserved = theta + delta
    anchor_op = svd_operator(v[:, anchor_columns])
    original_anchor_coef, original_anchor_response = project(anchor_op, ell - v @ theta)
    field_anchor_coef, field_anchor_response = project(anchor_op, ell - v @ field_theta)
    anchored = field_theta.copy()
    anchored[anchor_columns] += field_anchor_coef
    fvalue = v.T @ (v @ theta - ell)
    ffield = d.T @ (d @ theta - k)
    fpenalty = alpha * theta.copy()
    fpenalty[0] = 0
    forces = np.stack((fvalue, ffield, fpenalty))
    penalties = np.eye(len(theta)) * alpha
    penalties[0, 0] = 0
    system = v.T @ v + d.T @ d + penalties
    preserved_loss = objective(v, d, ell, k, preserved, alpha)
    quadratic = float(delta @ system @ delta)
    objective_change = float(preserved_loss[-1] - original[-1])
    external = k - represented
    remaining = e - feasible_response
    norm = lambda x: float(np.linalg.norm(x))
    ratio = lambda a, b: float(a / b) if b else float("nan")
    diagnostics = {
        "parameters": len(theta),
        "source_authors": len(ell),
        "density_constraints": len(k),
        "field_rank": len(field_op[1]),
        "value_rank": v_rank,
        "value_null_dimension": null.shape[1],
        "feasible_field_rank": len(null_op[1]),
        "anchor_rank": len(anchor_op[1]),
        "field_smallest_retained_singular": float(field_op[1][-1])
        if len(field_op[1])
        else 0,
        "field_condition": ratio(field_op[1][0], field_op[1][-1])
        if len(field_op[1])
        else float("nan"),
        "source_score_loss": original[0],
        "source_field_loss": original[1],
        "source_penalty": original[2],
        "source_objective": original[3],
        "target_energy": float(k @ k),
        "represented_target_energy": float(represented @ represented),
        "dictionary_external_loss": float(external @ external),
        "dictionary_external_share_of_realized": ratio(
            external @ external, original[1]
        ),
        "unused_dictionary_gain": float(original[1] - external @ external),
        "field_min_norm": norm(field_theta),
        "original_coefficient_norm": norm(theta),
        "field_norm_ratio": ratio(norm(field_theta), norm(theta)),
        "preserved_score_field_loss": preserved_loss[1],
        "preserved_score_field_gain": float(original[1] - preserved_loss[1]),
        "preserved_value_change": norm(v @ delta),
        "preserved_gradient_change": norm(d @ delta),
        "preserving_delta_norm": norm(delta),
        "preserving_delta_norm_ratio": ratio(norm(delta), norm(theta)),
        "preserving_penalty_change": float(preserved_loss[2] - original[2]),
        "preserving_objective_change": objective_change,
        "anchor_condition": ratio(anchor_op[1][0], anchor_op[1][-1])
        if len(anchor_op[1])
        else float("nan"),
        "original_anchor_residual": norm(ell - v @ theta - original_anchor_response),
        "original_anchor_coefficient_norm": norm(original_anchor_coef),
        "field_anchor_residual": norm(ell - v @ field_theta - field_anchor_response),
        "field_anchor_coefficient_norm": norm(field_anchor_coef),
        "anchored_field_loss": objective(v, d, ell, k, anchored, alpha)[1],
        "anchored_score_loss": objective(v, d, ell, k, anchored, alpha)[0],
        "anchored_penalty": objective(v, d, ell, k, anchored, alpha)[2],
        "force_value_norm": norm(fvalue),
        "force_field_norm": norm(ffield),
        "force_penalty_norm": norm(fpenalty),
        "force_value_field_inner": float(fvalue @ ffield),
        "force_value_penalty_inner": float(fvalue @ fpenalty),
        "force_field_penalty_inner": float(ffield @ fpenalty),
        "stationarity_error": norm(forces.sum(0)),
        "orthogonal_residual_inner": abs(float(external @ represented)),
        "feasible_residual_inner": abs(float(remaining @ feasible_response)),
        "parameter_projection_relative_error": norm(d @ field_theta - represented)
        / max(norm(represented), 1),
        "feasible_projection_relative_error": norm(d @ delta - feasible_response)
        / max(norm(feasible_response), 1),
        "objective_identity_relative_error": abs(objective_change - quadratic)
        / max(1, abs(objective_change), abs(quadratic)),
        "diagnostic_ls_solves": 4,
        "svd_decompositions": 4,
    }
    packet = {
        "field_theta": field_theta,
        "preserving_delta": delta,
        "anchored_field_theta": anchored,
        "original_anchor_coef": original_anchor_coef,
        "field_anchor_coef": field_anchor_coef,
        "anchor_columns": anchor_columns,
        "forces": forces,
        "field_singular_values": field_op[3],
        "feasible_singular_values": null_op[3],
        "anchor_singular_values": anchor_op[3],
    }
    return diagnostics, packet
