#!/usr/bin/env python3
"""Five observable reader coordinates and a coherent material-dependent lift."""

import numpy as np

from . import material_reader_field as base

VARIANTS = ("fixed_evidence_coordinates", "moving_evidence_coordinates")


def frame(primal, vectors):
    """Use the fixed response/source arm, common primal and four total actions."""
    p, v = np.asarray(primal, float), np.asarray(vectors, float)
    if p.ndim != 3 or p.shape[:2] != (2, 4) or v.shape != (2, 4, 3, 4, p.shape[-1]):
        raise ValueError("Two queries and four fixed compositions required")
    return np.concatenate((p[:, 3].mean(axis=0)[None], v[:, 3, 2].mean(axis=0)))


def frame_geometry(r):
    """Require full numerical row rank; report singular values without tuning."""
    r = np.asarray(r, float)
    if r.ndim != 2 or r.shape[0] != 5 or not np.isfinite(r).all():
        raise ValueError("Five finite frame rows required")
    s = np.linalg.svd(r, compute_uv=False)
    tol = max(r.shape) * np.finfo(float).eps * s[0]
    if np.sum(s > tol) != 5:
        raise ValueError("Five-coordinate frame lost numerical rank")
    return {
        "singular_values": s,
        "condition": float(s[0] / s[-1]),
        "rank_tolerance": tol,
    }


def right_inverse(r):
    """Full-row-rank Moore-Penrose inverse via the small frame Gram matrix."""
    frame_geometry(r)
    u = np.linalg.solve(r @ r.T, r).T
    np.testing.assert_allclose(r @ u, np.eye(5), atol=1e-8)
    return u


def inverse_tangent(r, dr, u):
    """Differentiate R^T(RR^T)^-1 in each supplied constant-rank direction."""
    si = np.linalg.solve(r @ r.T, np.eye(5))
    ds = dr @ r.T + r @ np.swapaxes(dr, -1, -2)
    return np.swapaxes(dr, -1, -2) @ si - u @ ds @ si


def lift_value(db, r, c, d0):
    """Minimum change to db satisfying the five predicted residual coordinates."""
    u = right_inverse(r)
    return db + u @ (c - r @ (db - d0))


def lift(db, ddb, r, dr, c, dc, d0):
    """Lift both value and derivative with frame, coordinate and baseline terms."""
    u = right_inverse(r)
    error = c - r @ (db - d0)
    derror = dc - np.einsum("fkh,h->fk", dr, db - d0) - ddb @ r.T
    d = db + u @ error
    dd = ddb + np.einsum("fhk,k->fh", inverse_tangent(r, dr, u), error) + derror @ u.T
    constraint = r @ (d - d0) - c
    tangent_constraint = np.einsum("fkh,h->fk", dr, d - d0) + dd @ r.T - dc
    change = d - db
    null_change = change - u @ (r @ change)
    checks = np.asarray(
        [
            np.max(abs(constraint)),
            np.max(abs(tangent_constraint)),
            np.max(abs(null_change)),
        ]
    )
    if not np.isfinite(d).all() or not np.isfinite(dd).all() or max(checks) > 1e-7:
        raise ValueError("Lift constraint or complement preservation failed")
    return d, dd, checks


def observation_coordinates(reader_target, frames, frame_derivatives, groups):
    """Use clamped value residuals for BODY product terms and original for context."""
    y = np.einsum("nkh,nh->nk", frames, reader_target)
    if len(y) != 28:
        raise ValueError("Fixed 28-observation contract required")
    np.testing.assert_array_equal(groups, np.repeat(np.arange(3), [8, 16, 4]))
    residuals = np.zeros_like(reader_target)
    for i in range(4):
        residuals[8 + 4 * i : 12 + 4 * i] = reader_target[2 * i + 1]
        residuals[24 + i] = reader_target[2 * i]
    return y + np.einsum("nkh,nh->nk", frame_derivatives, residuals)


def fit_coordinates(old_model, target):
    """One fixed source kernel solve, old numeric lambda, new source-group units."""
    x, v, kinds = [old_model[k] for k in ("x", "v", "kinds")]
    k = base.gram(x, v, kinds, x, v, kinds)
    w, zero = base.observation_weights(target, old_model["groups"])
    kw = k * w[:, None] * w[None]
    lam = float(old_model["regularizer"])
    h = kw + lam * np.eye(len(k))
    rhs = w[:, None] * target
    theta = np.linalg.solve(h, rhs)
    beta = w[:, None] * theta
    residual = np.linalg.norm(h @ theta - rhs) / max(
        np.linalg.norm(rhs), np.finfo(float).tiny
    )
    if not np.isfinite(beta).all() or residual > 1e-9:
        raise ValueError("Coordinate solve failed")
    fit = k @ beta
    loss = []
    for gi in range(3):
        use = old_model["groups"] == gi
        energy = float(np.sum(target[use] ** 2))
        loss.append(np.sum((fit[use] - target[use]) ** 2) / energy if energy else 0.0)
    return {
        "x": x,
        "v": v,
        "kinds": kinds,
        "beta": beta,
        "weights": w,
        "d0": np.zeros(5),
        "regularizer": lam,
        "target": target,
        "groups": old_model["groups"],
        "group_loss": np.asarray(loss),
        "zero_target_groups": zero,
        "normal_residual_relative": residual,
        "regularization_penalty": float(lam * np.sum(beta * (k @ beta))),
        "coefficients": beta.size,
    }


def readers(baseline, model, blocks, tangents, frame_value, frame_derivative):
    """Evaluate one coherent paired reader using only source and current inputs."""
    x, dx = [
        base.flatten(a, baseline["scale"], "grain_field") for a in (blocks, tangents)
    ]
    db, ddb = base.field(baseline, x, dx)
    c, dc = base.field(model, x, dx)
    r = model["fixed_frame"] if not bool(model["moving"]) else frame_value
    dr = (
        np.zeros_like(frame_derivative)
        if not bool(model["moving"])
        else frame_derivative
    )
    d, dd, checks = lift(db, ddb, r, dr, c, dc, baseline["d0"])
    return (
        np.stack((baseline["m"] + d / 2, baseline["m"] - d / 2)),
        np.stack((dd / 2, -dd / 2)),
        c,
        checks,
    )


def value_readers(baseline, model, blocks, frame_value):
    """Value-only evaluation for independent whole-field finite differences."""
    x = base.flatten(blocks, baseline["scale"], "grain_field")
    db, c = base.field(baseline, x), base.field(model, x)
    r = frame_value if bool(model["moving"]) else model["fixed_frame"]
    d = lift_value(db, r, c, baseline["d0"])
    return np.stack((baseline["m"] + d / 2, baseline["m"] - d / 2))
