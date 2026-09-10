#!/usr/bin/env python3
"""NEXT57: one source-fitted scalar and its compatible GRAIN derivatives."""

from __future__ import annotations

import numpy as np

OBJECTIVES = ("value", "joint")


def features(state, directions, positions, mean, basis):
    """Pool frozen BODY and final contextual states; differentiate that same map."""
    x, d, mu, u = (np.asarray(v, float) for v in (state, directions, mean, basis))
    pos = np.asarray(positions)
    if (
        x.ndim != 2
        or d.ndim != 3
        or d.shape[1:] != x.shape
        or mu.shape != (x.shape[1],)
        or u.ndim != 2
        or u.shape[0] != x.shape[1]
        or pos.ndim != 1
        or not np.issubdtype(pos.dtype, np.integer)
        or not len(pos)
        or np.any(np.diff(pos) <= 0)
        or pos[0] < 0
        or pos[-1] >= len(x) - 1
        or not all(np.isfinite(v).all() for v in (x, d, mu, u))
    ):
        raise ValueError(
            "Finite aligned states, directions, source basis and BODY required"
        )
    phi = np.concatenate(((x[pos].mean(axis=0) - mu) @ u, (x[-1] - mu) @ u))
    dphi = np.concatenate((d[:, pos].mean(axis=1) @ u, d[:, -1] @ u), axis=1)
    return phi, dphi


def source_arrays(phi, dphi, target, tau):
    """Validate one query's source-author values and directional responses."""
    x, d, y, t = (np.asarray(v, float) for v in (phi, dphi, target, tau))
    if (
        x.ndim != 2
        or len(x) < 3
        or d.ndim != 3
        or d.shape[0] != len(x)
        or d.shape[2] != x.shape[1]
        or y.shape != (len(x),)
        or t.shape != d.shape[:2]
        or not all(np.isfinite(v).all() for v in (x, d, y, t))
    ):
        raise ValueError(
            "At least three aligned finite source-author observations required"
        )
    return x, d, y, t


def design(phi, dphi, target, tau, model):
    """Construct explicitly averaged value/derivative losses and unpenalized bias."""
    x, d, y, t = source_arrays(phi, dphi, target, tau)
    z = (x - model["center"]) / model["scale"]
    dz = d / model["scale"]
    yn = (y - model["origin"]) / model["target_scale"]
    tn = t / model["target_scale"]
    av = np.column_stack((np.ones(len(x)), z)) / np.sqrt(len(x))
    ad = np.column_stack((np.zeros(t.size), dz.reshape(-1, x.shape[1]))) / np.sqrt(
        t.size
    )
    reg = np.column_stack((np.zeros(x.shape[1]), np.eye(x.shape[1])))
    reg *= np.sqrt(float(model["alpha"]))
    aa, bb = [av], [yn / np.sqrt(len(x))]
    if str(model["objective"]) == "joint":
        aa.append(ad)
        bb.append(tn.ravel() / np.sqrt(t.size))
    aa.append(reg)
    bb.append(np.zeros(x.shape[1]))
    return np.vstack(aa), np.concatenate(bb), z, dz, av, ad


def fit_scalar(phi, dphi, target, tau, objective, members, alpha=0.01):
    """Fit source-only normalizers and exactly one augmented least-squares model."""
    x, d, y, t = source_arrays(phi, dphi, target, tau)
    ids = np.asarray(members, int)
    if objective not in OBJECTIVES or alpha <= 0 or len(set(ids)) != len(x):
        raise ValueError(
            "Known objective, positive penalty and unique source membership required"
        )
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1
    model = {
        "center": x.mean(axis=0),
        "scale": scale,
        "origin": np.asarray(y.mean()),
        "target_scale": np.asarray(max(float(y.std()), 1e-6)),
        "alpha": np.asarray(alpha),
        "objective": np.asarray(objective),
        "members": ids,
    }
    a, b, z, dz, av, ad = design(x, d, y, t, model)
    coef, _, _, _ = np.linalg.lstsq(a, b, rcond=None)
    model["coef"] = coef
    yp, tp = apply_scalar(x, d, model)
    diag = {
        "source_authors": len(x),
        "features": x.shape[1],
        "value_rank": int(np.linalg.matrix_rank(av)),
        "directional_rank": int(np.linalg.matrix_rank(ad)),
        "joint_data_rank": int(np.linalg.matrix_rank(np.vstack((av, ad)))),
        "regularized_condition": float(np.linalg.cond(a)),
        "source_value_loss": float(np.mean(((yp - y) / model["target_scale"]) ** 2)),
        "source_derivative_loss": float(
            np.mean(((tp - t) / model["target_scale"]) ** 2)
        ),
        "penalty": float(alpha * np.sum(coef[1:] ** 2)),
        "normalized_direction_rms": float(np.sqrt(np.mean(dz**2))),
        "normalized_value_rms": float(np.sqrt(np.mean(z**2))),
        "last_direction_max": float(np.max(np.abs(d[:, :, x.shape[1] // 2 :]))),
        "normal_equation_max": float(np.max(np.abs(a.T @ (a @ coef - b)))),
    }
    return model, diag


def apply_scalar(phi, dphi, model):
    """Apply a frozen scalar without a teacher; return its exact directional derivative."""
    x, d = np.asarray(phi, float), np.asarray(dphi, float)
    if x.shape[-1] != len(model["center"]) or d.shape[-1] != x.shape[-1]:
        raise ValueError("Frozen feature dimensions required")
    z, dz = (x - model["center"]) / model["scale"], d / model["scale"]
    w = model["coef"][1:]
    return (
        model["origin"] + model["target_scale"] * (model["coef"][0] + z @ w),
        model["target_scale"] * (dz @ w),
    )
