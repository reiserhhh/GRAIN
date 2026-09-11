#!/usr/bin/env python3
"""Fixed-weight value/derivative conditioning and coherent teaching innovations."""

import numpy as np

from . import material_reader_field as base

LEVELS = ("V", "VB", "VBC")
FRAMES = ("fixed", "moving")
VARIANTS = tuple(f"{frame}_{level}" for frame in FRAMES for level in LEVELS)


def nested_models(template, order=(0, 1, 2)):
    """Assimilate source groups without changing their kernel or noise weights.

    Computation uses the whitened Gram matrix for numerical stability. Each
    returned beta has the full observation shape with unused entries zero;
    innovation_beta is the conditional update, not a raw full-fit group term.
    Alternative orders are provided for algebra tests, not empirical selection.
    """
    if sorted(order) != [0, 1, 2]:
        raise ValueError("Each of the three source groups is required once")
    x, v, kinds, y, groups, w = [
        np.asarray(template[k])
        for k in ("x", "v", "kinds", "target", "groups", "weights")
    ]
    n = len(x)
    lam = float(template["regularizer"])
    if (
        y.ndim != 2
        or y.shape[0] != n
        or w.shape != (n,)
        or groups.shape != (n,)
        or set(groups) != {0, 1, 2}
        or not np.isfinite(y).all()
        or not np.isfinite(w).all()
        or np.any(w <= 0)
        or not np.isfinite(lam)
        or lam <= 0
    ):
        raise ValueError(
            "Finite source observations and positive frozen weights required"
        )
    k = base.gram(x, v, kinds, x, v, kinds)
    h = w[:, None] * k * w[None] + lam * np.eye(n)
    rhs = w[:, None] * y
    retained = np.array([], dtype=int)
    theta = np.empty((0, y.shape[1]))
    beta_old = np.zeros_like(y, dtype=float)
    result = []
    for group in order:
        added = np.flatnonzero(groups == group)
        jj = h[np.ix_(added, added)]
        if len(retained):
            ii = h[np.ix_(retained, retained)]
            ij = h[np.ix_(retained, added)]
            transport = np.linalg.solve(ii, ij)
            schur = jj - ij.T @ transport
            residual = rhs[added] - ij.T @ theta
        else:
            transport = np.empty((0, len(added)))
            schur, residual = jj, rhs[added]
        np.testing.assert_allclose(schur, schur.T, atol=1e-9, rtol=1e-10)
        eigen = np.linalg.eigvalsh((schur + schur.T) / 2)
        if eigen[0] <= 0 or not np.isfinite(eigen).all():
            raise ValueError("Conditional source covariance is not positive definite")
        gamma = np.linalg.solve(schur, residual)
        theta = np.concatenate((theta - transport @ gamma, gamma))
        retained = np.concatenate((retained, added))
        hs = h[np.ix_(retained, retained)]
        direct = np.linalg.solve(hs, rhs[retained])
        relative = float(
            np.linalg.norm(theta - direct)
            / max(np.linalg.norm(direct), np.finfo(float).tiny)
        )
        normal = float(
            np.linalg.norm(hs @ theta - rhs[retained])
            / max(np.linalg.norm(rhs[retained]), np.finfo(float).tiny)
        )
        if relative > 1e-7 or normal > 1e-7:
            raise ValueError("Block conditioning disagrees with direct source solve")
        beta = np.zeros_like(beta_old)
        beta[retained] = w[retained, None] * theta
        selected = np.zeros(n, dtype=bool)
        selected[retained] = True
        loss, zero = [], []
        fitted = k @ beta
        for gi in range(3):
            use = groups == gi
            energy = float(np.sum(y[use] ** 2))
            zero.append(energy == 0)
            loss.append(
                float(np.sum((fitted[use] - y[use]) ** 2) / energy) if energy else 0
            )
        result.append(
            dict(template)
            | {
                "beta": beta,
                "innovation_beta": beta - beta_old,
                "selected": selected,
                "added_group": np.asarray(group),
                "group_loss": np.asarray(loss),
                "zero_target_groups": np.asarray(zero),
                "direct_relative": np.asarray(relative),
                "normal_residual_relative": np.asarray(normal),
                "schur_min_eigenvalue": np.asarray(eigen[0]),
                "schur_condition": np.asarray(eigen[-1] / eigen[0]),
                "innovation_target_rms": np.asarray(np.sqrt(np.mean(residual**2))),
                "regularization_penalty": np.asarray(lam * np.sum(beta * (k @ beta))),
                "coefficients": np.asarray(len(retained) * y.shape[1]),
            }
        )
        beta_old = beta
    return result


def error_change(target, before, after):
    """Signed squared-error identity for an actual nested-function update."""
    target, before, after = [np.asarray(a, float) for a in (target, before, after)]
    error, delta = before - target, after - before
    e0, de, cross, e1 = [
        float(x)
        for x in (
            np.sum(error**2),
            np.sum(delta**2),
            2 * np.sum(error * delta),
            np.sum((after - target) ** 2),
        )
    ]
    np.testing.assert_allclose(e1, e0 + de + cross, atol=1e-10)
    energy = float(np.sum(target**2))
    return {
        "before_error2": e0,
        "innovation_energy": de,
        "signed_cross_energy": cross,
        "after_error2": e1,
        "error_change_over_target": (e1 - e0) / energy if energy else float("nan"),
        "innovation_rms": float(np.sqrt(np.mean(delta**2))),
        "signed_target_projection": float(np.sum(delta * target)) / energy
        if energy
        else float("nan"),
    }
