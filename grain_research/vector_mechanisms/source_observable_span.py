#!/usr/bin/env python3
"""Complete the source-observable representer span without changing the objective."""

import numpy as np

from . import observable_reader_teaching as prior

VARIANTS = ("expanded_vector_control", "expanded_observable_teaching")


def extend_basis(u0, rulers):
    """Retain U0 exactly; append ruler directions above original-scale roundoff."""
    u0, r = np.asarray(u0, float), np.asarray(rulers, float)
    if r.ndim != 2 or u0.shape[0] != r.shape[1] or not np.isfinite(r).all():
        raise ValueError("Finite source ruler rows required")
    np.testing.assert_allclose(u0.T @ u0, np.eye(u0.shape[1]), atol=1e-10)
    residual = r - (r @ u0) @ u0.T
    scale = np.linalg.svd(r, compute_uv=False)[0]
    tol = max(r.shape) * np.finfo(float).eps * scale
    _, s, vt = np.linalg.svd(residual, full_matrices=False)
    extra = vt[s > tol].T
    if extra.shape[1]:
        extra -= u0 @ (u0.T @ extra)
        extra, _ = np.linalg.qr(extra, mode="reduced")
    u = np.column_stack((u0, extra))
    np.testing.assert_allclose(u.T @ u, np.eye(u.shape[1]), atol=1e-10)
    coverage = np.linalg.norm(r - (r @ u) @ u.T) / max(
        np.linalg.norm(r), np.finfo(float).tiny
    )
    if coverage > 10 * max(r.shape) * np.finfo(float).eps:
        raise ValueError("Ruler span incomplete")
    return u, {"ruler_coverage_relative": coverage, "extension_rank_tolerance": tol}


def normal_system(model, u, include_observables):
    """Build the unchanged source objective in the supplied orthonormal basis."""
    a, y, w = [model[k] for k in ("a", "target", "weights")]
    aw = a * w[:, None]
    h = np.kron(aw.T @ aw, np.eye(u.shape[1]))
    h += float(model["regularizer"]) * np.eye(len(h))
    b = (aw.T @ (y * w[:, None]) @ u).ravel()
    if include_observables:
        for r in (model["source_rulers"][:, :1], model["source_rulers"][:, 1:]):
            z, t, _ = prior.functional_rows(a, u, y, r, model["groups"])
            h += z.T @ z
            b += z.T @ t
    return h, b


def objective(model, include_observables):
    """Read the exact fitted objective, not an unweighted prediction score."""
    n = 3 if include_observables else 1
    return float(model["source_losses"][:n].sum() + model["regularization_penalty"])


def fit(model, unique_rulers, include_observables):
    """One nested-span SPD solve using frozen source packages only."""
    u0 = model["output_basis"]
    u, info = extend_basis(u0, unique_rulers)
    h, b = normal_system(model, u, include_observables)
    embedded = np.zeros((model["a"].shape[1], u.shape[1]))
    embedded[:, : u0.shape[1]] = model["theta"]
    gradient = (2 * (h @ embedded.ravel() - b)).reshape(embedded.shape)
    theta = np.linalg.solve(h, b).reshape(embedded.shape)
    residual = np.linalg.norm(h @ theta.ravel() - b) / max(
        np.linalg.norm(b), np.finfo(float).tiny
    )
    if residual > 1e-9 or not np.isfinite(theta).all():
        raise ValueError("Expanded solve did not converge")
    beta = model["transform"] @ theta @ u.T
    fitted = model["a"] @ theta @ u.T
    losses, zero = prior.losses(
        fitted - model["target"],
        model["target"],
        model["source_rulers"],
        model["groups"],
    )
    difference = np.linalg.norm(beta - model["beta"]) / np.linalg.norm(model["beta"])
    if not include_observables and difference > 1e-7:
        raise ValueError("Expanded vector control differs from restricted solution")
    extra = theta[:, u0.shape[1] :]
    result = (
        model
        | info
        | {
            "output_basis": u,
            "original_output_basis": u0,
            "theta": theta,
            "beta": beta,
            "source_losses": losses,
            "source_zero_energy": zero,
            "coefficient_count": np.asarray(theta.size),
            "normal_residual_relative": np.asarray(residual),
            "prior_beta_relative": np.asarray(difference),
            "regularization_penalty": np.asarray(
                float(model["regularizer"]) * np.sum(theta**2)
            ),
            "target_outside_relative": np.linalg.norm(
                model["target"] - model["target"] @ u @ u.T
            )
            / np.linalg.norm(model["target"]),
            "old_gradient_norm": np.linalg.norm(gradient[:, : u0.shape[1]]),
            "missing_gradient_norm": np.linalg.norm(gradient[:, u0.shape[1] :]),
            "extra_coefficient_energy": np.sum(extra**2),
            "extra_source_field_energy": np.sum((model["a"] @ extra) ** 2),
            "old_objective": objective(model, include_observables),
        }
    )
    result["new_objective"] = objective(result, include_observables)
    if result["new_objective"] > result["old_objective"] + 1e-9 * max(
        1, result["old_objective"]
    ):
        raise ValueError("Nested objective increased")
    return result


def direction_usage(predicted, native, model):
    """Post-freeze signed account of using directions unavailable to old readers."""
    u, origin = model["original_output_basis"], model["d0"]
    p, t = predicted - origin, native - origin
    added, missing = p - p @ u @ u.T, t - t @ u @ u.T
    energy = float(np.sum(missing**2))
    error = float(np.sum((added - missing) ** 2))
    return {
        "added_energy": float(np.sum(added**2)),
        "native_missing_energy": energy,
        "remaining_missing_error2": error,
        "added_native_dot": float(np.sum(added * missing)),
        "signed_explained_fraction": 1 - error / energy if energy else 0.0,
        "zero_native_missing": energy == 0,
    }
