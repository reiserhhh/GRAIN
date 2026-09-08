#!/usr/bin/env python3
"""Reference-only quadratic fields in Euclidean and directional-action norms."""

from __future__ import annotations

import numpy as np

from grain_research.vector_mechanisms.centered_adjoint import center_groups


def action_projection(displacement, gradient):
    """Split a field into its current token-direction action and null component."""
    d, g = np.asarray(displacement, float), np.asarray(gradient, float)
    if d.ndim != 2 or d.shape != g.shape or not np.isfinite(d + g).all():
        raise ValueError("Finite matched token matrices required")
    action = np.sum(d * g, axis=1)
    energy = np.sum(d * d, axis=1)
    parallel = (
        d
        * np.divide(action, energy, out=np.zeros_like(action), where=energy > 0)[
            :, None
        ]
    )
    return action, parallel, g - parallel


def spectral_dictionary(model, relative_tolerance=1e-10):
    """Keep every nonzero eigenmode of the symmetric M16 score operator."""
    u = np.asarray(model["projection"], float) / np.asarray(model["scale"], float)
    v = np.asarray(model["coef"], float).T
    if u.ndim != 2 or u.shape != v.shape or not np.isfinite(u + v).all():
        raise ValueError("Matched finite low-rank operator factors required")
    q, r = np.linalg.qr(np.column_stack([u, v]), mode="reduced")
    k = u.shape[1]
    product = r[:, :k] @ r[:, k:].T
    values, vectors = np.linalg.eigh((product + product.T) / 2)
    keep = np.abs(values) > relative_tolerance * np.max(np.abs(values))
    if not keep.any():
        raise ValueError("No nonzero action directions in the reference field")
    return {"vectors": q @ vectors[:, keep], "original": values[keep]}


def fit_action_dictionary(packets, dictionary, relative_penalty=0.01):
    """Fit both norms in an identical frozen function family, never using check authors."""
    refs = [p for p in packets if p["role"] == "reference"]
    if (
        not refs
        or relative_penalty <= 0
        or any(p["role"] not in {"reference", "evaluation"} for p in packets)
    ):
        raise ValueError("Explicit author roles and positive penalty required")
    v, prior = dictionary["vectors"], dictionary["original"]
    if not np.allclose(v.T @ v, np.eye(len(prior)), atol=1e-8):
        raise ValueError("Orthonormal frozen directions required")
    rs, gs, actions, weights = [], [], [], []
    for packet in refs:
        d = center_groups(packet["displacement"], packet["labels"])
        g = len(d) * center_groups(packet["integrated"], packet["labels"])
        a, _, _ = action_projection(d, g)
        rs.append(d @ v)
        gs.append(g @ v)
        actions.append(a)
        weights.append(np.full(len(d), 1 / (len(refs) * len(d))))
    r, g, a, w = map(np.concatenate, (rs, gs, actions, weights))
    square = r * r
    scale = np.sqrt(np.sum(w[:, None] * square**2, axis=0))
    scale = np.where(scale > 1e-12, scale, 1.0)
    x = square / scale
    original = prior * scale
    grams = {
        "gradient_refit": np.diag(np.sum(w[:, None] * square, axis=0) / scale**2),
        "action_refit": x.T @ (w[:, None] * x),
    }
    cross = {
        "gradient_refit": np.sum(w[:, None] * r * g, axis=0) / scale,
        "action_refit": x.T @ (w * a),
    }
    result = {**dictionary, "feature_scale": scale}
    for method, gram in grams.items():
        penalty = relative_penalty * np.trace(gram) / len(prior)
        regularized = gram + penalty * np.eye(len(prior))
        theta = original + np.linalg.solve(regularized, cross[method] - gram @ original)
        result[method] = theta / scale
        result[f"{method}_penalty"] = np.asarray(penalty)
        result[f"{method}_condition"] = np.asarray(np.linalg.cond(regularized))
    return result


def read_action_dictionary(displacement, labels, model):
    """Read every frozen field's scalar action on one material without a suffix model."""
    d = center_groups(displacement, labels)
    square = (d @ model["vectors"]) ** 2
    return {
        method: float(np.mean(square @ model[method]))
        for method in ("original", "gradient_refit", "action_refit")
    }


def action_diagnostics(packet, model):
    """Separate quadrature error, token cancellation, field error and action-null energy."""
    d = center_groups(packet["displacement"], packet["labels"])
    g = len(d) * center_groups(packet["integrated"], packet["labels"])
    action, parallel, null = action_projection(d, g)
    total = float(np.sum(g * g))
    r = d @ model["vectors"]
    rows = []
    for method in ("original", "gradient_refit", "action_refit"):
        predicted_field = (r * model[method]) @ model["vectors"].T
        predicted_action = (r**2) @ model[method]
        field_error = action - predicted_action
        rows.append(
            {
                "query": packet["query"],
                "role": packet["role"],
                "method": method,
                "tokens": len(d),
                "endpoint": packet["actual"],
                "integral": float(action.mean()),
                "prediction": float(predicted_action.mean()),
                "quadrature_error": float(action.mean() - packet["actual"]),
                "field_error": float(field_error.mean()),
                "token_action_mse": float(np.mean(field_error**2)),
                "gradient_mse": float(np.mean((g - predicted_field) ** 2)),
                "radial_energy_fraction": float(np.sum(parallel**2) / total),
                "null_action_max": float(np.max(np.abs(np.sum(d * null, axis=1)))),
                "signed_to_absolute_action": float(
                    abs(action.sum()) / np.abs(action).sum()
                ),
                "diagonal_token_error": float(np.mean(field_error**2) / len(d)),
                "offdiagonal_token_error": float(
                    field_error.mean() ** 2 - np.mean(field_error**2) / len(d)
                ),
            }
        )
    return rows
