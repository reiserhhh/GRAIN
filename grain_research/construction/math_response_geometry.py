#!/usr/bin/env python3
"""Source-referenced geometry of fixed conditional observers, not trait axes.

Euclidean overlap, empirical response overlap and transport error are distinct
estimands. The source support is a numerical row space, not a denoising mask.
"""

from __future__ import annotations

from itertools import product

import numpy as np


def gradient(operator, action):
    """Return the exact object derivative of the compiled bilinear observer."""
    action = np.asarray(action, float)
    if action.ndim != 2:
        raise ValueError("two-dimensional action rows required")
    return (
        operator["object_linear"]
        + ((action @ operator["source_action"].T) * operator["effective_dual"])
        @ operator["source_object"]
    )


def row_basis(x):
    """Use the standard machine-resolution SVD rank, never a tuned rank grid."""
    x = np.asarray(x, float)
    if x.ndim != 2 or not np.all(np.isfinite(x)):
        raise ValueError("finite two-dimensional rows required")
    _, s, vt = np.linalg.svd(x, full_matrices=False)
    tolerance = np.finfo(float).eps * max(x.shape) * (s[0] if len(s) else 0)
    return vt[s > tolerance], s, float(tolerance)


def principal_cosines(x, y):
    """Compute canonical subspace cosines without identifying their outputs."""
    bx, _, _ = row_basis(x)
    by, _, _ = row_basis(y)
    return np.clip(np.linalg.svd(bx @ by.T, compute_uv=False), 0, 1)


def support_parts(delta, basis):
    """Split displacement into source-observed support and its exact complement."""
    delta, basis = np.asarray(delta, float), np.asarray(basis, float)
    supported = (delta @ basis.T) @ basis
    return supported, delta - supported


def overlap(x, y):
    """Return signed cosine and source-unit disagreement, retaining zero cases."""
    x, y = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
    if x.shape != y.shape or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("aligned finite observations required")
    dot, nx, ny = float(x @ y), float(x @ x), float(y @ y)
    xc, yc = x - x.mean(), y - y.mean()
    centered_norm = float(np.linalg.norm(xc) * np.linalg.norm(yc))
    return {
        "n": len(x),
        "cosine": dot / np.sqrt(nx * ny) if nx * ny > 0 else None,
        "pearson": float(xc @ yc) / centered_norm if centered_norm > 0 else None,
        "x_rms": float(np.sqrt(nx / len(x))),
        "y_rms": float(np.sqrt(ny / len(y))),
        "difference_rms": float(np.sqrt(np.mean((y - x) ** 2))),
        "shared_mean_rms": float(np.sqrt(np.mean(((x + y) / 2) ** 2))),
        "target_contrast_rms": float(np.sqrt(np.mean(((y - x) / 2) ** 2))),
    }


def source_transport(x, y):
    """Fit one declared zero-intercept response-unit conversion on source only."""
    x, y = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
    energy = float(x @ x)
    if energy == 0:
        raise ValueError("zero source response cannot define a transport")
    beta = float(x @ y) / energy
    return {"beta": beta, "source_mse": float(np.mean((y - beta * x) ** 2))}


def transport_metrics(x, y, beta):
    """Apply a frozen unit conversion, with zero and blind-identity controls."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    error = y - float(beta) * x
    zero = float(np.mean(y**2))
    mse = float(np.mean(error**2))
    return {
        "mse": mse,
        "zero_mse": zero,
        "identity_mse": float(np.mean((y - x) ** 2)),
        "mse_over_zero": mse / zero if zero > 0 else None,
        "beta": float(beta),
    }


def cold_bank(spec, source_rows):
    """Freeze new object families crossed with known actions and author roles."""
    actions = sorted(
        {
            (r["target"], r["action_index"], r["action"], r["action_type"])
            for r in source_rows
            if r["role"] == "source"
        }
    )
    rows = []
    for (action_target, ai, action, action_type), domain in product(
        actions, sorted(spec["cold_objects"])
    ):
        for family, objects in enumerate(spec["cold_objects"][domain]):
            for variant, mode, own in product(
                ("bare", "joined", "decoy"), spec["modes"], (-1, 1)
            ):
                obj = objects[variant]
                subject = "I" if own == 1 else "my colleague"
                text = (
                    f"{subject.capitalize()} did {action} {obj}."
                    if mode == "actual"
                    else f"{subject.capitalize()} {'want' if own == 1 else 'wants'} to {action} {obj}."
                )
                at = text.index(action + " " + obj)
                rows.append(
                    {
                        "material_id": f"geometry_{domain}_{family}_{action_target}_{ai}_{variant}_{mode}_{own}",
                        "target": domain,
                        "family": family,
                        "action_target": action_target,
                        "action_index": ai,
                        "action_type": action_type,
                        "variant": variant,
                        "mode": mode,
                        "own": own,
                        "text": text,
                        "events": [
                            {
                                "event_id": "e",
                                "span": [0, len(text)],
                                "action_span": [at, at + len(action)],
                                "action": action,
                                "object": obj,
                                "actor": own,
                                "conceiver": own,
                                "mode": mode,
                                "bits": [
                                    1
                                    if action_target == domain
                                    and action_type == 1
                                    and variant != "decoy"
                                    else -1,
                                    1,
                                    own,
                                ],
                            }
                        ],
                    }
                )
    return rows
