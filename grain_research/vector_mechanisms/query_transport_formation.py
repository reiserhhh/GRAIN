#!/usr/bin/env python3
"""Signed telescoping of query-conditioned tangents before block25 MLP."""

import numpy as np

UPDATE_LABELS = tuple(
    f"block{i:02d}_{kind}" for i in range(17, 25) for kind in ("attention", "mlp")
) + ("block25_attention",)
STAGES = ("initial",) + UPDATE_LABELS
WINDOWS = (
    ("blocks17_20", 0, 9),
    ("blocks21_23", 9, 15),
    ("block24_attention25", 15, 18),
)


def prefix_updates(updates, labels):
    """Extract exactly the pre-MLP25 prefix; later increments cannot enter the sum."""
    a = np.asarray(updates)
    if (
        a.ndim != 3
        or a.shape[0] != 4
        or a.shape[1] < 17
        or tuple(labels[:17]) != UPDATE_LABELS
    ):
        raise ValueError("Expected four directions and ordered pre-MLP25 updates")
    return a[:, :17].astype(float, copy=True)


def decompose(initial, updates, shared, boundary):
    """Keep all signed updates and bound scalar disagreement by observed vector rounding."""
    i, u, w, d = [np.asarray(a, float) for a in (initial, updates, shared, boundary)]
    if (
        i.ndim != 3
        or i.shape[:2] != (2, 4)
        or d.shape != i.shape
        or u.shape != (2, 4, 17, i.shape[-1])
        or w.shape != (2, i.shape[-1])
        or any(not np.isfinite(a).all() for a in (i, u, w, d))
    ):
        raise ValueError(
            "Two queries, four factors, seventeen updates and two readouts required"
        )
    parts = np.concatenate((i[:, :, None, :], u), axis=2)
    rebuilt = parts.sum(2)
    error = rebuilt - d
    relative = np.linalg.norm(error, axis=(1, 2)) / np.maximum(
        np.linalg.norm(d, axis=(1, 2)), 1e-30
    )
    absolute = np.max(abs(error), axis=(1, 2))
    if not np.all((relative <= 1e-4) | (absolute <= 1e-5)):
        raise ValueError("Pre-MLP25 vector parity exceeds original tolerance")
    vector_difference = parts[0] - parts[1]
    phi = np.einsum("fsh,mh->mfs", vector_difference, w)
    query_actions = np.einsum("qfsh,mh->qmfs", parts, w)
    target = np.einsum("fh,mh->mf", d[0] - d[1], w)
    scalar_error = phi.sum(-1) - target
    bound = (
        np.linalg.norm(w, axis=1)[:, None]
        * np.linalg.norm(error[0] - error[1], axis=-1)[None, :]
    )
    if np.any(abs(scalar_error) > bound + 1e-12):
        raise ValueError("Scalar disagreement exceeds the vector-rounding bound")
    return {
        "parts": parts,
        "phi": phi,
        "query_actions": query_actions,
        "target": target,
        "vector_difference": vector_difference,
        "vector_relative": relative,
        "vector_absolute": absolute,
        "scalar_error": scalar_error,
        "scalar_bound": bound,
        "difference_norm": np.linalg.norm(d[0] - d[1], axis=-1),
    }


def moment_accounting(values, target):
    """Return full signed second-moment/covariance matrices with ddof=0, plus closure."""
    x, y = np.asarray(values, float), np.asarray(target, float)
    if (
        x.ndim != 2
        or y.shape != (len(x),)
        or not len(x)
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError("Matched finite observations and stage columns required")
    mean = x.mean(0)
    centered = x - mean
    second = x.T @ x / len(x)
    covariance = centered.T @ centered / len(x)
    e = np.mean(y * y)
    result = {
        "mean": mean,
        "second": second,
        "covariance": covariance,
        "share": x.T @ y / len(x) / e if e else np.full(x.shape[1], np.nan),
        "target_energy": float(e),
        "target_variance": float(y.var()),
        "target_mean": float(y.mean()),
        "reconstructed_energy": float(second.sum()),
        "reconstructed_variance": float(covariance.sum()),
        "diagonal_energy": float(np.trace(second)),
        "cross_energy": float(second.sum() - np.trace(second)),
        "sum_error_max": float(abs(x.sum(1) - y).max()),
        "moment_covariance_closure": float(
            np.max(abs(second - covariance - np.outer(mean, mean)))
        ),
    }
    return result
