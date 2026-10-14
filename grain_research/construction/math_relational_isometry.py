#!/usr/bin/env python3
"""Supported orthogonal transport of vector observation rulers, not trait fitting."""

from __future__ import annotations

import numpy as np


def action_columns(values, rows, cells):
    """Extract a one-wording four-corner action; metadata is calibration only."""
    x = np.asarray(values, float)
    if x.ndim != 2 or len(x) != len(rows) or not np.isfinite(x).all():
        raise ValueError("Finite aligned vector bank required")
    d = []
    for f, c in cells:
        ids = [
            i
            for i, r in enumerate(rows)
            if r["family"] == f and (r["condition"] + 1) // 2 == c
        ]
        corners = [(rows[i]["polarity"], rows[i]["owner"]) for i in ids]
        if len(ids) != 4 or set(corners) != {(-1, -1), (-1, 1), (1, -1), (1, 1)}:
            raise ValueError("One wording and all four corners required")
        d.append(sum(rows[i]["polarity"] * rows[i]["owner"] * x[i] for i in ids) / 4)
    return np.stack(d, axis=1)


def supported_polar(source, target):
    """Fit only the identified support of an orthogonal Procrustes map."""
    x, y = np.asarray(source, float), np.asarray(target, float)
    if (
        x.ndim != 2
        or x.shape != y.shape
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError("Finite paired anchor matrices required")
    for a in (x, y):
        s = np.linalg.svd(a, compute_uv=False)
        if (
            len(s) != a.shape[1]
            or np.count_nonzero(s > np.finfo(float).eps * max(a.shape) * s[0])
            != a.shape[1]
        ):
            raise ValueError(
                "Rank-deficient support cannot declare a complete local ruler"
            )
    qx, rx = np.linalg.qr(x, mode="reduced")
    qy, ry = np.linalg.qr(y, mode="reduced")
    u, s, vt = np.linalg.svd(ry @ rx.T)
    return {"left": qy @ u, "right": qx @ vt.T, "cross_singular": s}


def score_vectors(values, bundle, arm):
    """Score numeric vectors and retain origins/complement; no design label input."""
    x = np.asarray(values, float)
    if (
        x.ndim != 2
        or x.shape[1] != len(bundle["source_center"])
        or not np.isfinite(x).all()
    ):
        raise ValueError("Finite frozen-layout vectors required")
    delta = x - bundle["context_center"]
    dual = bundle["dual"]
    raw = (x - bundle["source_center"]) @ dual.T
    if arm == "no_rotation":
        coordinates = delta @ dual.T
        complement = delta - coordinates @ bundle["source_columns"].T
    elif arm in ("matched_anchors", "wrong_anchors"):
        prefix = "matched" if arm == "matched_anchors" else "wrong"
        left, right = bundle[prefix + "_left"], bundle[prefix + "_right"]
        support = delta @ left
        coordinates = support @ (right.T @ dual.T)
        complement = delta - support @ left.T
    else:
        raise ValueError("Only registered fixed arms admitted")
    origin = (bundle["context_center"] - bundle["source_center"]) @ dual.T
    return {
        "coordinates": coordinates,
        "complement": complement,
        "raw_source_coordinates": raw,
        "origin_coordinates": np.repeat(origin[None], len(x), axis=0),
    }


def gram_account(source, target):
    """Report shape and exact best orthogonal residual without fitting scale."""
    x, y = np.asarray(source, float), np.asarray(target, float)
    polar = supported_polar(x, y)
    gx, gy = x.T @ x, y.T @ y
    nx, ny = np.linalg.norm(gx), np.linalg.norm(gy)
    predicted = polar["left"] @ (polar["right"].T @ x)
    minimum = np.sum(x * x) + np.sum(y * y) - 2 * polar["cross_singular"].sum()
    return {
        "source_gram": gx,
        "target_gram": gy,
        "normalized_gram_alignment": float(np.sum(gx * gy) / (nx * ny)),
        "normalized_gram_shape_gap": float(np.linalg.norm(gx / nx - gy / ny)),
        "raw_gram_gap_over_source": float(np.linalg.norm(gy - gx) / nx),
        "action_norm_ratio": float(np.linalg.norm(y) / np.linalg.norm(x)),
        "best_rotation_residual_sq": float(minimum),
        "direct_rotation_residual_sq": float(np.sum((y - predicted) ** 2)),
        "relative_best_rotation_residual": float(
            np.sqrt(max(0, minimum) / np.sum(y * y))
        ),
    }
