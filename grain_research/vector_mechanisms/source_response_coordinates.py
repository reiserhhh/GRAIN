#!/usr/bin/env python3
"""Response-informed source coordinates for a coupled scalar selector."""

from __future__ import annotations

import numpy as np

from . import source_selector_potential as previous
from .compiled_query_transfer import QUERIES
from .source_state_transport import features

PINV_RTOL = 1e-10


def capacity(stats, joint_coef):
    """Source-only minimum-norm derivative projection, never a predictor candidate."""
    g = (stats["gd"] + stats["gd"].T) / 2
    values, vectors = np.linalg.eigh(g)
    if values[-1] <= 0 or values[0] < -PINV_RTOL * values[-1]:
        raise ValueError("Positive semidefinite source Gram required")
    keep = values > PINV_RTOL * values[-1]
    u = vectors[:, keep]
    coef = u @ ((u.T @ stats["cd"]) / values[keep, None])
    minimum, current = previous.loss(coef, stats), previous.loss(joint_coef, stats)
    if minimum["d_loss"] < -1e-7 or minimum["d_loss"] > 1 + 1e-7:
        raise ValueError("Source derivative projection energy outside bounds")
    return {
        "design_width": len(g),
        "derivative_rank": int(keep.sum()),
        "pinv_rtol": PINV_RTOL,
        "derivative_representable_fraction": 1 - minimum["d_loss"],
        "derivative_minimum_residual_fraction": minimum["d_loss"],
        "value_loss_at_derivative_minimum": minimum["x_loss"],
        "joint_derivative_loss": current["d_loss"],
        "joint_value_loss": current["x_loss"],
        "joint_derivative_excess": current["d_loss"] - minimum["d_loss"],
        "minimum_coef_norm": float(np.linalg.norm(coef)),
        "joint_coef_norm": float(np.linalg.norm(joint_coef)),
        "retained_condition_number": float(values[-1] / values[keep][0]),
    }


def selected(items, members):
    """Require each permitted author's exact query pair before reading payloads."""
    out = [i for i in items if i["index"] in members]
    if len(out) != 2 * len(members) or {(i["index"], i["query"]) for i in out} != {
        (s, q) for s in members for q in QUERIES
    }:
        raise ValueError("Exact source-query materials required")
    return out


def build_basis(items, members, anchors, scale, rank=32, alpha=0.01):
    """Learn a fixed-width standardized cross-moment subspace only from sources.

    This uses four directional observations, not a complete function Jacobian or
    a certified active subspace. Input scales and feature scales are separate.
    """
    picked = selected(items, members)
    if set(anchors) != set(QUERIES):
        raise ValueError("Exact two split-specific anchors required")
    hidden = picked[0]["data"]["state"].shape[1]
    mean = np.zeros(hidden)
    for i in picked:
        x = np.asarray(i["data"]["state"], float)
        use = previous.allowed(i["data"]["mask"], len(x))
        mean += x[use].mean(0) / len(picked)
    variance = np.zeros(hidden)
    for i in picked:
        x = np.asarray(i["data"]["state"], float)
        use = previous.allowed(i["data"]["mask"], len(x))
        variance += ((x[use] - mean) ** 2).mean(0) / len(picked)
    input_scale = np.sqrt(variance)
    input_scale[input_scale < 1e-12] = 1
    columns, normalizers = [], {}
    for q in QUERIES:
        material = []
        sx, sd = 0.0, 0.0
        for i in picked:
            if i["query"] != q:
                continue
            data = i["data"]
            ell, dell = previous.logits(anchors[q], i["k"], i["dk"], data, scale)
            e0, d0 = previous.logits(anchors[q], i["k0"], i["dk0"], data, scale)
            use = previous.allowed(data["mask"], len(ell))
            y = previous.gauge(ell - e0, data["mask"])[use]
            dy = previous.gauge(dell - d0, data["mask"])[:, use]
            z = previous.gauge((data["state"] - mean) / input_scale, data["mask"])[use]
            dz = previous.gauge(i["d"] / input_scale, data["mask"])[:, use]
            sx += float(np.mean(y * y)) / len(members)
            sd += float(np.mean(dy * dy)) / len(members)
            material.append((z, dz, y, dy))
        sx, sd = np.sqrt(sx), np.sqrt(sd)
        if min(sx, sd) <= 1e-12:
            raise ValueError("Source target RMS must be positive")
        cx, cd = (
            np.zeros((hidden, len(anchors[q]))),
            np.zeros((hidden, len(anchors[q]))),
        )
        for z, dz, y, dy in material:
            weight = 1 / (len(members) * len(z))
            cx += weight * (z.T @ y) / sx
            for f in range(4):
                cd += weight / 4 * (dz[f].T @ dy[f]) / sd
        columns.extend([cx, cd])
        normalizers[q] = {"value_rms": float(sx), "direction_rms": float(sd)}
    cross = np.concatenate(columns, axis=1)
    u, singular, _ = np.linalg.svd(cross, full_matrices=False)
    threshold = max(cross.shape) * np.finfo(float).eps * singular[0]
    available = int(np.sum(singular > threshold))
    if not 0 < rank <= available:
        raise ValueError(
            f"Requested rank {rank} unsupported: source cross-moment rank {available}"
        )
    basis = u[:, :rank] / input_scale[:, None]
    center = np.zeros(2 * rank + 3)
    phis = []
    for i in picked:
        data = i["data"]
        p, _ = features(data["state"], None, mean, basis, "causal_context")
        use = previous.allowed(data["mask"], len(p))
        phis.append(p[use])
        center += p[use].mean(0) / len(picked)
    feature_scale = np.sqrt(
        sum(((p - center) ** 2).mean(0) for p in phis) / len(picked)
    )
    feature_scale[feature_scale < 1e-12] = 1
    center[-3], feature_scale[-3] = 0, 1
    model = {
        "mean": mean,
        "basis": basis,
        "center": center,
        "scale": feature_scale,
        "members": np.asarray(members),
        "mode": np.asarray("causal_context"),
        "rank": np.asarray(rank),
        "alpha": np.asarray(alpha),
        "input_scale": input_scale,
        "cross_singular": singular,
    }
    audit = {
        "cross_rank": available,
        "cross_columns": cross.shape[1],
        "kept_rank": rank,
        "retained_cross_energy": float(
            np.sum(singular[:rank] ** 2) / np.sum(singular**2)
        ),
        "last_retained_singular": float(singular[rank - 1]),
        "numerical_rank_threshold": float(threshold),
        "normalizers": normalizers,
    }
    return model, audit


def fit_joint(items, members, query, base, anchor, scale):
    """Exactly one joint solve, with the same alpha and zero gauge as NEXT74."""
    stats = previous.source_statistics(items, members, query, base, anchor, scale)
    active = np.arange(len(base["center"])) != len(base["center"]) - 3
    gx, gd = (
        stats["gx"] / (stats["heads"] * stats["rx"]),
        stats["gd"] / (stats["heads"] * stats["rd"]),
    )
    cx, cd = (
        stats["cx"] / (stats["heads"] * stats["rx"]),
        stats["cd"] / (stats["heads"] * stats["rd"]),
    )
    gram = (gx + gd)[np.ix_(active, active)] + float(base["alpha"]) * np.eye(
        active.sum()
    )
    coef = np.zeros_like(cx)
    coef[active] = np.linalg.solve(gram, (cx + cd)[active])
    if not np.isfinite(coef).all():
        raise ValueError("Finite source joint solve required")
    model = {k: np.array(v, copy=True) for k, v in base.items()} | {
        "coef": coef,
        "anchor": np.asarray(anchor),
        "query": np.asarray(query),
        "objective": np.asarray("response_coordinates_joint"),
        "source_rx": np.asarray(stats["rx"]),
        "source_rd": np.asarray(stats["rd"]),
    }
    training = previous.loss(coef, stats) | {
        "normal_error": float(np.max(abs(gram @ coef[active] - (cx + cd)[active]))),
        "source_rx": stats["rx"],
        "source_rd": stats["rd"],
    }
    return model, training
