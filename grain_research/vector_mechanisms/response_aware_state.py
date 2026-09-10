#!/usr/bin/env python3
"""Fixed-feature, source-only value and GRAIN-direction matching."""

from __future__ import annotations

import numpy as np

from .compiled_query_transfer import QUERIES
from .source_state_transport import features

OBJECTIVES = ("normalized_value", "joint")


def sufficient_statistics(items, members, model):
    """Accumulate exact equal-author/query weighted objectives without stacking targets."""
    selected = [i for i in items if i["index"] in members]
    if (
        len(selected) != 2 * len(members)
        or {(i["index"], i["query"]) for i in selected}
        != {(s, q) for s in members for q in QUERIES}
        or not np.array_equal(model["members"], members)
    ):
        raise ValueError("Exact source reference and paired materials required")
    width, hidden = model["coef"].shape
    stats = {
        "gx": np.zeros((width, width)),
        "gd": np.zeros((width, width)),
        "cx": np.zeros((width, hidden)),
        "cd": np.zeros((width, hidden)),
        "rx": 0.0,
        "rd": 0.0,
        "hidden": hidden,
    }
    for item in selected:
        x, y, d, dy = (np.asarray(item[k], float) for k in ("x", "y", "d", "dy"))
        if (
            x.ndim != 2
            or x.shape != y.shape
            or d.shape != (4, *x.shape)
            or dy.shape != d.shape
            or not all(np.isfinite(v).all() for v in (x, y, d, dy))
        ):
            raise ValueError(
                "Aligned finite source values and four directions required"
            )
        phi, dp = features(x, d, model["mean"], model["basis"], str(model["mode"]))
        z, dz = (phi - model["center"]) / model["scale"], dp / model["scale"]
        weight = 1 / (len(members) * 2 * len(x))
        target = y - x
        stats["gx"] += weight * (z.T @ z)
        stats["cx"] += weight * (z.T @ target)
        stats["rx"] += weight * float(np.sum(target**2)) / hidden
        for f in range(4):
            target = dy[f] - d[f]
            stats["gd"] += weight / 4 * (dz[f].T @ dz[f])
            stats["cd"] += weight / 4 * (dz[f].T @ target)
            stats["rd"] += weight / 4 * float(np.sum(target**2)) / hidden
    if min(stats["rx"], stats["rd"]) <= 1e-12:
        raise ValueError("Positive finite source residual normalizers required")
    return stats


def losses(coef, stats, alpha):
    """Evaluate both normalized losses even for a value-only fitted coefficient."""
    values = {}
    for tag in ("x", "d"):
        values[tag] = float(
            1
            + (
                np.sum(coef * (stats["g" + tag] @ coef))
                - 2 * np.sum(coef * stats["c" + tag])
            )
            / (stats["hidden"] * stats["r" + tag])
        )
    mask = np.ones(len(coef), bool)
    mask[-3] = False
    values["penalty"] = float(alpha * np.sum(coef[mask] ** 2))
    return values


def fit_response_maps(items, members, model):
    """Fit matched value-only and joint maps, retaining every non-coefficient asset."""
    stats = sufficient_statistics(items, members, model)
    hx, hd = stats["hidden"] * stats["rx"], stats["hidden"] * stats["rd"]
    alpha = float(model["alpha"])
    penalty = np.full(len(model["coef"]), alpha)
    penalty[-3] = 0
    out, training = {}, {}
    training["historical_primal"] = losses(model["coef"], stats, alpha)
    for objective in OBJECTIVES:
        eta = int(objective == "joint")
        gram = stats["gx"] / hx + eta * stats["gd"] / hd + np.diag(penalty)
        rhs = stats["cx"] / hx + eta * stats["cd"] / hd
        coef = np.linalg.solve(gram, rhs)
        if not np.isfinite(coef).all():
            raise ValueError("Finite fixed-objective solution required")
        out[objective] = {k: np.array(v, copy=True) for k, v in model.items()} | {
            "coef": coef,
            "source_R_X": np.asarray(stats["rx"]),
            "source_R_D": np.asarray(stats["rd"]),
            "objective": np.asarray(objective),
        }
        training[objective] = losses(coef, stats, alpha) | {
            "normal_equation_max_error": float(np.max(abs(gram @ coef - rhs)))
        }
    return out, training, stats
