#!/usr/bin/env python3
"""Source-calibrated joint observation geometry, not personality truth."""

from __future__ import annotations

import numpy as np

ARMS = (
    "G_location",
    "Q_location",
    "joint_location",
    "coupling",
    "shifted_coupling",
    "joint_with_coupling",
    "joint_with_shifted",
)
METRICS = ("all_strangers", "location_matched_strangers")
TIE_TOLERANCE = 1e-12


def panel_statistics(grain, queries):
    """Preserve marginal locations/covariances and exact comment co-location."""
    g, y = np.asarray(grain, float), np.asarray(queries, float)
    if (
        g.ndim != 2
        or y.ndim != 2
        or len(g) != len(y)
        or min(g.shape + y.shape) < 2
        or not np.isfinite(g).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError("Aligned finite multi-coordinate comments required")
    mg, my = g.mean(0), y.mean(0)
    gc, yc = g - mg, y - my
    return {
        "mu_g": mg,
        "mu_y": my,
        "second_g": g.T @ g / len(g),
        "second_y": y.T @ y / len(y),
        "cross": gc.T @ yc / len(g),
        "shifted": gc.T @ np.roll(yc, len(g) // 2, axis=0) / len(g),
        "n": len(g),
    }


def stack_statistics(panels):
    """Stack equally weighted panels without pooling unequal comment counts."""
    if not panels:
        raise ValueError("At least one panel required")
    return {key: np.stack([p[key] for p in panels]) for key in panels[0]}


def inverse_root(covariance):
    """Reject singular reference units; never tune a ridge or truncate rank."""
    cov = np.asarray(covariance, float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or not np.isfinite(cov).all():
        raise ValueError("Finite square reference covariance required")
    eigen, vectors = np.linalg.eigh((cov + cov.T) / 2)
    tol = 100 * np.finfo(float).eps * len(cov) * max(float(eigen[-1]), 0)
    if eigen[0] <= tol:
        raise ValueError("Degenerate source covariance; no regularizer rescue")
    root = (vectors / np.sqrt(eigen)) @ vectors.T
    return root, eigen, float(tol)


def _rms(values):
    """Use a neutral unit for an exactly constant source block, explicitly."""
    rms = float(np.sqrt(np.mean(np.sum(np.asarray(values) ** 2, axis=1))))
    return rms if rms > 1e-14 else 1.0, rms <= 1e-14


def fit_reference(source):
    """Freeze marginal units/profile centers using reference panels only."""
    mg, my = source["mu_g"].mean(0), source["mu_y"].mean(0)
    sg = source["second_g"].mean(0) - np.outer(mg, mg)
    sy = source["second_y"].mean(0) - np.outer(my, my)
    wg, eg, tg = inverse_root(sg)
    wy, ey, ty = inverse_root(sy)
    ref = {
        "mu_g": mg,
        "mu_y": my,
        "cov_g": sg,
        "cov_y": sy,
        "root_g": wg,
        "root_y": wy,
        "eigen_g": eg,
        "eigen_y": ey,
        "machine_tolerance": np.array([tg, ty]),
        "cross": source["cross"].mean(0),
        "shifted": source["shifted"].mean(0),
    }
    blocks = _blocks(source, ref)
    units = [_rms(b) for b in blocks]
    ref["scales"] = np.array([u[0] for u in units])
    ref["neutral_block"] = np.array([u[1] for u in units])
    return ref


def _blocks(statistics, reference):
    """Represent relative levels and both paired/shifted rectangular tensors."""
    s, r = statistics, reference
    result = [
        (s["mu_g"] - r["mu_g"]) @ r["root_g"],
        (s["mu_y"] - r["mu_y"]) @ r["root_y"],
    ]
    for key in ("cross", "shifted"):
        whitened = np.einsum(
            "ij,njk,kl->nil", r["root_g"], s[key] - r[key], r["root_y"]
        )
        result.append(whitened.reshape(len(whitened), -1))
    return result


def profiles(statistics, reference):
    """Return all seven fixed arms, preserving source-scaled mean locations."""
    g, q, c, shifted = [
        b / scale
        for b, scale in zip(
            _blocks(statistics, reference), reference["scales"], strict=True
        )
    ]
    return {
        "G_location": g,
        "Q_location": q,
        "joint_location": np.column_stack((g, q)) / np.sqrt(2),
        "coupling": c,
        "shifted_coupling": shifted,
        "joint_with_coupling": np.column_stack((g, q, c)) / np.sqrt(3),
        "joint_with_shifted": np.column_stack((g, q, shifted)) / np.sqrt(3),
    }


def squared_distances(left, right):
    """Use direct differences rather than cancellation-prone norm expansion."""
    a, b = np.asarray(left, float), np.asarray(right, float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("Common finite profile coordinates required")
    return np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)


def trace_distance(difference, covariance_g, covariance_y):
    """Compute the coordinate-invariant tensor norm without whitening roots."""
    e = np.asarray(difference, float)
    left = np.linalg.solve(covariance_g, e)
    right = np.linalg.solve(covariance_y, e.T)
    return float(np.trace(left @ right))


def candidate_wins(distances, location_distances, hard_count=3):
    """Evaluate each possible nominated identity, excluding it from negatives."""
    d, locations = np.asarray(distances, float), np.asarray(location_distances, float)
    if (
        d.ndim != 2
        or d.shape[0] != d.shape[1]
        or d.shape != locations.shape
        or len(d) <= hard_count
        or not np.isfinite(d).all()
        or not np.isfinite(locations).all()
    ):
        raise ValueError("At least four aligned finite held authors required")
    n = len(d)
    all_wins, hard_wins = np.zeros((n, n)), np.zeros((n, n))
    nearest = np.argsort(locations, axis=1, kind="stable")
    for i in range(n):
        for j in range(n):
            negatives = np.flatnonzero(np.arange(n) != j)
            hard = nearest[i][nearest[i] != j][:hard_count]
            for target, indexes in ((all_wins, negatives), (hard_wins, hard)):
                delta = d[i, indexes] - d[i, j]
                target[i, j] = np.mean(
                    (delta > TIE_TOLERANCE) + 0.5 * (abs(delta) <= TIE_TOLERANCE)
                )
    return all_wins, hard_wins


def bootstrap_summary(author_values, seed=20261009, repetitions=2000):
    """Resample authors after averaging their dependent six panel comparisons."""
    x = np.asarray(author_values, float)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("Finite author-by-endpoint values required")
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(x), size=(repetitions, len(x)))
    means = x[indexes].mean(1)
    return x.mean(0), np.quantile(means, [0.025, 0.975], axis=0), means
