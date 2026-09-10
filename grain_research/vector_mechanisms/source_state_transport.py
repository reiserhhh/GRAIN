#!/usr/bin/env python3
"""Source-only differentiable state transport with coupled primal and tangent."""

from __future__ import annotations

import hashlib

import numpy as np
from sklearn.utils.extmath import randomized_svd

from .compiled_query_transfer import QUERIES, validate_sources
from .native_material_reader import direction

ARMS = ("identity", "pointwise", "causal_context")


def source_set(cohort, index):
    """Return the unchanged source4 leave-one-out/new8 all-source contract."""
    members = [i for i in range(4) if cohort != "source4" or i != index]
    validate_sources(cohort, index, members)
    return members


def row_weights(items):
    """Give equal authors, equal queries per author, equal tokens per material."""
    slots = [(int(i["index"]), i["query"]) for i in items]
    if not slots or len(set(slots)) != len(slots):
        raise ValueError("Distinct author-query materials required")
    authors = sorted({i for i, _ in slots})
    rows = []
    for item in items:
        x = np.asarray(item["x"])
        if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
            raise ValueError("Finite token-by-hidden source material required")
        queries = sum(i == item["index"] for i, _ in slots)
        rows.append(np.full(len(x), 1 / (len(authors) * queries * len(x))))
    return np.concatenate(rows)


def features(x, d, mean, basis, mode):
    """Use fixed layout and only preceding/current tokens; differentiate the same map."""
    x, mean, basis = (np.asarray(v, float) for v in (x, mean, basis))
    if (
        mode not in ARMS[1:]
        or x.ndim != 2
        or len(x) < 2
        or x.shape[1] != len(mean)
        or basis.shape[0] != len(mean)
    ):
        raise ValueError(
            "Aligned source basis, current sequence and known map required"
        )
    p = (x - mean) @ basis
    s = np.linspace(0, 1, len(x))
    parts = [p]
    if mode == "causal_context":
        parts.append(np.cumsum(p, axis=0) / np.arange(1, len(x) + 1)[:, None])
    phi = np.column_stack(parts + [np.ones(len(x)), s, s * s])
    if d is None:
        return phi, None
    d = np.asarray(d, float)
    if d.ndim != 3 or d.shape[1:] != x.shape or not np.isfinite(d).all():
        raise ValueError("Aligned finite full-sequence directions required")
    dp = d @ basis
    tangent_parts = [dp]
    if mode == "causal_context":
        tangent_parts.append(
            np.cumsum(dp, axis=1) / np.arange(1, len(x) + 1)[None, :, None]
        )
    dphi = np.concatenate(tangent_parts + [np.zeros((len(d), len(x), 3))], axis=2)
    return phi, dphi


def fit_maps(items, members, rank=32, alpha=0.01, seed=52):
    """Fit two fixed maps on permitted primal states only, sharing a weighted PCA."""
    selected = [i for i in items if i["index"] in members]
    slots = {(i["index"], i["query"]) for i in selected}
    if len(selected) != 2 * len(members) or slots != {
        (i, q) for i in members for q in QUERIES
    }:
        raise ValueError("Exact permitted source-query pairs required")
    w = row_weights(selected)
    x = np.concatenate([i["x"] for i in selected]).astype(float)
    y = np.concatenate([i["y"] for i in selected]).astype(float)
    if (
        y.shape != x.shape
        or not np.isfinite(y).all()
        or not 0 < rank < min(x.shape)
        or alpha <= 0
    ):
        raise ValueError(
            "Aligned finite source targets and supported fixed rank required"
        )
    mean = w @ x
    weighted = (x - mean) * np.sqrt(w[:, None])
    _, singular, vt = randomized_svd(
        weighted, n_components=rank, n_iter=3, random_state=seed
    )
    if singular[-1] <= max(weighted.shape) * np.finfo(float).eps * singular[0]:
        raise ValueError("Source rank does not support the fixed PCA dimension")
    basis = vt.T
    out = {}
    for mode in ARMS[1:]:
        phi = np.concatenate(
            [features(i["x"], None, mean, basis, mode)[0] for i in selected]
        )
        bias = phi.shape[1] - 3
        center = w @ phi
        scale = np.sqrt(w @ ((phi - center) ** 2))
        scale[scale < 1e-12] = 1
        center[bias], scale[bias] = 0, 1
        z = (phi - center) / scale
        penalty = np.full(z.shape[1], alpha)
        penalty[bias] = 0
        gram = z.T @ (w[:, None] * z) + np.diag(penalty)
        coef = np.linalg.solve(gram, z.T @ (w[:, None] * (y - x)))
        out[mode] = {
            "mean": mean,
            "basis": basis,
            "center": center,
            "scale": scale,
            "coef": coef,
            "members": np.asarray(members, int),
            "singular": singular,
            "alpha": np.asarray(alpha),
            "rank": np.asarray(rank),
            "mode": np.asarray(mode),
        }
    return out


def apply_map(x, d, model):
    """Return full state/tangent and their lossless factorized increments."""
    mode = str(model["mode"])
    phi, dphi = features(x, d, model["mean"], model["basis"], mode)
    z = (phi - model["center"]) / model["scale"]
    dz = dphi / model["scale"]
    return x + z @ model["coef"], d + dz @ model["coef"], z, dz


def body_directions(data):
    """Match the original H16 BODY relative-RMS directions and float32 units."""
    x, pos, grain = data["state"], data["positions"], data["grain"]
    if (
        x.ndim != 2
        or grain.shape != (len(pos), 4)
        or pos.ndim != 1
        or not np.issubdtype(pos.dtype, np.integer)
        or len(pos) < 2
        or np.any(np.diff(pos) <= 0)
        or pos.min() < 0
        or pos.max() >= len(x) - 1
    ):
        raise ValueError(
            "Exact BODY addresses, four GRAIN factors and H16 schema required"
        )
    out = np.zeros((4, *x.shape), np.float32)
    for f in range(4):
        d, _ = direction(x[pos].astype(float), grain, f)
        out[f, pos] = d
    return out.astype(float)


def array_hash(a):
    """Hash shape, dtype and exact contiguous values for the factorized state freeze."""
    a = np.ascontiguousarray(a)
    h = hashlib.sha256(str((a.shape, a.dtype.str)).encode())
    h.update(a.tobytes())
    return h.hexdigest()
