#!/usr/bin/env python3
"""Finite source-linked relational fields, without predictor or neural fitting."""

import numpy as np

from grain_research.construction.math_joint_target_atlas import characters

COMPONENTS = ("joint", "profile", "pair")
SPACES = ("full", "RR", "RC", "CR", "CC")
TARGET_MODES = np.array([0, 1, 3, 7, 15])


def compile_bank(means, codes):
    """Keep all31 profile characters and the exact within-profile pairing term."""
    x, g = np.asarray(means, float), np.asarray(codes)
    if x.ndim != 3 or x.shape[1] != 2 or g.shape != (len(x),):
        raise ValueError("Two matched source windows and profile codes required")
    if not np.isfinite(x).all() or not np.isin(g, np.arange(32)).all():
        raise ValueError("Finite vectors and full inherited profile codes required")
    n = np.bincount(g.astype(int), minlength=32)
    if n.min() < 1:
        raise ValueError("No missing-profile smoothing")
    cells = np.array([x[g == k].mean(0) for k in range(32)])
    centers = cells.mean(0)
    h = characters()[:, 1:]
    weights = (h[g.astype(int)] / (32 * n[g.astype(int), None])).T
    a, b = (x[:, v] - centers[v] for v in (0, 1))
    pa, pb = (cells[:, v] - centers[v] for v in (0, 1))
    return {
        "centers": centers,
        "joint_a": a,
        "joint_b": b,
        "joint_w": weights,
        "profile_a": pa,
        "profile_b": pb,
        "profile_w": h.T / 32,
        "pair_a": a - pa[g.astype(int)],
        "pair_b": b - pb[g.astype(int)],
        "pair_w": weights,
        "codes": g.astype(int),
    }


def sides(bank, component, space, basis):
    """Project each leg separately; retain both oriented cross-space channels."""
    if component not in COMPONENTS or space not in SPACES:
        raise ValueError("Unknown operator account")
    a, b, w = (bank[f"{component}_{k}"] for k in ("a", "b", "w"))
    if space != "full":
        ar, br = (a @ basis) @ basis.T, (b @ basis) @ basis.T
        a = ar if space[0] == "R" else a - ar
        b = br if space[1] == "R" else b - br
    return a, b, w


def contraction(left, right, component0, component1, space, basis):
    """Compute signed HS accounts by dual Gram products, not dense tensor fits."""
    a, b, w = sides(left, component0, space, basis)
    c, d, v = sides(right, component1, space, basis)
    kernel = (a @ c.T) * (b @ d.T)
    return np.sum((w @ kernel) * v, axis=1)


def read(bank, component, space, left, right, basis):
    """Score matched numeric windows with a fixed source reference and operator."""
    x, y = np.asarray(left, float), np.asarray(right, float)
    if x.shape != y.shape or x.shape[-1] != bank["centers"].shape[-1]:
        raise ValueError("Matched vector windows required")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Finite cold windows required")
    a, b, w = sides(bank, component, space, basis)
    xx, yy = x - bank["centers"][0], y - bank["centers"][1]
    return ((xx @ a.T) * (yy @ b.T)) @ w.T


def field(bank, component, space, companion, view, basis):
    """Return all31 companion-conditioned covectors for one event window."""
    if view not in (0, 1):
        raise ValueError("Window0 or1 required")
    a, b, w = sides(bank, component, space, basis)
    y = np.asarray(companion, float) - bank["centers"][1 - view]
    own, other = (a, b) if view == 0 else (b, a)
    return (w * (other @ y)[None, :]) @ own


def repaire(bank, rng):
    """Exchange the second window within full profiles, keeping source means."""
    ix = np.arange(len(bank["codes"]))
    for g in range(32):
        ids = np.flatnonzero(bank["codes"] == g)
        ix[ids] = rng.permutation(ids)
    out = dict(bank)
    for component in ("joint", "pair"):
        out[f"{component}_b"] = bank[f"{component}_b"][ix]
    return out, ix


def cold(bank, basis, record):
    """Read two addressed event sets; reject labels, duplicates and empty sets."""
    if set(record) != {"frame_key", "windows"} or len(record["windows"]) != 2:
        raise ValueError("Only two addressed numeric windows permitted")
    seen, means = set(), []
    d = bank["centers"].shape[-1]
    for events in record["windows"]:
        if not events:
            raise ValueError("Missing window is unavailable, not a zero companion")
        vectors = []
        for event in events:
            if set(event) != {"event_id", "vector"}:
                raise ValueError("No targets, answers or author features at inference")
            uid = event["event_id"]
            if not isinstance(uid, str) or not uid or uid in seen:
                raise ValueError("Distinct identified events required")
            seen.add(uid)
            v = np.asarray(event["vector"], float)
            if v.shape != (d,) or not np.isfinite(v).all():
                raise ValueError("Finite registered vector dimension required")
            vectors.append(v)
        means.append(np.mean(vectors, axis=0))
    values = np.stack([read(bank, c, "full", *means, basis) for c in COMPONENTS])
    return {"frame_key": record["frame_key"], "operator_readings": values.tolist()}
