#!/usr/bin/env python3
"""Source-calibrated relational coordinates and declared author expectations.

Scoring accepts numeric vectors only. Calibration design labels never enter
the application scorer. Coordinates are instrument directions, not traits.
"""

from __future__ import annotations

import itertools

import numpy as np


def fit_frame(source, columns):
    """Construct one full-column-rank source dual without target responses."""
    x, d = np.asarray(source, float), np.asarray(columns, float)
    if x.ndim != 2 or d.ndim != 2 or x.shape[1] != d.shape[0]:
        raise ValueError("Source observations and action columns must share dimensions")
    if not np.isfinite(x).all() or not np.isfinite(d).all() or not len(x):
        raise ValueError("Finite nonempty calibration required")
    u, s, vt = np.linalg.svd(d, full_matrices=False)
    cutoff = np.finfo(float).eps * max(d.shape) * s[0]
    if len(s) != d.shape[1] or np.count_nonzero(s > cutoff) != d.shape[1]:
        raise ValueError("Rank-deficient frame cannot declare independent coordinates")
    dual = (vt.T / s) @ u.T
    return {
        "center": x.mean(0),
        "columns": d,
        "dual": dual,
        "singular": s,
        "inverse_norm": np.asarray(1 / s[-1]),
        "condition": np.asarray(s[0] / s[-1]),
    }


def score_vectors(values, frame):
    """Score a numeric batch; no family, role, condition or target-answer input."""
    x = np.asarray(values, float)
    if x.ndim != 2 or x.shape[1] != len(frame["center"]):
        raise ValueError("Frozen vector layout required")
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite observations are not zero-filled")
    delta = x - frame["center"]
    coordinates = delta @ frame["dual"].T
    complement = delta - coordinates @ frame["columns"].T
    return {"coordinates": coordinates, "complement": complement}


def make_profile_laws(rows, theta, cells):
    """Generate fixed-margin synthetic event laws for testing author assembly.

    Theta is generation truth only. The vector scorer never receives these rows
    or values. Conditions outside the bank have no events, not imputed responses.
    """
    t = np.asarray(theta, float)
    if t.ndim != 3 or t.shape[1:] != (6, 2) or not np.isfinite(t).all():
        raise ValueError("Profiles require finite six-by-two design parameters")
    if np.max(abs(t)) > 1:
        raise ValueError("Probabilities require parameters in [-1,1]")
    cells = tuple(cells)
    if not cells or len(set(cells)) != len(cells):
        raise ValueError("Unique supported cells required")
    keys = [
        (
            r["wording"],
            r["family"],
            (r["condition"] + 1) // 2,
            r["polarity"],
            r["owner"],
        )
        for r in rows
    ]
    wordings = sorted({k[0] for k in keys})
    expected = set(itertools.product(wordings, cells, (-1, 1), (-1, 1)))
    flattened = [(w, (f, c), b, o) for w, f, c, b, o in keys]
    if len(set(flattened)) != len(rows) or set(flattened) != expected:
        raise ValueError("Complete unique four-corner banks required")
    p = np.stack([1 + b * o * t[:, f, c] for _, f, c, b, o in keys], axis=1) / (
        4 * len(wordings) * len(cells)
    )
    if np.min(p) < 0 or not np.allclose(p.sum(1), 1, atol=1e-12, rtol=0):
        raise ValueError("Invalid declared probability laws")
    return p


def compose_profiles(laws, scores):
    """Integrate declared event probabilities, never a neural partition mass."""
    p = np.asarray(laws, float)
    if p.ndim != 2 or not np.isfinite(p).all() or np.min(p) < 0:
        raise ValueError("Finite nonnegative author laws required")
    if not np.allclose(p.sum(1), 1, atol=1e-12, rtol=0):
        raise ValueError("Author law must normalize, not be silently renormalized")
    out = {}
    for key, value in scores.items():
        a = np.asarray(value, float)
        if a.ndim != 2 or a.shape[0] != p.shape[1] or not np.isfinite(a).all():
            raise ValueError("Scores must address the exact finite evidence bank")
        out[key] = p @ a
    return out


def action_columns(values, rows, cells):
    """Compute evaluation factorial columns; never refit the application dual."""
    x = np.asarray(values, float)
    columns = []
    for f, c in cells:
        ids = [
            i
            for i, r in enumerate(rows)
            if r["family"] == f and (r["condition"] + 1) // 2 == c
        ]
        if len(ids) != 8:
            raise ValueError("Two wordings and all behavior/actor corners required")
        columns.append(
            sum(rows[i]["polarity"] * rows[i]["owner"] * x[i] for i in ids) / 8
        )
    return np.stack(columns, axis=1)


def lexical_law_distance(observations, laws):
    """Measure exact coordinate-class distribution changes, not mean changes."""
    g, p = np.asarray(observations, float), np.asarray(laws, float)
    if g.ndim != 2 or p.ndim != 2 or p.shape[1] != len(g):
        raise ValueError("Lexical rows and profile weights must align")
    classes = {}
    for i, row in enumerate(g):
        classes.setdefault(tuple(row), []).append(i)
    mass = np.stack([p[:, ids].sum(1) for ids in classes.values()], axis=1)
    return float(np.max(abs(mass - mass[0]))), mass
