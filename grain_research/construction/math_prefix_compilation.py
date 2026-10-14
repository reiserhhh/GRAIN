#!/usr/bin/env python3
"""Source-only explicit functions of query-independent local prefix coordinates."""

from __future__ import annotations

from itertools import combinations_with_replacement

import numpy as np

SPANS = tuple(
    f"{axis}_{part}" for axis in ("A", "T") for part in ("property", "actor", "neutral")
)
NAMES = tuple(f"{span}_u{j}" for span in SPANS for j in range(2))
PROPERTY = (0, 1, 6, 7)
ARMS = (
    "source_constant",
    "property_linear",
    "all_span_linear",
    "property_order_quadratic",
)


def prefix_coordinates(arrays, spans, frame, question_start, layer=16):
    """Read six declared body spans only; query endpoint arrays are never accessed."""
    u = np.asarray(frame, float)
    if u.ndim != 2 or u.shape[1] != 2 or not np.isfinite(u).all():
        raise ValueError("finite two-column source frame required")
    if not np.allclose(u.T @ u, np.eye(2), atol=1e-10):
        raise ValueError("source frame must be orthonormal")
    if set(spans) != set(SPANS):
        raise ValueError("six complete body spans required")
    values = []
    for span in SPANS:
        positions = spans[span]["positions"]
        if not positions or min(positions) < 0 or max(positions) >= question_start:
            raise ValueError("body spans must precede the question")
        key = f"L{layer}_{span}"
        if key not in arrays:
            raise ValueError(f"missing body array: {key}")
        h = np.asarray(arrays[key], float)
        if h.shape != (len(positions), u.shape[0]) or not np.isfinite(h).all():
            raise ValueError("finite token-by-hidden body array required")
        values.extend(h.mean(axis=0) @ u)
    order = int(
        min(spans["T_property"]["positions"]) < min(spans["A_property"]["positions"])
    )
    return np.asarray(values), order


def feature_map(x, order, arm):
    """Construct the four fixed views without owner, polarity, query or family IDs."""
    a, s = np.asarray(x, float), np.asarray(order, float)
    if a.ndim != 2 or a.shape[1] != 12 or s.shape != (len(a),):
        raise ValueError("twelve body coordinates and relation-order bit required")
    if not np.isfinite(a).all() or not np.isin(s, (0, 1)).all():
        raise ValueError("finite coordinates and binary order required")
    if arm == "source_constant":
        return np.empty((len(a), 0)), []
    if arm == "all_span_linear":
        return a.copy(), list(NAMES)
    if arm == "property_linear":
        return a[:, PROPERTY].copy(), [NAMES[j] for j in PROPERTY]
    if arm != "property_order_quadratic":
        raise ValueError("unknown fixed compilation arm")
    base = np.column_stack((a[:, PROPERTY], s))
    names = [NAMES[j] for j in PROPERTY] + ["relation_order"]
    pairs = list(combinations_with_replacement(range(5), 2))
    return (
        np.column_stack((base, *(base[:, i] * base[:, j] for i, j in pairs))),
        names + [f"{names[i]}*{names[j]}" for i, j in pairs],
    )


def compile_source(phi, target, alpha=1.0):
    """Fit unpenalized intercept and fixed Ridge slopes on supplied source rows only."""
    x, y = np.asarray(phi, float), np.asarray(target, float)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) < 2:
        raise ValueError(
            "matched source matrices with at least two observations required"
        )
    if not np.isfinite(x).all() or not np.isfinite(y).all() or alpha <= 0:
        raise ValueError("finite source observations and positive alpha required")
    mean, scale, intercept = x.mean(axis=0), x.std(axis=0), y.mean(axis=0)
    active = scale > np.finfo(float).eps * np.maximum(1, np.max(np.abs(x), axis=0))
    safe_scale = np.where(active, scale, 1)
    z = (x - mean) / safe_scale
    z[:, ~active] = 0
    coefficient = np.linalg.solve(
        z.T @ z + alpha * np.eye(x.shape[1]), z.T @ (y - intercept)
    )
    raw = coefficient / safe_scale[:, None]
    return {
        "mean": mean.tolist(),
        "scale": safe_scale.tolist(),
        "active": active.tolist(),
        "coefficient": coefficient.tolist(),
        "raw_coefficient": raw.tolist(),
        "intercept": intercept.tolist(),
        "raw_intercept": (intercept - mean @ raw).tolist(),
        "alpha": float(alpha),
        "source_rows": len(x),
    }


def apply_function(phi, function):
    """Evaluate a frozen numeric function; never recompute application references."""
    x = np.asarray(phi, float)
    mean, scale = np.asarray(function["mean"]), np.asarray(function["scale"])
    active = np.asarray(function["active"], bool)
    b = np.asarray(function["intercept"])
    coefficient = np.asarray(function["coefficient"], float).reshape(len(mean), len(b))
    if x.ndim != 2 or x.shape[1] != len(mean) or not np.isfinite(x).all():
        raise ValueError("finite complete predictor columns required")
    z = (x - mean) / scale
    z[:, ~active] = 0
    return z @ coefficient + b


def owner_contrasts(metadata, scores):
    """Compute all same-family owner-by-polarity contrasts at both relation orders."""
    y = np.asarray(scores, float)
    if y.ndim != 2 or len(y) != len(metadata) or not np.isfinite(y).all():
        raise ValueError("matched finite outputs required for contrasts")
    index = {}
    for i, row in enumerate(metadata):
        key = row["family"], row["mask"]
        if key in index:
            raise ValueError("duplicate family/cell address")
        index[key] = i
    rows = []
    for family in sorted({row["family"] for row in metadata}):
        roles = {row["role"] for row in metadata if row["family"] == family}
        if len(roles) != 1:
            raise ValueError("one role per lexical family required")
        for axis, offset, polarity, owner in (("A", 2, 1, 4), ("T", 0, 2, 8)):
            for order in (0, 1):
                masks = [
                    offset + 16 * order + p * polarity + o * owner
                    for o, p in ((0, 0), (0, 1), (1, 0), (1, 1))
                ]
                if any((family, mask) not in index for mask in masks):
                    raise ValueError("complete owner-by-polarity corners required")
                corners = y[[index[family, mask] for mask in masks]]
                rows.append(
                    {
                        "family": family,
                        "role": next(iter(roles)),
                        "axis": axis,
                        "order": order,
                        "masks": masks,
                        "values": (
                            corners[1] - corners[0] - corners[3] + corners[2]
                        ).tolist(),
                    }
                )
    return rows


def compare_values(observed, predicted):
    """Report signed action fidelity separately from level correlation; constant r is null."""
    y, p = np.asarray(observed, float), np.asarray(predicted, float)
    if y.ndim != 1 or p.shape != y.shape or not len(y) or not np.isfinite([y, p]).all():
        raise ValueError("matched finite scalar observations required")
    e = p - y
    stable = (y.std() > 1e-12) and (p.std() > 1e-12)
    return {
        "n": len(y),
        "rmse": float(np.sqrt(np.mean(e**2))),
        "bias": float(e.mean()),
        "r": float(np.corrcoef(y, p)[0, 1]) if stable else None,
        "observed_mean": float(y.mean()),
        "predicted_mean": float(p.mean()),
        "sign_agreement": float(np.mean(np.sign(p) == np.sign(y))),
    }
