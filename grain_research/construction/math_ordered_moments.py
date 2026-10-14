"""HM14: position-tagged observations and source-only finite action functions."""

from __future__ import annotations

import numpy as np

from grain_research.construction.math_prefix_compilation import NAMES, SPANS

ARMS = ("zero_action", "span_mean", "ordered_first", "ordered_complete")
PROPERTY = ("A_property", "T_property")
ORDERED_NAMES = tuple(
    f"{span}_position{k}_u{j}" for k in (1, 2, 3) for span in PROPERTY for j in range(2)
)
ALL_NAMES = (*NAMES, *ORDERED_NAMES)


def normalized_positions(tokens):
    """Use a fixed symmetric position grid; one-token spans have no placement axis."""
    if not isinstance(tokens, (int, np.integer)) or tokens < 1:
        raise ValueError("positive integer token count required")
    return np.zeros(1) if tokens == 1 else np.linspace(-1, 1, tokens)


def ordered_moments(matrix, degree=3):
    """Compute all-dimensional position moments on a declared finite token span."""
    h = np.asarray(matrix, float)
    if h.ndim != 2 or min(h.shape) < 1 or not np.isfinite(h).all():
        raise ValueError("finite nonempty token matrix required")
    if not isinstance(degree, int) or degree < 0:
        raise ValueError("nonnegative integer moment degree required")
    p = normalized_positions(len(h))
    return np.asarray([(p**k) @ h / len(h) for k in range(degree + 1)])


def reconstruct_moments(moments, tokens):
    """Invert the full fixed-length moment encoding, not an underdetermined truncation."""
    m = np.asarray(moments, float)
    if m.ndim != 2 or m.shape[0] != tokens or not np.isfinite(m).all():
        raise ValueError("one full moment per token and known length required")
    p = normalized_positions(tokens)
    v = np.vander(p, N=tokens, increasing=True).T
    return np.linalg.solve(v, tokens * m)


def observation(bodies, frame):
    """Observe six means and twelve ordered property coordinates without query/label IDs."""
    u = np.asarray(frame, float)
    if (
        u.ndim != 2
        or u.shape[1] != 2
        or not np.isfinite(u).all()
        or not np.allclose(u.T @ u, np.eye(2), atol=1e-10)
        or set(bodies) != set(SPANS)
    ):
        raise ValueError(
            "complete six-span states and frozen rank2 orthonormal frame required"
        )
    projected = {}
    means = []
    for name in SPANS:
        h = np.asarray(bodies[name], float)
        if (
            h.ndim != 2
            or h.shape[1] != len(u)
            or not len(h)
            or not np.isfinite(h).all()
        ):
            raise ValueError("finite complete token-by-hidden state required")
        moments = ordered_moments(h, 3)
        projected[name] = moments @ u
        means.extend(projected[name][0])
    extra = [
        projected[name][k, j] for k in (1, 2, 3) for name in PROPERTY for j in range(2)
    ]
    return np.asarray([*means, *extra])


def feature_indices(arm):
    """Return the four fixed views; no degree or rank selection from held outcomes."""
    if arm not in ARMS:
        raise ValueError("unknown fixed observation arm")
    count = {
        "zero_action": 0,
        "span_mean": 12,
        "ordered_first": 16,
        "ordered_complete": 24,
    }[arm]
    return np.arange(count)


def family_weights(families):
    """Give each source lexical family equal loss mass, preserving mean row weight1."""
    a = np.asarray(families)
    if a.ndim != 1 or not len(a):
        raise ValueError("nonempty source family addresses required")
    unique, counts = np.unique(a, return_counts=True)
    count_map = dict(zip(unique.tolist(), counts.tolist(), strict=True))
    return np.asarray([len(a) / (len(unique) * count_map[f]) for f in a])


def compile_action(delta, target, families, alpha=1.0):
    """Fit a fixed weighted Ridge action law with source RMS scales and no intercept."""
    x, y = np.asarray(delta, float), np.asarray(target, float)
    if (
        x.ndim != 2
        or y.shape != (len(x),)
        or len(x) < 2
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or not np.isfinite(alpha)
        or alpha <= 0
    ):
        raise ValueError(
            "finite matched source action data and positive alpha required"
        )
    weights = family_weights(families)
    if len(weights) != len(x):
        raise ValueError("source family count mismatch")
    scale = np.sqrt(np.sum(weights[:, None] * x**2, axis=0) / np.sum(weights))
    threshold = (
        100 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(x), initial=0)))
    )
    active = scale > threshold
    safe = np.where(active, scale, 1.0)
    z = x / safe
    z[:, ~active] = 0
    coefficient = np.linalg.solve(
        z.T @ (weights[:, None] * z) + alpha * np.eye(x.shape[1]),
        z.T @ (weights * y),
    )
    return {
        "scale": safe.tolist(),
        "active": active.tolist(),
        "coefficient": coefficient.tolist(),
        "raw_coefficient": (coefficient / safe).tolist(),
        "intercept": 0.0,
        "centering": False,
        "alpha": float(alpha),
        "source_rows": len(x),
        "source_families": len(np.unique(families)),
        "roundoff_active_threshold": threshold,
        "source_weight_sum": float(weights.sum()),
    }


def apply_action(delta, function):
    """Apply source-frozen numeric coefficients without application recalibration."""
    x = np.asarray(delta, float)
    b = np.asarray(function["raw_coefficient"], float)
    if x.ndim != 2 or x.shape[1] != len(b) or not np.isfinite(x).all():
        raise ValueError("finite matched frozen observation differences required")
    if function["intercept"] != 0 or function["centering"]:
        raise ValueError("action must preserve identical-input zero")
    return x @ b
