#!/usr/bin/env python3
"""Complete evidence-cube operators; descriptive algebra, not precision search."""

from __future__ import annotations

import itertools

import numpy as np

from .math_cross_target_operator import TARGETS
from .math_passive_event_composition import action_line, record


def grid(n=6):
    """Return the declared independent low/high corners in lexicographic order."""
    if not isinstance(n, int) or n < 1:
        raise ValueError("Positive integer dimension required")
    return np.asarray(list(itertools.product((-1.0, 1.0), repeat=n)))


def masks(n=6):
    """Index every interaction, including the constant, without selection."""
    return np.asarray(list(itertools.product((0, 1), repeat=n)), dtype=int)


def basis(points, n=6):
    """Evaluate multiaffine monomials, including points inside the cube."""
    x = np.asarray(points, dtype=float)
    if x.ndim != 2 or x.shape[1] != n or not np.isfinite(x).all():
        raise ValueError("Finite point matrix required")
    return np.prod(np.where(masks(n)[None, :, :] == 1, x[:, None, :], 1), axis=2)


def coefficients(values, n=6):
    """Compute the full orthogonal coefficients of an exactly complete cube."""
    v = np.asarray(values, dtype=float)
    if v.shape[0] != 2**n or not np.isfinite(v).all():
        raise ValueError("Complete finite cube required")
    return (basis(grid(n), n).T @ v.reshape(2**n, -1) / 2**n).reshape(v.shape)


def evaluate(coef, points=None, order=None, n=6):
    """Evaluate a fixed coefficient table, optionally truncating by declared order."""
    c = np.asarray(coef, dtype=float).copy()
    if c.shape[0] != 2**n or not np.isfinite(c).all():
        raise ValueError("Complete finite coefficient table required")
    if order is not None:
        if not isinstance(order, int) or not 0 <= order <= n:
            raise ValueError("Declared interaction order required")
        c[masks(n).sum(axis=1) > order] = 0
    x = grid(n) if points is None else points
    return (basis(x, n) @ c.reshape(2**n, -1)).reshape((len(x), *c.shape[1:]))


def order_energy(coef, n=6):
    """Report squared coefficient energy separately from the constant level."""
    c = np.asarray(coef, dtype=float).reshape(2**n, -1)
    orders = masks(n).sum(axis=1)
    energy = [float(np.square(c[orders == k]).sum()) for k in range(n + 1)]
    variation = sum(energy[1:])
    return {
        "energy": energy,
        "nonconstant_energy": variation,
        "interaction_fraction": sum(energy[2:]) / variation
        if variation > 1e-24
        else None,
    }


def cosine(a, b):
    """Compare same-shaped finite response arrays without a square-matrix assumption."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Matched finite response arrays required")
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.sum(a * b) / denominator) if denominator > 1e-24 else None


def decode(logits):
    """Decode actual or counterfactual ten-candidate logits with the same link."""
    z = np.asarray(logits, dtype=float)
    if z.shape[-1] != 10 or not np.isfinite(z).all():
        raise ValueError("Ten finite candidate logits required")
    shifted = z - z.max(axis=-1, keepdims=True)
    p = np.exp(shifted)
    p /= p.sum(axis=-1, keepdims=True)
    return {
        "score": p @ (np.arange(10) / 9),
        "log_odds": np.logaddexp.reduce(z[..., 5:], axis=-1)
        - np.logaddexp.reduce(z[..., :5], axis=-1),
    }


def edges(values, n=6):
    """List all high-minus-low conditional differences for every input bit."""
    v = np.asarray(values, dtype=float)
    if v.shape[0] != 2**n or not np.isfinite(v).all():
        raise ValueError("Complete finite cube required")
    x = grid(n)
    return np.asarray(
        [
            [v[k + 2 ** (n - 1 - j)] - v[k] for k in np.flatnonzero(x[:, j] < 0)]
            for j in range(n)
        ]
    )


def jacobian(coef, points, n=6):
    """Differentiate the explicitly declared multiaffine interpolant."""
    c, x = np.asarray(coef, dtype=float), np.asarray(points, dtype=float)
    if c.shape != (2**n, n):
        raise ValueError("Square response map required")
    out = np.zeros((len(x), n, n))
    mm = masks(n)
    for j in range(n):
        selected = mm[:, j] == 1
        remaining = mm[selected].copy()
        remaining[:, j] = 0
        b = np.prod(np.where(remaining[None] == 1, x[:, None, :], 1), axis=2)
        out[:, :, j] = b @ c[selected]
    return out


def injectivity_diagnostic(coef, source_linear, n=6):
    """Give fixed sufficient certificates; failure never implies noninjectivity."""
    j = jacobian(coef, grid(n), n)
    sym = (j + j.transpose(0, 2, 1)) / 2
    lower = float(np.linalg.eigvalsh(sym).min())
    a = np.asarray(source_linear, dtype=float)
    if a.shape != (n, n) or not np.isfinite(a).all():
        raise ValueError("Frozen finite source frame required")
    singular = np.linalg.svd(a, compute_uv=False)
    invertible = singular[-1] > 1e-12
    result = {
        "symmetric_corner_min_eigenvalue": lower,
        "strong_monotonicity_certificate": lower > 0,
        "source_frame_condition": float(singular[0] / singular[-1])
        if invertible
        else None,
        "source_frame_kappa": None,
        "source_frame_injectivity_certificate": False,
        "interpolant_only": True,
        "failure_is_not_noninjectivity": True,
    }
    if invertible:
        transformed = np.asarray([np.linalg.solve(a, v) for v in j])
        kappa = float(np.linalg.norm(transformed - np.eye(n), ord=2, axis=(1, 2)).max())
        result.update(
            source_frame_kappa=kappa,
            source_frame_injectivity_certificate=kappa < 1,
        )
    return result


def linear_operator(coef, n=6):
    """Convert mean first-order coefficients into high-minus-low sensitivities."""
    c = np.asarray(coef, dtype=float)
    mm = masks(n)
    return 2 * np.column_stack(
        [c[np.flatnonzero(mm[:, j] & (mm.sum(1) == 1))[0]] for j in range(n)]
    )


def documents(material):
    """Construct qualified six-event accounts with independent behavior alternatives."""
    if material["targets"] != list(TARGETS):
        raise ValueError("Unchanged six-target order required")
    docs = []
    for family, (corner, x) in itertools.product((0, 1), enumerate(grid())):
        records, lines = [], []
        for j, target in enumerate(TARGETS):
            polarity = int(x[j] > 0)
            event = f"K{j + 1}"
            records.append(record(f"R{j + 1}", event, family, polarity))
            lines.append(
                action_line(
                    event, "A", material["behaviours"][target][family][polarity]
                )
            )
        docs.append(
            {
                "doc_id": len(docs),
                "family": family,
                "role": "source" if family == 0 else "application",
                "corner": corner,
                "x": x.tolist(),
                "body": "\n".join(lines),
                "records": records,
            }
        )
    return docs
