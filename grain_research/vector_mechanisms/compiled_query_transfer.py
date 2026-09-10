#!/usr/bin/env python3
"""Query-reader borrowing and exact bilinear contrast accounting, without a model."""

import numpy as np

QUERIES = ("neuroticism", "confirmation_seeking")
MODES = ("own", "pooled")


def validate_sources(cohort, index, sources):
    """Reject duplicate, absent, or current-author source members in a reader slot."""
    if cohort not in ("source4", "new8") or index not in range(
        4 if cohort == "source4" else 8
    ):
        raise ValueError("Unknown frozen author slot")
    expected = set(range(4)) - ({index} if cohort == "source4" else set())
    if len(sources) != len(expected) or set(sources) != expected:
        raise ValueError("Reader source exclusion differs")


def borrow(readers, directions):
    """Return [recipient, donor, mode, factor] actions with a fixed recipient input."""
    w, d = np.asarray(readers, float), np.asarray(directions, float)
    if (
        w.ndim != 3
        or d.ndim != 3
        or w.shape[:2] != (2, 2)
        or d.shape[:2] != (2, 4)
        or w.shape[-1] != d.shape[-1]
        or not np.isfinite(w).all()
        or not np.isfinite(d).all()
    ):
        raise ValueError(
            "Two queries, two modes, four finite aligned directions required"
        )
    return np.einsum("rfh,pmh->rpmf", d, w)


def contrasts(readers, directions):
    """Split the N-minus-confirmation diagonal contrast symmetrically, not causally."""
    w, d = np.asarray(readers, float), np.asarray(directions, float)
    a = borrow(w, d)
    reader = np.einsum("mh,fh->mf", w[0] - w[1], (d[0] + d[1]) / 2)
    transport = np.einsum("mh,fh->mf", (w[0] + w[1]) / 2, d[0] - d[1])
    return {"contrast": a[0, 0] - a[1, 1], "reader": reader, "transport": transport}


def cosine(a, b):
    """Return an angle diagnostic, leaving zero-vector angles undefined."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / den) if den else float("nan")


def fidelity(actual, predicted):
    """Measure descriptive action fidelity; factors are not independent participants."""
    y, p = np.asarray(actual, float), np.asarray(predicted, float)
    if y.ndim != 1 or y.shape != p.shape or len(y) < 3 or not np.isfinite([y, p]).all():
        raise ValueError("Matched finite one-dimensional actions required")
    centered_y, centered_p = y - y.mean(), p - p.mean()
    energy = np.dot(y, y)
    return {
        "action_r": cosine(centered_y, centered_p),
        "action_mse_over_zero": float(np.square(y - p).sum() / energy)
        if energy
        else float("nan"),
        "action_rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        "actual_action_rms": float(np.sqrt(np.mean(y * y))),
        "predicted_action_rms": float(np.sqrt(np.mean(p * p))),
        "same_sign": int(np.sum(y * p > 0)),
        "nonzero_targets": int(np.count_nonzero(y)),
        "actions": len(y),
    }


def contrast_accounting(reader, transport):
    """Keep means, population variances, covariance and signed uncentered energy."""
    r, t = np.asarray(reader, float), np.asarray(transport, float)
    if r.ndim != 1 or r.shape != t.shape or not len(r) or not np.isfinite([r, t]).all():
        raise ValueError("Finite matched contrast components required")
    c = r + t
    energy = np.mean(c * c)
    cov = np.mean((r - r.mean()) * (t - t.mean()))
    result = {
        "reader_mean": float(r.mean()),
        "transport_mean": float(t.mean()),
        "contrast_mean": float(c.mean()),
        "reader_variance": float(r.var()),
        "transport_variance": float(t.var()),
        "contrast_variance": float(c.var()),
        "covariance": float(cov),
        "reader_rms": float(np.sqrt(np.mean(r * r))),
        "transport_rms": float(np.sqrt(np.mean(t * t))),
        "contrast_rms": float(np.sqrt(energy)),
        "reader_projected_share": float(np.mean(r * c) / energy)
        if energy
        else float("nan"),
        "transport_projected_share": float(np.mean(t * c) / energy)
        if energy
        else float("nan"),
        "signed_cross_energy": float(2 * np.mean(r * t) / energy)
        if energy
        else float("nan"),
        "opposite_signs": int(np.sum(r * t < 0)),
        "variance_closure_error": float(abs(c.var() - r.var() - t.var() - 2 * cov)),
        "energy_closure_error": float(
            abs(energy - np.mean(r * r) - np.mean(t * t) - 2 * np.mean(r * t))
        ),
    }
    return result
