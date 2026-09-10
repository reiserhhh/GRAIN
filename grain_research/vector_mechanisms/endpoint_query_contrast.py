#!/usr/bin/env python3
"""Symmetric bilinear query contrast on frozen endpoint vectors and readers."""

import numpy as np

TERMS = ("evidence_change", "reader_difference", "total")
OBSERVABLES = ("scalar", "key", "value", "total_action")


def decompose(vectors, readers):
    """Split paired-query inner products without changing either query target.

    vectors has [query, ..., hidden], readers [query, hidden]. The first output
    axis enumerates the two symmetric terms and their sum.
    """
    p, r = np.asarray(vectors, float), np.asarray(readers, float)
    if (
        p.ndim < 2
        or p.shape[0] != 2
        or r.shape != (2, p.shape[-1])
        or not all(np.isfinite(x).all() for x in (p, r))
    ):
        raise ValueError("Exactly two aligned finite query vectors/readers required")
    e = (p[0] - p[1]) @ r.mean(axis=0)
    d = p.mean(axis=0) @ (r[0] - r[1])
    direct = p[0] @ r[0] - p[1] @ r[1]
    np.testing.assert_allclose(e + d, direct, atol=1e-10, rtol=1e-10)
    return np.stack((e, d, e + d))


def split_packets(primal, vectors, readers):
    """Return [arm,term] scalars and [arm,component,term,factor] tangents."""
    return {
        "scalar_terms": decompose(primal, readers).T,
        "action_terms": np.moveaxis(decompose(vectors, readers), 0, 2),
    }


def geometry(readers):
    """Describe two source readers only; no new learned basis or target selection."""
    r = np.asarray(readers, float)
    if r.ndim != 2 or len(r) != 2 or not np.isfinite(r).all():
        raise ValueError("Two finite source readers required")
    mean, diff = r.mean(axis=0), r[0] - r[1]
    norms = np.linalg.norm(r, axis=1)
    s = np.linalg.svd(r, compute_uv=False)

    def cosine(a, b):
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(a @ b / denom) if denom else 0.0

    return {
        "N_norm": float(norms[0]),
        "confirmation_norm": float(norms[1]),
        "mean_norm": float(np.linalg.norm(mean)),
        "difference_norm": float(np.linalg.norm(diff)),
        "query_cosine": cosine(r[0], r[1]),
        "mean_difference_cosine": cosine(mean, diff),
        "second_over_first_singular": float(s[1] / s[0]) if s[0] else 0.0,
        "reader_rank": int(np.sum(s > (s[0] * max(r.shape) * np.finfo(float).eps))),
    }


def moments(terms):
    """Raw energy and within-column author-centered covariance of two terms.

    Axis0 is authors, axis1 terms, any trailing columns are fixed GRAIN factors.
    Centering is descriptive only and never modifies a prediction.
    """
    x = np.asarray(terms, float)
    if x.ndim < 2 or x.shape[1] != 3 or len(x) < 3 or not np.isfinite(x).all():
        raise ValueError("Finite author-by-three-term panel required")
    np.testing.assert_allclose(x[:, 0] + x[:, 1], x[:, 2], atol=1e-10)
    e, d, total = (x[:, i] for i in range(3))
    z = x - x.mean(axis=0, keepdims=True)
    a, b, t = (z[:, i] for i in range(3))
    energy, var = np.mean(total * total), np.mean(t * t)
    result = {name + "_mean": float(np.mean(x[:, i])) for i, name in enumerate(TERMS)}
    result.update(
        {
            name + "_rms": float(np.sqrt(np.mean(x[:, i] ** 2)))
            for i, name in enumerate(TERMS)
        }
    )
    result.update(
        {
            name + "_author_variance": float(np.mean(z[:, i] ** 2))
            for i, name in enumerate(TERMS)
        }
    )
    result.update(
        raw_cross_energy=float(2 * np.mean(e * d)),
        author_cross_covariance=float(2 * np.mean(a * b)),
        evidence_centered_projection=float(np.mean(a * t) / var)
        if var > 1e-24
        else 0.0,
        reader_centered_projection=float(np.mean(b * t) / var) if var > 1e-24 else 0.0,
        raw_identity_error=float(
            abs(energy - np.mean(e * e) - np.mean(d * d) - 2 * np.mean(e * d))
        ),
        centered_identity_error=float(
            abs(var - np.mean(a * a) - np.mean(b * b) - 2 * np.mean(a * b))
        ),
    )
    return result


def error_account(actual, predicted):
    """Keep term errors, their signed cross term and descriptive centered errors."""
    y, p = np.asarray(actual, float), np.asarray(predicted, float)
    if y.shape != p.shape:
        raise ValueError("Aligned target and prediction required")
    m = moments(p - y)
    return {
        "evidence_mse": m["evidence_change_rms"] ** 2,
        "reader_mse": m["reader_difference_rms"] ** 2,
        "total_mse": m["total_rms"] ** 2,
        "signed_cross_mse": m["raw_cross_energy"],
        "evidence_centered_mse": m["evidence_change_author_variance"],
        "reader_centered_mse": m["reader_difference_author_variance"],
        "total_centered_mse": m["total_author_variance"],
        "centered_cross_mse": m["author_cross_covariance"],
        "raw_identity_error": m["raw_identity_error"],
        "centered_identity_error": m["centered_identity_error"],
    }
