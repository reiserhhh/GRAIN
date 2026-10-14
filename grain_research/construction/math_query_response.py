#!/usr/bin/env python3
"""HM34 fixed-metric source-only extension of early-query response covectors."""

from __future__ import annotations

import numpy as np


def columns(values):
    """Require a finite physical-coordinate matrix with samples in columns."""
    x = np.asarray(values, float)
    if x.ndim != 2 or min(x.shape) < 1 or not np.isfinite(x).all():
        raise ValueError("finite nonempty physical-coordinate columns required")
    return x


def source_columns(records):
    """Reject application or validation records before fitting any operator."""
    if len(records) != 8 or any(r["role"] != "source" for r in records):
        raise ValueError("exactly8 source-only pair/wording records required")
    keys = [(r["pair"], r["wording"]) for r in records]
    if set(keys) != {(p, w) for p in (0, 1) for w in range(4)}:
        raise ValueError("complete source pair/wording crossing required")
    if len({r["query"] for r in records}) != 1:
        raise ValueError("query spaces must not be concatenated")
    return (
        columns(np.stack([r["d"] for r in records], axis=1)),
        columns(np.stack([r["g"] for r in records], axis=1)),
    )


def fit(d, g):
    """Fit one minimum-norm extension with a fixed Gram-eigenvalue tolerance."""
    d, g = columns(d), columns(g)
    if d.shape[1] != g.shape[1]:
        raise ValueError("paired source columns required")
    gram = d.T @ d
    eigenvalues, vectors = np.linalg.eigh(gram)
    tolerance = (
        100 * np.finfo(float).eps * max(gram.shape) * max(float(eigenvalues[-1]), 0.0)
    )
    if eigenvalues[0] < -max(tolerance, np.finfo(float).tiny):
        raise ValueError("Gram matrix violates fixed PSD tolerance")
    keep = eigenvalues > tolerance
    inverse = (vectors[:, keep] / eigenvalues[keep]) @ vectors[:, keep].T
    reconstructed = g @ inverse @ gram
    return {
        "D": d,
        "G": g,
        "gram_inverse": inverse,
        "eigenvalues": eigenvalues,
        "rank": int(keep.sum()),
        "tolerance": tolerance,
        "source_interpolation_error": float(np.linalg.norm(reconstructed - g)),
        "source_target_norm": float(np.linalg.norm(g)),
    }


def predict(model, delta):
    """Return a source-supported response and retain the unidentified complement."""
    d, g = columns(model["D"]), columns(model["G"])
    x = np.asarray(delta, float)
    if x.shape != (d.shape[0],) or not np.isfinite(x).all():
        raise ValueError("matched finite query coordinates required")
    weights = model["gram_inverse"] @ (d.T @ x)
    supported = d @ weights
    unsupported = x - supported
    return {
        "predicted": g @ weights,
        "weights": weights,
        "supported": supported,
        "unsupported": unsupported,
    }


def paired_gradient(cube):
    """Recover pure P and every fixed H/B/Q response by paired averaging."""
    x = np.asarray(cube, float)
    if x.shape[:4] != (2, 2, 2, 2) or not np.isfinite(x).all():
        raise ValueError("complete H/B/P/Q gradient cube required")
    fixed = (x[:, :, 0] - x[:, :, 1]) / 2
    common = fixed.mean(axis=(0, 1, 2))
    return common, fixed


def paired_tokens(left, right):
    """Keep identical ordered query tokens, roles and semantic pairing."""
    for key in ("query", "pair", "wording", "role"):
        if left[key] != right[key]:
            raise ValueError("paired query carriers must share group and role")
    if (
        left["early_query_side"] != 0
        or right["early_query_side"] != 1
        or left["early_query_token_ids"] != right["early_query_token_ids"]
        or not left["early_query_token_ids"]
    ):
        raise ValueError("exact nonempty ordered P token sequence required")
    return tuple(left["early_query_token_ids"])


def error_parts(predicted, common, fixed, delta, actual):
    """Separate operator, contextual interaction and finite-action errors."""
    p, c, f, d = (np.asarray(v, float) for v in (predicted, common, fixed, delta))
    if p.shape != c.shape or p.shape != f.shape or p.shape != d.shape:
        raise ValueError("matched physical covectors and displacement required")
    values = {
        "predicted": float(np.sum(p * d)),
        "common_local": float(np.sum(c * d)),
        "fixed_local": float(np.sum(f * d)),
        "actual": float(actual),
        "operator_error": float(np.sum((p - c) * d)),
        "interaction_error": float(np.sum((c - f) * d)),
        "finite_error": float(np.sum(f * d) - actual),
    }
    if not all(np.isfinite(v) for v in values.values()):
        raise ValueError("finite addressed action comparison required")
    values["identity_error"] = abs(
        values["predicted"]
        - actual
        - sum(
            values[k] for k in ("operator_error", "interaction_error", "finite_error")
        )
    )
    return values


def statistics(actual, predicted):
    """Keep magnitude error, zero-response reference and nonzero sign counts."""
    y, p = (np.asarray(v, float) for v in (actual, predicted))
    if y.shape != p.shape or y.size == 0 or not np.isfinite(y + p).all():
        raise ValueError("matched nonempty finite outputs required")
    rms = float(np.sqrt(np.mean(y * y)))
    rmse = float(np.sqrt(np.mean((p - y) ** 2)))
    return {
        "values": int(y.size),
        "actual_rms": rms,
        "prediction_rms": float(np.sqrt(np.mean(p * p))),
        "rmse": rmse,
        "relative_rmse": rmse / rms if rms else None,
        "zero_response_rmse": rms,
        "sign_agreement": int(np.sum(y * p > 0)),
        "actual_nonzero": int(np.count_nonzero(y)),
    }
