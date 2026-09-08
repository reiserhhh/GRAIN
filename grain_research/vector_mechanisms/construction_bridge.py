#!/usr/bin/env python3
"""Finite natural-response accounting; no neural execution or outcome fitting."""

from __future__ import annotations

import hashlib

import numpy as np


def author_split(authors, salt, reference_count=12):
    """Split unique authors by a salted hash, never by measured responses."""
    if (
        len(set(authors)) != len(authors)
        or not 2 <= reference_count <= len(authors) - 2
    ):
        raise ValueError("Unique authors and two nontrivial groups required")
    order = sorted(
        range(len(authors)),
        key=lambda i: hashlib.sha256(f"{salt}|{authors[i]}".encode()).hexdigest(),
    )
    mask = np.zeros(len(authors), dtype=bool)
    mask[order[:reference_count]] = True
    return mask


def validate_arrays(gradients, displacement, natural, patch):
    """Require complete A,V,E,Q,D arrays with one gradient per author/view/query."""
    g, d, y, m = map(np.asarray, (gradients, displacement, natural, patch))
    if d.ndim != 5 or g.shape != (d.shape[0], d.shape[1], d.shape[3], d.shape[4]):
        raise ValueError("Gradient/displacement axes do not align")
    if y.shape != d.shape[:-1] or m.shape != y.shape:
        raise ValueError("Natural and patch outcomes must match all cells")
    if not all(np.isfinite(x).all() for x in (g, d, y, m)):
        raise ValueError("Nonfinite observations")


def fit_reference(gradients, natural, reference):
    """Estimate equal-author gradients, scales and cell means on reference only."""
    ref = np.asarray(reference, bool)
    if ref.sum() < 2 or ref.shape != (len(gradients),):
        raise ValueError("At least two aligned reference authors required")
    g = np.asarray(gradients)[ref].mean(axis=1).mean(axis=0)
    y = np.asarray(natural)[ref]
    scale = y.std(axis=(0, 1, 2))
    if np.any(scale <= 1e-12):
        raise ValueError("Reference response units degenerate; do not divide by zero")
    return g, scale, y.mean(axis=0)


def crossed_predictions(reference_gradient, displacement):
    """Return A,V,E,reader,recipient scores in the declared common vector space."""
    g, d = np.asarray(reference_gradient), np.asarray(displacement)
    if d.ndim != 5 or g.shape != d.shape[-2:] or not np.isfinite(g).all():
        raise ValueError("Aligned finite query readers and displacement required")
    return np.einsum("rd,aveqd->averq", g, d)


def candidate_predictions(reference_gradient, displacement, cell_means, scale=None):
    """Compare fixed shared/private constructions without any evaluation targets."""
    q = displacement.shape[-2]
    unit = np.ones(q) if scale is None else np.asarray(scale)
    if unit.shape != (q,) or np.any(unit <= 0) or not np.isfinite(unit).all():
        raise ValueError("Positive finite query units required")
    g = reference_gradient / unit[:, None]
    b = crossed_predictions(g, displacement)
    own = np.diagonal(b, axis1=-2, axis2=-1).copy()
    return {
        "zero": np.zeros_like(own),
        "reference_cell_mean": np.broadcast_to(cell_means / unit, own.shape).copy(),
        "shared_evidence_private_reader": b.mean(axis=-1),
        "private_evidence_shared_reader": b.mean(axis=-2),
        "private_evidence_private_reader": own,
    }, b


def two_order_allocation(crossed, left, right):
    """Allocate the diagonal contrast exactly; report interaction without readding."""
    b = np.asarray(crossed)
    qq, qr = b[..., left, left], b[..., left, right]
    rq, rr = b[..., right, left], b[..., right, right]
    return {
        "difference": qq - rr,
        "readout": ((qq - rq) + (qr - rr)) / 2,
        "evidence": ((qq - qr) + (rq - rr)) / 2,
        "interaction": qq - qr - rq + rr,
    }


def response_account(natural, patch, individual_linear, transported):
    """Keep transported, gradient, nonlinear and unpatched contributions signed."""
    return {
        "transported": transported,
        "gradient_transport_error": individual_linear - transported,
        "local_nonlinearity": patch - individual_linear,
        "outside_site": natural - patch,
    }


def covariance_matrix(x, y=None, centered_within_cell=True):
    """Empirical ddof=0 covariance; condition means never masquerade as author signal."""
    x = np.asarray(x, float)
    y = x if y is None else np.asarray(y, float)
    if x.ndim < 3 or x.shape != y.shape or len(x) < 2:
        raise ValueError("Matched authors, cells, queries required")
    x, y = x.reshape(len(x), -1, x.shape[-1]), y.reshape(len(y), -1, y.shape[-1])
    axes = 0 if centered_within_cell else (0, 1)
    x, y = x - x.mean(axis=axes, keepdims=True), y - y.mean(axis=axes, keepdims=True)
    return np.einsum("acq,acr->qr", x, y) / (x.shape[0] * x.shape[1])


def covariance_account(natural, predicted, centered_within_cell=True):
    """Exact four-term cross-query covariance; the remainder is descriptive only."""
    e = natural - predicted
    cov = lambda x, y=None: covariance_matrix(x, y, centered_within_cell)
    return {
        "observed": cov(natural),
        "functional": cov(predicted),
        "functional_remainder": cov(predicted, e),
        "remainder_functional": cov(e, predicted),
        "remainder": cov(e),
    }


def correlation(x, y):
    """Pearson with explicit undefined results, never a fake zero for constants."""
    x, y = np.asarray(x).ravel(), np.asarray(y).ravel()
    x, y = x - x.mean(), y - y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.clip(x @ y / denom, -1, 1)) if denom > 1e-18 else None


def quality(target, prediction):
    """Effect fidelity and magnitude, not questionnaire accuracy or variance shares."""
    y, p = np.asarray(target), np.asarray(prediction)
    power, mse = float(np.mean(y**2)), float(np.mean((p - y) ** 2))
    return {
        "r": correlation(y, p),
        "mse": mse,
        "mae": float(np.mean(np.abs(p - y))),
        "target_rms": float(np.sqrt(power)),
        "prediction_rms": float(np.sqrt(np.mean(p**2))),
        "mse_over_zero": mse / power if power > 1e-18 else None,
        "sign_agreement": float(np.mean(np.sign(y) == np.sign(p))),
    }


def bootstrap_quality(target, prediction, draws=1000, seed=20260908):
    """Resample entire first-axis authors, conditional on frozen reference fitting."""
    y, p = np.asarray(target), np.asarray(prediction)
    if y.shape != p.shape or len(y) < 2:
        raise ValueError("Paired author arrays required")
    rng = np.random.default_rng(seed)
    observed = quality(y, p)
    samples = {name: [] for name in ("r", "mse", "mse_over_zero")}
    for _ in range(draws):
        take = rng.integers(len(y), size=len(y))
        scores = quality(y[take], p[take])
        for name, values in samples.items():
            if scores[name] is not None:
                values.append(scores[name])
    for name, values in samples.items():
        interval = (
            np.quantile(values, [0.025, 0.975]).tolist() if values else [None, None]
        )
        observed[name + "_lower"], observed[name + "_upper"] = interval
    return observed
