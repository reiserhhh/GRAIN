#!/usr/bin/env python3
"""Balanced target/wording geometry on one declared vocabulary reference law."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

FAMILIES = ("named", "described", "anchored")
AXES = ("extraversion", "neuroticism")
PAIRS = ((0, 1), (0, 2), (1, 2))
ORBIT = np.array(((1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1)))


def field_gram(fields, reference_logits, normalizers, token_ids):
    """Center all six fields under the SAME law and keep full score offsets."""
    u, z, a = [np.asarray(x, float) for x in (fields, reference_logits, normalizers)]
    ids = np.asarray(token_ids)
    if u.ndim != 3 or u.shape[0] != 6 or z.shape != u.shape[1:]:
        raise ValueError("Six aligned full-vocabulary fields required")
    if a.shape != u.shape[:2] or ids.shape != u.shape[1:2]:
        raise ValueError("Aligned normalizers and selected native tokens required")
    if ids.dtype.kind not in "iu" or ids.min() < 0 or ids.max() >= u.shape[-1]:
        raise ValueError("Native token outside vocabulary")
    if not all(np.isfinite(x).all() for x in (u, z, a)):
        raise ValueError("Nonfinite field/reference")
    q = np.exp(z - logsumexp(z, axis=1, keepdims=True))
    if np.any(q == 0):
        raise ArithmeticError("Reference underflow; cannot assert positive support")
    means = np.sum(u * q[None], axis=-1)
    centered = u - means[:, :, None]
    weighted = centered.transpose(1, 0, 2) * np.sqrt(q)[:, None, :]
    gram = weighted @ weighted.swapaxes(1, 2)
    selected = u[:, np.arange(len(ids)), ids]
    return {
        "gram": gram,
        "reference_mean_u": means,
        "selected_u": selected,
        "selected_shape": selected - means,
        "calibration": means - a,
        "reconstructed_k": selected - a,
        "reference_min_probability": q.min(axis=1),
    }


def gram_accounts(gram):
    """Compute finite Hilbert FANOVA and the four balanced label partitions."""
    g = np.asarray(gram, float)
    if g.ndim != 3 or g.shape[1:] != (6, 6) or not np.isfinite(g).all():
        raise ValueError("Finite six-field Gram required")
    scale = np.maximum(1.0, np.max(abs(g), axis=(1, 2)))
    if np.any(np.max(abs(g - g.swapaxes(1, 2)), axis=(1, 2)) > 1e-10 * scale):
        raise ValueError("Nonsymmetric Gram")
    if np.any(np.linalg.eigvalsh(g)[:, 0] < -1e-10 * scale):
        raise ValueError("Non-PSD Gram; no projection repair")
    ones, signs = np.ones(6) / 6, np.array([1, 1, 1, -1, -1, -1]) / 6
    gm = (g[:, :3, :3] + g[:, 3:, 3:] + g[:, :3, 3:] + g[:, 3:, :3]) / 4
    gd = (g[:, :3, :3] + g[:, 3:, 3:] - g[:, :3, 3:] - g[:, 3:, :3]) / 4
    common = np.einsum("i,tij,j->t", ones, g, ones)
    target = np.einsum("i,tij,j->t", signs, g, signs)
    wording = np.trace(gm, axis1=1, axis2=2) / 3 - common
    interaction = np.trace(gd, axis1=1, axis2=2) / 3 - target
    total = np.trace(g, axis1=1, axis2=2) / 6
    cross = np.stack([gd[:, f, h] for f, h in PAIRS], axis=1)
    orbit = np.stack(
        [(2 / 3) * sum(e[f] * e[h] * gd[:, f, h] for f, h in PAIRS) for e in ORBIT],
        axis=1,
    )
    norms = np.diagonal(gd, axis1=1, axis2=2)
    cosines = []
    for f, h in PAIRS:
        product = norms[:, f] * norms[:, h]
        denominator = np.sqrt(np.where(product > 0, product, 0))
        cosines.append(
            np.divide(
                gd[:, f, h],
                denominator,
                out=np.full(len(g), np.nan),
                where=(norms[:, f] > 0) & (norms[:, h] > 0),
            )
        )
    cosines = np.stack(cosines, axis=1)
    all_defined = np.isfinite(cosines).all(axis=1)
    mean_cosine = np.where(all_defined, np.sum(cosines, axis=1) / 3, np.nan)
    margin = orbit[:, 0] - np.max(orbit[:, 1:], axis=1)
    return {
        "difference_gram": gd,
        "common_energy": common,
        "target_energy": target,
        "wording_energy": wording,
        "interaction_energy": interaction,
        "total_energy": total,
        "energy_closure_error": total - common - target - wording - interaction,
        "cross_family_covariance": cross,
        "cross_family_cosines": cosines,
        "mean_cross_family_cosine": mean_cosine,
        "orbit_scores": orbit,
        "intended_support": orbit[:, 0],
        "orbit_mean_error": orbit.mean(axis=1),
        "support_identity_error": orbit[:, 0] - (2 * target - interaction),
        "intended_margin": margin,
        "strict_first": margin > 1e-10,
        "tied_first": abs(margin) <= 1e-10,
        "not_first": margin < -1e-10,
    }
