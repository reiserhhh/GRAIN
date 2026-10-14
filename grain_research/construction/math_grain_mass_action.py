#!/usr/bin/env python3
"""Source-only anonymous marks define finite, mass-preserving evidence actions.

Exponential tilting and covariance geometry are standard methods. This module
tests a concrete GRAIN-to-vector interface, not a new attention theorem.
"""

from __future__ import annotations

import numpy as np


def source_geometry(marks, mass, visible, pairs):
    """Use original expressions only, never held response values, to define axes."""
    g, m, mask = (
        np.asarray(marks, float),
        np.asarray(mass, float),
        np.asarray(visible, bool),
    )
    if (
        g.ndim != 3
        or g.shape[-1] != 4
        or m.shape != g.shape[:2]
        or mask.shape != m.shape
    ):
        raise ValueError("Addressed four-coordinate marks and physical masses required")
    if not np.isfinite(g).all() or not np.isfinite(m).all() or (m < 0).any():
        raise ValueError("Finite marks and nonnegative masses required")
    source = [p for p in pairs if p["key"][0] == 0]
    ids = sorted({p[side] for p in source for side in ("low", "high")})
    weights = m[ids] * mask[ids]
    total = weights.sum()
    if not total > 0 or (weights.sum(1) <= 0).any() or len(source) != 48:
        raise ValueError("All48 original pairs and96 nonempty source bodies required")
    mean = np.einsum("ui,uij->j", weights, g[ids]) / total
    centered = g[ids] - mean
    covariance = np.einsum("ui,uij,uik->jk", weights, centered, centered) / total
    eig, vectors = np.linalg.eigh(covariance)
    keep = eig > max(float(eig.max()), 1.0) * 1e-10
    inverse = (vectors[:, keep] / eig[keep]) @ vectors[:, keep].T
    body_weights = m * mask
    body_mean = np.einsum("ui,uij->uj", body_weights, g) / body_weights.sum(1)[:, None]
    contrast = np.stack(
        [
            np.mean(
                [
                    body_mean[p["high"]] - body_mean[p["low"]]
                    for p in source
                    if p["key"][2] == family
                ],
                axis=0,
            )
            for family in range(6)
        ]
    )
    norm = np.sqrt(np.maximum(np.einsum("qj,jk,qk->q", contrast, inverse, contrast), 0))
    available = norm > 1e-10
    axes = np.zeros_like(contrast)
    axes[available] = (contrast[available] @ inverse) / norm[available, None]
    return {
        "mean": mean,
        "covariance": covariance,
        "inverse": inverse,
        "axes": axes,
        "contrast": contrast,
        "norm": norm,
        "available": available,
        "source_ids": np.array(ids),
        "rank": np.array(keep.sum()),
    }


def scores(marks, mass, visible, axes, clip=3.0):
    """Center within physical body mass; keep source-defined target covectors."""
    g, m, mask = (
        np.asarray(marks, float),
        np.asarray(mass, float),
        np.asarray(visible, bool),
    )
    weight = m * mask
    if (weight.sum(1) <= 0).any() or clip <= 0:
        raise ValueError("Nonempty body measure and positive fixed clipping required")
    center = np.einsum("ui,uij->uj", weight, g) / weight.sum(1)[:, None]
    s = np.einsum("uij,qj->uqi", g - center[:, None], axes)
    return np.clip(s, -clip, clip) * mask[:, None]


def shifted_scores(score, visible):
    """Move the same score multiset by a fixed half-cycle within each body."""
    s, mask = np.asarray(score, float), np.asarray(visible, bool)
    wrong = np.zeros_like(s)
    for user in range(len(s)):
        ix = np.flatnonzero(mask[user])
        wrong[user][:, ix] = np.roll(s[user][:, ix], len(ix) // 2, axis=1)
    return wrong


def tilt(mass, visible, score, epsilon):
    """Solve a KL-penalized finite preference change, preserving total body mass."""
    m, mask, s = (
        np.asarray(mass, float),
        np.asarray(visible, bool),
        np.asarray(score, float),
    )
    if m.shape != mask.shape or m.shape != s.shape or (m < 0).any():
        raise ValueError("Matching nonnegative physical mass, mask and score required")
    if not np.isfinite(m).all() or not np.isfinite(s).all() or not np.isfinite(epsilon):
        raise ValueError("Finite intervention required")
    b = m * mask
    if (b.sum(1) <= 0).any():
        raise ValueError("Cannot tilt an unobserved body")
    logit = epsilon * s
    origin = np.where(b > 0, logit, -np.inf).max(1, keepdims=True)
    weighted = b * np.exp(np.where(b > 0, logit - origin, 0))
    return m * (~mask) + weighted * b.sum(1, keepdims=True) / weighted.sum(
        1, keepdims=True
    )


def intervention_masses(mass, visible, score, epsilon):
    """Return baseline then target/alignment/sign arms in one fixed order."""
    wrong = shifted_scores(score, visible)
    arms = [np.asarray(mass, float)]
    for q in range(6):
        for values in (score, wrong):
            for sign in (-1, 1):
                arms.append(tilt(mass, visible, values[:, q], sign * epsilon))
    return np.stack(arms, axis=1)


def response_field(readings, pairs):
    """Keep all finite signs, focuses and owners; do not label them personality."""
    r = np.asarray(readings, float)
    if r.shape != (288, 25, 2, 2, 6) or not np.isfinite(r).all():
        raise ValueError("All288 units,25 arms,two focuses/modes,six readouts required")
    arms = r[:, 1:].reshape(288, 6, 2, 2, 2, 2, 6)
    action = (arms[:, :, :, 1] - arms[:, :, :, 0]) / 2
    baseline = np.zeros((3, 2, 6, 2, 2, 2, 2, 6))
    field = np.zeros((3, 2, 6, 2, 2, 6, 2, 2, 2, 6))
    for p in pairs:
        variant, wording, family, condition, owner = p["key"]
        ix = (variant, wording, family, (condition + 1) // 2, (owner + 1) // 2)
        baseline[ix] = (r[p["high"], 0] - r[p["low"], 0]) / 2
        field[ix] = (action[p["high"]] - action[p["low"]]) / 2
    focused = field[..., 0, :, :] - field[..., 1, :, :]
    baseline_focused = baseline[..., 0, :, :] - baseline[..., 1, :, :]
    return {
        "action": action,
        "paired": field,
        "actor_odd": (focused[:, :, :, :, 1] - focused[:, :, :, :, 0]) / 2,
        "actor_even": (focused[:, :, :, :, 1] + focused[:, :, :, :, 0]) / 2,
        "baseline_actor_odd": (
            baseline_focused[:, :, :, :, 1] - baseline_focused[:, :, :, :, 0]
        )
        / 2,
    }


def selectivity(field, read_scale):
    """Compare same-family cue/readout to off-target responses in source units."""
    x = np.asarray(field, float) / np.asarray(read_scale, float)
    if x.shape != (3, 2, 6, 2, 6, 2, 2, 6):
        raise ValueError(
            "Full actor-odd family/condition/action/readout field required"
        )
    rows = []
    for variant, wording, family, condition, alignment, mode in np.ndindex(
        3, 2, 6, 2, 2, 2
    ):
        y = x[variant, wording, family, condition, family, alignment, mode]
        target = abs(y[family])
        off = np.delete(abs(y), family).mean()
        rows.append(
            {
                "variant": variant,
                "wording": wording,
                "family": family,
                "condition": condition,
                "alignment": alignment,
                "mode": mode,
                "target_signed": float(y[family]),
                "target_abs": float(target),
                "off_target_mean_abs": float(off),
                "selectivity": float(target - off),
            }
        )
    return rows
