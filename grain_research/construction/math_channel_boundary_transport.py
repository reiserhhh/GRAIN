#!/usr/bin/env python3
"""Direct-sum boundary laws and paired reading, not neural causal attribution."""

from __future__ import annotations

import numpy as np
from scipy.special import softmax

from grain_research.construction import math_body_relation_graph as B


def channel_graphs(query, keys):
    """Keep all strict-causal QK channels and their self mass separately."""
    q, k = np.asarray(query, float), np.asarray(keys, float)
    if q.ndim != 3 or k.ndim != 3 or q.shape[0] != k.shape[0]:
        raise ValueError("Aligned sequence/head/dimension required")
    n, heads, dim = q.shape
    if not n or dim != k.shape[2] or heads % k.shape[1]:
        raise ValueError("Nonempty compatible GQA input required")
    if not np.isfinite(q).all() or not np.isfinite(k).all():
        raise ValueError("Finite input required")
    score = np.stack(
        [q[:, h] @ k[:, h // (heads // k.shape[1])].T for h in range(heads)]
    )
    mask = np.tril(np.ones((n, n), bool))
    full = softmax(np.where(mask[None], score, -np.inf), axis=2)
    strict = np.where(np.tril(mask, -1)[None], full, 0)
    mass = strict.sum(2)
    t = np.divide(
        strict, mass[:, :, None], out=np.zeros_like(strict), where=mass[:, :, None] > 0
    )
    return t, np.diagonal(full, axis1=1, axis2=2)


def channel_laws(transition, labels, direct=False):
    """Solve each channel with the same named and unresolved boundaries."""
    t = np.asarray(transition, float)
    if t.ndim != 3 or t.shape[1] != t.shape[2]:
        raise ValueError("Headwise square transitions required")
    solver = B.direct_law if direct else B.boundary_law
    return np.stack([solver(head, labels) for head in t])


def harmonic_collapse_account(transition, laws, labels):
    """Decompose solving-before-averaging with a finite covariance identity."""
    t, h, labels = (
        np.asarray(transition, float),
        np.asarray(laws, float),
        np.asarray(labels),
    )
    mean_t, mean_h = t.mean(0), h.mean(0)
    scalar = B.boundary_law(mean_t, labels)
    covariance = np.einsum("hij,hjc->ic", t - mean_t, h - mean_h) / len(t)
    ix = labels < 0
    propagated = mean_h - scalar
    equation = propagated - mean_t @ propagated
    return {
        "scalar_law": scalar,
        "mean_law": mean_h,
        "covariance": covariance,
        "collapse_rms": float(np.sqrt(np.mean(propagated**2))),
        "identity_error": float(np.max(np.abs(equation[ix] - covariance[ix]))),
    }


def measure(logits, projected_values, gate, event_ids):
    """Keep unnormalized action/mass and headwise conditional event accounts."""
    logits, c, gate = (np.asarray(a, float) for a in (logits, projected_values, gate))
    ids = np.asarray(event_ids)
    if logits.ndim != 2 or c.shape != logits.shape or gate.shape != logits.shape:
        raise ValueError("Head/token logits, values and gates must align")
    if ids.shape != (logits.shape[1],) or not np.isfinite([logits, c, gate]).all():
        raise ValueError("Finite arrays and aligned event IDs required")
    if np.any(gate < 0) or np.any(gate > 1):
        raise ValueError("Gate must be a boundary probability")
    p = softmax(logits, axis=1) * gate
    mass = p.sum(1)
    if np.any(mass <= 0):
        raise ValueError("Absent named observation, not a zero score")
    numerator = np.sum(p * c, axis=1)
    event_mass, event_numerator = [], []
    for g in sorted(set(ids.tolist())):
        selected = ids == g
        event_mass.append(p[:, selected].sum(1) / mass)
        event_numerator.append(np.sum(p[:, selected] * c[:, selected], axis=1) / mass)
    event_mass, event_contribution = np.array(event_mass).T, np.array(event_numerator).T
    return {
        "mass": mass,
        "numerator": numerator,
        "heads": numerator / mass,
        "score": float(np.sum(numerator / mass)),
        "raw_action": float(np.sum(numerator)),
        "event_mass": event_mass,
        "event_contribution": event_contribution,
    }


def conditioning_account(paired, averaged):
    """Separate numerator pairing from the effect of changed retained mass."""
    numerator = (paired["numerator"] - averaged["numerator"]) / paired["mass"]
    normalization = (
        -(paired["mass"] - averaged["mass"]) * averaged["heads"] / paired["mass"]
    )
    error = paired["heads"] - averaged["heads"] - numerator - normalization
    return {
        "numerator_term": float(numerator.sum()),
        "normalization_term": float(normalization.sum()),
        "difference": paired["score"] - averaged["score"],
        "identity_error": float(np.max(np.abs(error))),
    }
