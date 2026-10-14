#!/usr/bin/env python3
"""Body-only directed boundary transport, distinct from neural attribution."""

from __future__ import annotations

import numpy as np
from scipy.special import softmax


def causal_graph(query, keys, event_ids=None, position_only=False):
    """Construct a strict-causal head-mean transition, optionally event-local."""
    q, k = np.asarray(query, float), np.asarray(keys, float)
    if q.ndim != 3 or k.ndim != 3 or q.shape[0] != k.shape[0]:
        raise ValueError("Aligned full-token queries/keys required")
    n, heads, dim = q.shape
    if dim != k.shape[2] or heads % k.shape[1] or not np.isfinite(q).all():
        raise ValueError("Finite compatible GQA inputs required")
    if not np.isfinite(k).all() or not n:
        raise ValueError("Nonempty finite graph required")
    scores = np.stack(
        [q[:, h] @ k[:, h // (heads // k.shape[1])].T for h in range(heads)]
    )
    mask = np.tril(np.ones((n, n), bool))
    full = softmax(np.where(mask[None], scores, -np.inf), axis=2).mean(0)
    strict = np.tril(np.ones((n, n), bool), -1)
    if event_ids is not None:
        ids = np.asarray(event_ids)
        if ids.shape != (n,):
            raise ValueError("Event IDs do not align")
        strict &= ids[:, None] == ids[None, :]
    if position_only:
        weights = strict.astype(float)
    else:
        weights = np.where(strict, full, 0)
    mass = weights.sum(1)
    weights = np.divide(
        weights, mass[:, None], out=np.zeros_like(weights), where=mass[:, None] > 0
    )
    return weights, np.diag(full)


def boundary_law(transition, labels):
    """Compute A/B/unresolved absorption by a finite directed harmonic recursion."""
    t, labels = np.asarray(transition, float), np.asarray(labels)
    n = len(t)
    if t.shape != (n, n) or labels.shape != (n,) or not np.isfinite(t).all():
        raise ValueError("Finite aligned square transition and root labels required")
    if np.any(t < 0) or np.any(np.triu(t) != 0) or np.any(t.sum(1) > 1 + 1e-10):
        raise ValueError("Strict lower triangular substochastic transition required")
    if not set(labels.tolist()).issubset({-1, 0, 1, 2}):
        raise ValueError("Roots must be A0/B1/unresolved2 or unlabelled-1")
    if not {0, 1}.issubset(set(labels.tolist())):
        raise ValueError("Both named boundaries required; do not invent scores")
    out = np.zeros((n, 3))
    for i in range(n):
        if labels[i] >= 0:
            out[i, labels[i]] = 1
        else:
            out[i] = t[i, :i] @ out[:i]
            out[i, 2] += max(0.0, 1 - t[i].sum())
    return out


def direct_law(transition, labels):
    """Keep one-step named-root mass and explicitly mark other destinations unknown."""
    t, labels = np.asarray(transition, float), np.asarray(labels)
    out = np.zeros((len(t), 3))
    for i in range(len(t)):
        if labels[i] >= 0:
            out[i, labels[i]] = 1
        else:
            for c in (0, 1):
                out[i, c] = t[i, labels == c].sum()
            out[i, 2] = 1 - out[i, :2].sum()
    return out


def gated_reading(logits, values, gate, event_ids):
    """Return normalized positive reading, retained mass and exact event accounts."""
    logits, v, gate = (
        np.asarray(logits, float),
        np.asarray(values, float),
        np.asarray(gate, float),
    )
    ids = np.asarray(event_ids)
    if (
        logits.ndim != 2
        or v.ndim != 3
        or v.shape[:2] != (logits.shape[1], logits.shape[0])
    ):
        raise ValueError("Head-token logits and token-head-value arrays required")
    if gate.shape != (logits.shape[1],) or ids.shape != gate.shape:
        raise ValueError("Gate and events must align")
    if np.any(gate < 0) or np.any(gate > 1) or not np.isfinite(gate).all():
        raise ValueError("Positive finite probability gate required")
    p = softmax(logits, axis=1) * gate[None]
    mass = p.sum(1)
    if np.any(mass <= 0):
        raise ValueError("No named evidence; observation absent, not zero")
    p /= mass[:, None]
    totals, event_mass, event_reads = [], [], []
    for g in sorted(set(ids.tolist())):
        ix = ids == g
        pi = p[:, ix].sum(1)
        weighted = np.einsum("hi,ihd->hd", p[:, ix], v[ix])
        conditional = np.divide(
            weighted, pi[:, None], out=np.zeros_like(weighted), where=pi[:, None] > 0
        )
        totals.append(weighted)
        event_mass.append(pi)
        event_reads.append(conditional)
    return {
        "reading": np.sum(totals, axis=0),
        "retained_mass": mass,
        "event_mass": np.asarray(event_mass).T,
        "event_read": np.stack(event_reads, axis=1),
        "probabilities": p,
    }


def author_laws(points, epsilon=0.5):
    """Create same-marginal evidence laws differing in behavior-subject relations."""
    x = np.asarray(points, float)
    if x.shape != (16, 4) or not 0 < epsilon <= 1:
        raise ValueError("Complete four-bit design and bounded epsilon required")
    z = (x[:, 0] * x[:, 2] + x[:, 1] * x[:, 3]) / 2
    return (1 + epsilon * z) / 16, (1 - epsilon * z) / 16
