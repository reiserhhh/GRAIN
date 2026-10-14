#!/usr/bin/env python3
"""Score-free finite attention routing, event pairing and evidence laws."""

from __future__ import annotations

import itertools

import numpy as np
from scipy.special import softmax


def points():
    """Return the full ordered four-bit design without aliasing event actors."""
    return np.asarray(list(itertools.product((-1, 1), repeat=4)), dtype=float)


def event_means(values, groups, heads):
    """Repeat GQA value heads and average each declared nonempty token group."""
    v = np.asarray(values, dtype=float)
    if v.ndim != 3 or heads % v.shape[1] or not np.isfinite(v).all():
        raise ValueError("Finite sequence/KV-head/dimension values required")
    expanded = np.repeat(v, heads // v.shape[1], axis=1)
    seen, result = set(), []
    for group in groups:
        ix = np.asarray(group)
        if (
            ix.ndim != 1
            or not len(ix)
            or not np.issubdtype(ix.dtype, np.integer)
            or ix.min() < 0
            or ix.max() >= len(v)
            or len(set(ix)) != len(ix)
            or seen.intersection(ix)
        ):
            raise ValueError("Nonempty disjoint in-range groups required")
        seen.update(ix)
        result.append(expanded[ix].mean(axis=0))
    if not result:
        raise ValueError("At least one declared event required")
    return np.stack(result, axis=1)


def reading(keys, values, query, groups):
    """Keep actual body mass and exact event-uniform/within-event contributions."""
    k, v, q = (np.asarray(x, float) for x in (keys, values, query))
    if k.shape != v.shape or k.ndim != 3 or q.ndim != 2:
        raise ValueError("Paired GQA keys/values and head queries required")
    if q.shape[1] != k.shape[2] or not np.isfinite(k).all() or not np.isfinite(q).all():
        raise ValueError("Finite compatible queries and keys required")
    mu = event_means(v, groups, len(q))
    ratio = len(q) // k.shape[1]
    weights = np.stack([softmax(k[:, h // ratio] @ q[h]) for h in range(len(q))])
    masses = np.stack([weights[:, ix].sum(axis=1) for ix in groups], axis=1)
    weighted = np.stack(
        [
            np.stack([weights[h, ix] @ v[ix, h // ratio] for ix in groups])
            for h in range(len(q))
        ]
    )
    uniform_events = masses[..., None] * mu
    pairing_events = weighted - uniform_events
    mass = masses.sum(axis=1)
    if np.any(mass <= 0):
        raise ValueError("Body attention underflow: do not invent an observation")
    actual = weighted.sum(axis=1)
    uniform_body = np.mean(v[np.concatenate(groups)], axis=0)
    uniform_body = np.repeat(uniform_body, ratio, axis=0)
    return {
        "actual": actual,
        "conditional": actual / mass[:, None],
        "event_uniform": uniform_events.sum(axis=1),
        "pairing": pairing_events.sum(axis=1),
        "conditional_uniform": uniform_events.sum(axis=1) / mass[:, None],
        "conditional_pairing": pairing_events.sum(axis=1) / mass[:, None],
        "uniform_body": uniform_body,
        "masses": masses,
        "means": mu,
        "pairing_events": pairing_events,
        "body_mass": mass,
    }


def finite_action(high, low):
    """Decompose a body action across event means, allocation and token pairing."""
    mh, vh = high["masses"], high["means"]
    ml, vl = low["masses"], low["means"]
    return {
        "value": (((mh + ml) / 2)[..., None] * (vh - vl)).sum(axis=1),
        "allocation": ((mh - ml)[..., None] * (vh + vl) / 2).sum(axis=1),
        "pairing": high["pairing"] - low["pairing"],
        "actual": high["actual"] - low["actual"],
    }


def directions(source):
    """Calibrate unit behaviour directions from actor-balanced query-free values."""
    a = np.asarray(source, float)
    if a.ndim != 5 or a.shape[1:3] != (2, 2) or not np.isfinite(a).all():
        raise ValueError("target/polarity/actor/head/dimension source required")
    delta = (a[:, 1] - a[:, 0]).mean(axis=1)
    norm = np.linalg.norm(delta.reshape(len(delta), -1), axis=1)
    if np.any(norm <= 1e-12):
        raise ValueError("No source value direction; do not tune a replacement")
    return delta / norm[:, None, None], norm


def evidence_laws(epsilon=0.5):
    """Construct two laws with identical single-bit marginals but unequal yg."""
    if not 0 < epsilon <= 1:
        raise ValueError("epsilon must be in (0,1]")
    char = np.prod(points()[:, (1, 2)], axis=1)
    return (1 + epsilon * char) / 16, (1 - epsilon * char) / 16


def project(vector, direction):
    """Use a fixed source covector; no answer score, fitted head or label enters."""
    v, d = np.asarray(vector, float), np.asarray(direction, float)
    if v.shape != d.shape or v.ndim != 2 or not np.isfinite(v).all():
        raise ValueError("Aligned finite head/dimension vectors required")
    return float(np.sum(v * d))
