#!/usr/bin/env python3
"""Finite qualified-event construction, reading and author evidence laws."""

from __future__ import annotations

import itertools

import numpy as np

SIGNS = np.array(list(itertools.product((-1, 1), repeat=3)), dtype=int)
POWERS = np.array(list(itertools.product((0, 1), repeat=3)), dtype=int)
CHARACTERS = np.prod(SIGNS[:, None, :] ** POWERS[None, :, :], axis=2)
AUTHOR_CORNERS = np.flatnonzero(SIGNS[:, 2] == 1)
LAW_COUNTS = np.array([[7, 3, 3, 7], [3, 7, 7, 3]], dtype=int)


def design(n_traits=6):
    """Enumerate mid anchors, singles and every unordered pair with two owners."""
    if n_traits < 2:
        raise ValueError("At least two behavior slots required")
    for owner in (-1, 1):
        yield {"kind": "base", "levels": [1] * n_traits, "owner": owner}
    for i, sign, owner in itertools.product(range(n_traits), (-1, 1), (-1, 1)):
        levels = [1] * n_traits
        levels[i] = sign + 1
        yield {"kind": "single", "i": i, "sign": sign, "levels": levels, "owner": owner}
    for i, j in itertools.combinations(range(n_traits), 2):
        for x, y, owner in SIGNS:
            levels = [1] * n_traits
            levels[i], levels[j] = int(x + 1), int(y + 1)
            yield {
                "kind": "pair",
                "i": i,
                "j": j,
                "levels": levels,
                "owner": int(owner),
            }


def key(row):
    """Address a full evidence configuration, not a filename or inferred author."""
    return tuple(row["levels"]), row["owner"]


def compose(base, first, second):
    """Form an anchored counterfactual in common K/V coordinates."""
    arrays = [np.asarray(a, dtype=float) for a in (base, first, second)]
    if len({a.shape for a in arrays}) != 1 or not all(
        np.isfinite(a).all() for a in arrays
    ):
        raise ValueError("Finite aligned states required")
    return arrays[1] + arrays[2] - arrays[0]


def attention(k, v, q, d):
    """Read a declared source-frozen GQA observer, without fitting native answers."""
    k, v, q, d = (np.asarray(a, dtype=float) for a in (k, v, q, d))
    if (
        k.shape != v.shape
        or k.ndim != 3
        or q.shape != d.shape
        or q.ndim != 3
        or q.shape[-1] != k.shape[-1]
        or q.shape[1] % k.shape[1]
        or not all(np.isfinite(a).all() for a in (k, v, q, d))
        or not len(k)
    ):
        raise ValueError("Finite compatible key/value/query/covector arrays required")
    heads = np.arange(q.shape[1]) // (q.shape[1] // k.shape[1])
    logits = np.einsum("ahd,thd->ath", k[:, heads], q)
    weights = np.exp(logits - logits.max(axis=0))
    weights /= weights.sum(axis=0)
    content = np.einsum("ahd,thd->ath", v[:, heads], d)
    return (weights * content).sum(axis=(0, 2))


def attention_loop(k, v, q, d):
    """Independent primitive per-head readout for numerical verification."""
    result = []
    ratio = q.shape[1] // k.shape[1]
    for query, direction in zip(q, d, strict=True):
        score = 0.0
        for h in range(len(query)):
            logits = k[:, h // ratio] @ query[h]
            p = np.exp(logits - max(logits))
            p = p / sum(p)
            score += sum(p * (v[:, h // ratio] @ direction[h]))
        result.append(score)
    return np.array(result)


def coefficients(corners):
    """Return every joint character, not a selected interaction winner."""
    f = np.asarray(corners, dtype=float)
    if f.shape[0] != 8 or not np.isfinite(f).all():
        raise ValueError("Complete finite eight-corner field required")
    return np.tensordot(CHARACTERS.T / 8, f, axes=(1, 0))


def account(actual, additive_read, additive_state):
    """Split anchored defect into reading and encoding terms with signed closure."""
    actual, additive_read, additive_state = (
        np.asarray(a, float) for a in (actual, additive_read, additive_state)
    )
    if len({a.shape for a in (actual, additive_read, additive_state)}) != 1:
        raise ValueError("Common observation shapes required")
    if not all(np.isfinite(a).all() for a in (actual, additive_read, additive_state)):
        raise ValueError("Missing observations cannot be zero-filled")
    return {
        "defect": actual - additive_read,
        "reading": additive_state - additive_read,
        "encoding": actual - additive_state,
    }


def cold_scores(observations, counts, owners):
    """Integrate qualified observations with explicit counts; reject other actors."""
    obs, counts, owners = (
        np.asarray(observations, float),
        np.asarray(counts),
        np.asarray(owners),
    )
    if (
        obs.ndim != 4
        or obs.shape[1] != 4
        or counts.shape != (2, 4)
        or owners.shape != (obs.shape[0], 4)
        or not np.all(owners == 1)
        or not np.isfinite(obs).all()
        or not np.isfinite(counts).all()
        or np.any(counts <= 0)
        or np.any(counts != np.floor(counts))
    ):
        raise ValueError(
            "A-qualified four-event observations and positive integer counts required"
        )
    return np.einsum("lc,pcft->plft", counts / counts.sum(axis=1, keepdims=True), obs)


def rms(a):
    """Report declared observation units without converting them to accuracy."""
    a = np.asarray(a, float)
    return float(np.sqrt(np.mean(a * a)))
