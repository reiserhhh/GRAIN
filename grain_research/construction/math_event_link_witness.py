#!/usr/bin/env python3
"""Keep continuous endpoint composition and event links in one target witness."""

from __future__ import annotations

import numpy as np

KERNELS = ("linear", "euclidean_distance")
COMPONENTS = ("PP", "PR", "RP", "RR")


def observations(vectors, conditions):
    """Validate finite author windows, without estimating a label-based encoder."""
    x, c = np.asarray(vectors, float), np.asarray(conditions)
    if (
        x.ndim != 3
        or x.shape[1] < 3
        or x.shape[2] < 1
        or c.shape != x.shape[:2]
        or not np.isfinite(x).all()
        or not np.isin(c, (0, 1)).all()
    ):
        raise ValueError("Finite windows and addressed binary conditions required")
    return x, c


def node_kernels(x, c, y, d):
    """Evaluate two fixed kernels on vector plus orthogonal condition one-hot."""
    x, c, y, d = (
        np.asarray(x, float),
        np.asarray(c),
        np.asarray(y, float),
        np.asarray(d),
    )
    if (
        x.ndim != 2
        or y.ndim != 2
        or x.shape[1] != y.shape[1]
        or c.shape != (len(x),)
        or d.shape != (len(y),)
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or not np.isin(c, (0, 1)).all()
        or not np.isin(d, (0, 1)).all()
    ):
        raise ValueError("Aligned finite node vectors and conditions required")
    linear = x @ y.T + (c[:, None] == d[None, :])
    nx = np.einsum("ij,ij->i", x, x) + 1
    ny = np.einsum("ij,ij->i", y, y) + 1
    dist2 = nx[:, None] + ny[None, :] - 2 * linear
    if np.min(dist2, initial=0) < -1e-10:
        raise ArithmeticError("Invalid Euclidean squared distance")
    if np.array_equal(x, y) and np.array_equal(c, d):
        np.fill_diagonal(dist2, 0)
    energy = (
        np.sqrt(nx)[:, None] + np.sqrt(ny)[None, :] - np.sqrt(np.maximum(dist2, 0))
    ) / 2
    return np.stack((linear, energy))


def pair_accounts(left, right):
    """Contract P/R source and observation measures, retaining all signed terms."""
    if left.shape != right.shape or left.ndim != 5:
        raise ValueError("Kernel by author by author by endpoint by endpoint required")
    jj = np.mean(left * right, axis=(-2, -1))
    jp = np.mean(left.mean(-1) * right.mean(-1), axis=-1)
    pj = np.mean(left.mean(-2) * right.mean(-2), axis=-1)
    pp = left.mean((-2, -1)) * right.mean((-2, -1))
    return np.stack((pp, pj - pp, jp - pp, jj - jp - pj + pp), axis=1)


def grams(x, c, y, d, block=24, after_x=None, after_y=None, guard=None):
    """Evaluate full event-link Grams in bounded blocks, not a mean-vector kernel."""
    x, c = observations(x, c)
    y, d = observations(y, d)
    if x.shape[-1] != y.shape[-1] or block < 1:
        raise ValueError("Same vector space and positive block size required")
    ax = (
        np.tile(np.arange(1, x.shape[1]), (len(x), 1))
        if after_x is None
        else np.asarray(after_x)
    )
    ay = (
        np.tile(np.arange(1, y.shape[1]), (len(y), 1))
        if after_y is None
        else np.asarray(after_y)
    )
    for a, z in ((ax, x), (ay, y)):
        if (
            a.shape != (len(z), z.shape[1] - 1)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(np.sort(a, axis=1) != np.arange(1, z.shape[1]))
        ):
            raise ValueError("After endpoints must retain their complete multiset")
    out = np.empty((2, 4, len(x), len(y)))
    for start in range(0, len(x), block):
        stop = min(start + block, len(x))
        if guard is not None:
            guard()
        xx, cc = x[start:stop], c[start:stop]
        m, n = x.shape[1], y.shape[1]
        full = node_kernels(
            xx.reshape(-1, x.shape[-1]),
            cc.ravel(),
            y.reshape(-1, y.shape[-1]),
            d.ravel(),
        )
        full = full.reshape(2, len(xx), m, len(y), n).transpose(0, 1, 3, 2, 4)
        left = full[..., : m - 1, : n - 1]
        right = np.take_along_axis(
            full, ax[start:stop][None, :, None, :, None], axis=-2
        )
        right = np.take_along_axis(right, ay[None, None, :, None, :], axis=-1)
        out[:, :, start:stop] = pair_accounts(left, right)
    return out


def target_weights(codes):
    """Compile all five equal-author binary-group contrasts, with no target selection."""
    g = np.asarray(codes)
    if (
        g.ndim != 1
        or not np.issubdtype(g.dtype, np.integer)
        or np.any((g < 0) | (g > 31))
    ):
        raise ValueError("Complete inherited five-bit source profiles required")
    weights = []
    for q in range(5):
        high = ((g >> q) & 1).astype(bool)
        if not high.any() or high.all():
            raise ValueError("Both source target groups must be observed")
        weights.append(high / high.sum() - (~high) / (~high).sum())
    return np.asarray(weights)


def relink(conditions):
    """Audit vector pairing while fixing endpoint bags and condition-pair counts."""
    c = np.asarray(conditions)
    if c.ndim != 2 or c.shape[1] < 3 or not np.isin(c, (0, 1)).all():
        raise ValueError("Addressed binary condition sequences required")
    out = np.tile(np.arange(1, c.shape[1]), (len(c), 1))
    strata = 2 * c[:, :-1] + c[:, 1:]
    for u, row in enumerate(strata):
        for key in range(4):
            positions = np.flatnonzero(row == key)
            out[u, positions] = np.roll(out[u, positions], 1)
    return out


def canonical_window(record, dimensions):
    """Bind one real window to author, unique event, position and publication time."""
    if (
        set(record) != {"record_id", "author", "regime", "events"}
        or record["regime"] != "raw_pandora_publication"
    ):
        raise ValueError("Only the declared publication observation regime allowed")
    author = record["author"]
    if not isinstance(author, str) or not author or len(record["events"]) != 12:
        raise ValueError("One addressed twelve-event author window required")
    seen, rows = set(), []
    for e in record["events"]:
        if set(e) != {
            "event_id",
            "owner",
            "position",
            "timestamp",
            "vector",
            "condition",
        }:
            raise ValueError(
                "Labels, native answers and undeclared event fields rejected"
            )
        if (
            not isinstance(e["event_id"], str)
            or not e["event_id"]
            or e["event_id"] in seen
            or e["owner"] != author
        ):
            raise ValueError("Distinct events belonging to this author required")
        seen.add(e["event_id"])
        if not isinstance(e["position"], int) or e["condition"] not in (0, 1):
            raise ValueError("Known original positions and conditions required")
        v = np.asarray(e["vector"], float)
        if (
            v.shape != (dimensions,)
            or not np.isfinite(v).all()
            or not np.isfinite(e["timestamp"])
        ):
            raise ValueError("Finite registered vectors and timestamps required")
        rows.append(e)
    rows.sort(key=lambda e: e["timestamp"])
    if np.any(np.diff([e["timestamp"] for e in rows]) <= 0) or np.any(
        np.diff([e["position"] for e in rows]) != 1
    ):
        raise ValueError("Strict adjacent attested publication events required")
    return (
        np.asarray([e["vector"] for e in rows]),
        np.asarray([e["condition"] for e in rows]),
        [e["event_id"] for e in rows],
    )


def cold_read(source_vectors, source_conditions, weights, record):
    """Return four retained target accounts, plus a clearly marked relation audit."""
    x, c, ids = canonical_window(record, source_vectors.shape[-1])
    readings = []
    for after in (None, relink(c[None])):
        g = grams(source_vectors, source_conditions, x[None], c[None], after_y=after)
        readings.append(np.einsum("kcis,qi->kcqs", g, weights)[..., 0])
    a, b = readings
    return {
        "record_id": record["record_id"],
        "event_ids": ids,
        "attested_accounts": a.tolist(),
        "relinked_audit_accounts": b.tolist(),
        "attested_total": a.sum(axis=1).tolist(),
        "relinked_audit_total": b.sum(axis=1).tolist(),
        "changed_after_slots": int(
            np.count_nonzero(relink(c[None])[0] != np.arange(1, 12))
        ),
        "audit_is_new_human_text": False,
    }
