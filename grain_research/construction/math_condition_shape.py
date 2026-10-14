#!/usr/bin/env python3
"""Conditional covariance contrasts retaining mean, choice, drift and shape."""

import numpy as np

from . import math_condition_pairing as P

VERSION = "HM187_condition_shape_v1"
KINDS = ("pooled", "raw_pairing", "drift", "shape")


def moments(vectors, conditions):
    """Return finite empirical covariance accounts, not causal response estimates."""
    z, c = np.asarray(vectors, float), np.asarray(conditions)
    first = P.moments(z, c)
    t = z.shape[-2]
    centered = z - first["mean"][..., None, :]
    pooled = np.einsum("...ti,...tj->...ij", centered, centered) / t
    raw = (
        np.einsum(
            "...t,...ti,...tj->...ij", c - first["p"][..., None], centered, centered
        )
        / t
    )
    factor = np.zeros_like(first["p"])
    np.divide(
        1 - 2 * first["p"],
        first["opportunity"],
        out=factor,
        where=first["support"],
    )
    drift = factor[..., None, None] * np.einsum(
        "...i,...j->...ij", first["covariance"], first["covariance"]
    )
    shape = raw - drift
    counts = np.stack(((c == 0).sum(-1), (c == 1).sum(-1)), axis=-1)
    delta = np.full_like(shape, np.nan)
    np.divide(
        shape,
        first["opportunity"][..., None, None],
        out=delta,
        where=first["support"][..., None, None],
    )
    return first | {
        "operators": np.stack((pooled, raw, drift, shape), axis=-3),
        "conditional_shape_delta": delta,
        "counts": counts,
        "dispersion_support": (counts >= 2).all(-1),
    }


def fields(operators, labels, folds=None):
    """Construct all source-target contrasts with unchanged author-balanced groups."""
    x = np.asarray(operators, float)
    if x.ndim != 5 or x.shape[1:3] != (2, 4) or x.shape[-1] != x.shape[-2]:
        raise ValueError("Author/two-view/four-kind complete square operators required")
    if not np.isfinite(x).all() or len(x) != len(labels):
        raise ValueError("Aligned finite source observations required")
    w, counts = P.target_weights(labels, folds)
    result = np.einsum("bqu,uvkij->bkvqij", w, x)
    return {"fields": result, "weights": w, "counts": counts}


def statistics(bank):
    """Retain complete crossed-author/view operator inner products, including signs."""
    f = np.asarray(bank, float)
    if f.ndim != 6 or f.shape[:4] != (2, 4, 2, 5) or not np.isfinite(f).all():
        raise ValueError("Two folds/four kinds/two views/five targets required")
    rows = []
    for k, name in enumerate(KINDS):
        for q in list(range(5)) + [None]:
            ids = [q] if q is not None else list(range(5))
            a, b = f[0, k][:, ids], f[1, k][:, ids]
            cross = float(np.sum(a.mean(0) * b.mean(0)))
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            rows.append(
                {
                    "kind": name,
                    "target": P.TARGETS[q] if q is not None else "all",
                    "signed_product": cross,
                    "coherence": float(2 * cross / denom) if denom else None,
                    "same_view_crossauthor": float(np.sum(a * b) / 2),
                    "different_view_crossauthor": float(np.sum(a * b[::-1]) / 2),
                    "within_fold_crossview": float(
                        (np.sum(a[0] * a[1]) + np.sum(b[0] * b[1])) / 2
                    ),
                }
            )
    return rows


def author_comparison(shape, late_counts, permutation=None):
    """Compare self with exact-count-matched other authors; keep missing matches."""
    x, n = np.asarray(shape, float), np.asarray(late_counts)
    if x.ndim != 4 or x.shape[1] != 2 or n.shape != (len(x),):
        raise ValueError("Two-view author operators and aligned late counts required")
    gram = x[:, 0].reshape(len(x), -1) @ x[:, 1].reshape(len(x), -1).T
    match = (n[:, None] == n[None, :]) & ~np.eye(len(x), dtype=bool)
    supported = match.sum(1) > 0
    stranger = np.full(len(x), np.nan)
    stranger[supported] = (
        np.sum(gram * match, axis=1)[supported] / match.sum(1)[supported]
    )
    order = np.arange(len(x)) if permutation is None else np.asarray(permutation)
    if (
        order.shape != n.shape
        or sorted(order.tolist()) != list(range(len(n)))
        or not np.array_equal(n[order], n)
    ):
        raise ValueError("Late author permutation must preserve exact condition counts")
    self_values = gram[np.arange(len(n)), order]
    primary = (
        float(np.mean((self_values - stranger)[supported])) if supported.any() else None
    )
    return {
        "primary": primary,
        "self": self_values,
        "stranger": stranger,
        "support": supported,
    }


def permute_authors(late_counts, rng):
    """Shuffle late identities within count cells, preserving unsupported singletons."""
    n = np.asarray(late_counts)
    if n.ndim != 1:
        raise ValueError("One count per author required")
    order = np.arange(len(n))
    for count in np.unique(n):
        ids = np.flatnonzero(n == count)
        order[ids] = rng.permutation(ids)
    return order


def cold_record(bank, basis, mapping, record):
    """Score addressed numerical events without source labels or native answers."""
    f, q, a = (
        np.asarray(bank, float),
        np.asarray(basis, float),
        np.asarray(mapping, float),
    )
    if f.shape != (4, 2, 5, 15, 15) or q.shape != (1024, 15) or a.shape != q.shape:
        raise ValueError("Complete fixed source bundle required")
    if not all(np.isfinite(v).all() for v in (f, q, a)):
        raise ValueError("Finite bundle required")
    if set(record) != {"frame_key", "events"} or not isinstance(
        record["frame_key"], str
    ):
        raise ValueError("Addressed numerical observations only")
    ids, slots, vectors, conditions = [], [], [], []
    for e in record["events"]:
        if set(e) != {"event_id", "slot", "vector", "condition"}:
            raise ValueError("No text,labels,answers or other metadata allowed")
        if (
            not isinstance(e["event_id"], str)
            or type(e["slot"]) is not int
            or e["slot"] not in range(12)
            or type(e["condition"]) is not int
            or e["condition"] not in (0, 1)
        ):
            raise ValueError("Original event IDs,slots and binary conditions required")
        ids.append(e["event_id"])
        slots.append(e["slot"])
        vectors.append(e["vector"])
        conditions.append(e["condition"])
    if not ids or len(ids) != len(set(ids)) or len(slots) != len(set(slots)):
        raise ValueError("Nonempty unique event identities and slots required")
    h, c = np.asarray(vectors, float), np.asarray(conditions, int)
    if h.shape != (len(ids), 1024):
        raise ValueError("Original1024-coordinate events required")
    m = moments(h @ q, c)
    old = P.moments(h @ a, c)
    return {
        "frame_key": record["frame_key"],
        "count": len(ids),
        "counts": m["counts"].tolist(),
        "p": float(m["p"]),
        "opportunity": float(m["opportunity"]),
        "conditional_support": bool(m["support"]),
        "dispersion_support": bool(m["dispersion_support"]),
        "original_GRAIN_mean": old["mean"].tolist(),
        "original_GRAIN_mean_delta": old["delta"].tolist()
        if bool(old["support"])
        else None,
        "readings": {
            name: np.einsum("kvqij,ij->kvq", f, m["operators"][i]).tolist()
            for i, name in enumerate(KINDS)
        },
        "conditional_shape_delta": np.einsum(
            "kvqij,ij->kvq", f, m["conditional_shape_delta"]
        ).tolist()
        if bool(m["support"])
        else None,
    }
