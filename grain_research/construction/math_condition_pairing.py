#!/usr/bin/env python3
"""Finite condition-expression pairing, retaining level, choice and opportunity."""

import numpy as np

VERSION = "HM186_condition_pairing_v1"
TARGETS = ("E", "N", "A", "C", "O")
KINDS = ("level", "pairing")
SPACES = ("full", "retained", "complement")


def moments(vectors, conditions):
    """Return observed moments; absent conditional contrasts stay explicitly missing."""
    h, c = np.asarray(vectors, float), np.asarray(conditions)
    if (
        h.ndim < 2
        or h.shape[:-1] != c.shape
        or h.shape[-2] == 0
        or not np.isfinite(h).all()
        or not np.isin(c, (0, 1)).all()
    ):
        raise ValueError(
            "Nonempty finite vectors and aligned binary conditions required"
        )
    mean, p = h.mean(-2), c.mean(-1)
    mixed = np.einsum("...t,...td->...d", c, h) / h.shape[-2]
    covariance = np.einsum("...t,...td->...d", c - p[..., None], h) / h.shape[-2]
    support = (p > 0) & (p < 1)
    opportunity = p * (1 - p)
    delta = np.full_like(mean, np.nan)
    np.divide(
        covariance,
        opportunity[..., None],
        out=delta,
        where=support[..., None],
    )
    return {
        "mean": mean,
        "p": p,
        "mixed": mixed,
        "covariance": covariance,
        "opportunity": opportunity,
        "support": support,
        "delta": delta,
    }


def target_weights(labels, folds=None):
    """Construct signed natural-marginal group means, not fitted output weights."""
    y = np.asarray(labels)
    if y.ndim != 2 or y.shape[1] != 5 or not np.isin(y, (0, 1)).all():
        raise ValueError("Five inherited binary target labels required")
    if folds is not None:
        folds = np.asarray(folds)
        if folds.shape != (len(y),) or set(folds.tolist()) != {0, 1}:
            raise ValueError("Two fixed disjoint author folds required")
    groups = [np.ones(len(y), bool)] if folds is None else [folds == b for b in (0, 1)]
    weights, counts = [], []
    for group in groups:
        w, n = np.zeros((5, len(y))), np.empty((5, 2), int)
        for q in range(5):
            for value in (0, 1):
                ids = group & (y[:, q] == value)
                n[q, value] = ids.sum()
                if n[q, value] == 0:
                    raise ValueError("Missing trait side; no smoothing or imputation")
                w[q, ids] = (2 * value - 1) / n[q, value]
        weights.append(w)
        counts.append(n)
    return np.asarray(weights), np.asarray(counts)


def source_fields(mean, covariance, labels, folds=None):
    """Keep both observed level and within-author pairing under identical target groups."""
    mean, covariance = np.asarray(mean, float), np.asarray(covariance, float)
    if (
        mean.ndim != 3
        or mean.shape != covariance.shape
        or len(mean) != len(labels)
        or not np.isfinite(mean).all()
        or not np.isfinite(covariance).all()
    ):
        raise ValueError("Finite aligned author/view vector moments required")
    weights, counts = target_weights(labels, folds)
    x = np.stack((mean, covariance), axis=1)
    fields = (weights.reshape(-1, len(mean)) @ x.reshape(len(mean), -1)).reshape(
        len(weights), 5, 2, mean.shape[1], mean.shape[2]
    )
    fields = fields.transpose(0, 2, 3, 1, 4)
    return {"fields": fields, "weights": weights, "counts": counts}


def statistics(fields):
    """Compare crossed author/text families; signed products are never clipped."""
    f = np.asarray(fields, float)
    if f.ndim != 6 or f.shape[:5] != (2, 2, 2, 5, 3):
        raise ValueError("Two folds/kinds/views,five targets,three spaces required")
    if not np.isfinite(f).all():
        raise ValueError("Finite source fields required")
    rows = []
    for kind, name in enumerate(KINDS):
        for q in list(range(5)) + [None]:
            selected = [q] if q is not None else list(range(5))
            for s, space in enumerate(SPACES):
                a = f[0, kind][:, selected, s]
                b = f[1, kind][:, selected, s]
                cross = float(np.sum(a.mean(0) * b.mean(0)))
                denominator = np.linalg.norm(a) * np.linalg.norm(b)
                rows.append(
                    {
                        "kind": name,
                        "target": TARGETS[q] if q is not None else "all",
                        "space": space,
                        "signed_inner_product": cross,
                        "cross_author_coherence": float(2 * cross / denominator)
                        if denominator
                        else None,
                        "same_view_cross_author": float(np.sum(a * b) / 2),
                        "different_view_cross_author": float(np.sum(a * b[::-1]) / 2),
                        "within_fold_cross_view": float(
                            (np.sum(a[0] * a[1]) + np.sum(b[0] * b[1])) / 2
                        ),
                    }
                )
    return rows


def permute_profiles(labels, folds, rng):
    """Shuffle whole five-label assignments within fixed source author folds."""
    target_weights(labels, folds)
    result = np.asarray(labels).copy()
    for fold in (0, 1):
        ids = np.flatnonzero(folds == fold)
        result[ids] = result[rng.permutation(ids)]
    return result


def relink_conditions(conditions, rng):
    """Relink only condition addresses, preserving each author's exact condition count."""
    c = np.asarray(conditions)
    if c.ndim < 1 or c.shape[-1] == 0 or not np.isin(c, (0, 1)).all():
        raise ValueError("Nonempty binary condition rows required")
    order = np.argsort(rng.random(c.shape), axis=-1)
    return np.take_along_axis(c, order, axis=-1)


def cold_record(fields, record):
    """Read numerical events only, with missing opportunity distinct from neutral scores."""
    f = np.asarray(fields, float)
    if f.shape != (2, 2, 5, 3, 1024) or not np.isfinite(f).all():
        raise ValueError("Complete frozen two-kind source bank required")
    if set(record) != {"frame_key", "events"} or not isinstance(
        record["frame_key"], str
    ):
        raise ValueError("Addressed numerical observations only")
    ids, slots, h, c = [], [], [], []
    for event in record["events"]:
        if set(event) != {"event_id", "slot", "vector", "condition"}:
            raise ValueError("No text,labels,answers or other metadata allowed")
        if (
            not isinstance(event["event_id"], str)
            or type(event["slot"]) is not int
            or event["slot"] not in range(12)
            or type(event["condition"]) is not int
            or event["condition"] not in (0, 1)
        ):
            raise ValueError(
                "Original event identity/slot and binary condition required"
            )
        ids.append(event["event_id"])
        slots.append(event["slot"])
        h.append(event["vector"])
        c.append(event["condition"])
    if len(ids) != len(set(ids)) or len(slots) != len(set(slots)):
        raise ValueError("Repeated event identity or slot")
    if not h:
        return {"frame_key": record["frame_key"], "count": 0, "readings": None}
    h, c = np.asarray(h, float), np.asarray(c, int)
    if h.shape != (len(ids), 1024):
        raise ValueError("Original1024-coordinate observations required")
    m = moments(h, c)
    readings = {
        name: np.einsum("kvqsd,d->kvqs", f, m[name]).tolist()
        for name in ("mean", "mixed", "covariance")
    }
    readings["conditional_delta"] = (
        np.einsum("kvqsd,d->kvqs", f, m["delta"]).tolist()
        if bool(m["support"])
        else None
    )
    return {
        "frame_key": record["frame_key"],
        "count": len(ids),
        "condition_counts": [int(np.sum(c == value)) for value in (0, 1)],
        "p": float(m["p"]),
        "opportunity": float(m["opportunity"]),
        "delta_support": bool(m["support"]),
        "readings": readings,
    }
