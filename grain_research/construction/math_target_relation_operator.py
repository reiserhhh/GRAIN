#!/usr/bin/env python3
"""Source-target bilinear witnesses, preserving within/between profile structure."""

from __future__ import annotations

import numpy as np

TARGETS = ("common", "E", "N", "A", "C", "O")
COMPONENTS = ("total", "within_profile", "between_profile")


def validate_source(means, codes):
    """Require finite paired author vectors and complete inherited profile codes."""
    x, c = np.asarray(means, float), np.asarray(codes)
    if (
        x.ndim != 3
        or x.shape[1] != 2
        or x.shape[0] < 4
        or x.shape[2] < 1
        or c.shape != (len(x),)
        or c.dtype.kind not in "iu"
        or not np.isin(c, np.arange(32)).all()
        or not np.isfinite(x).all()
    ):
        raise ValueError("Paired finite author vectors and five-bit codes required")
    return x, c


def centered_block(indices, size):
    """Embed one empirical centered sum, including zero singleton sums."""
    ix = np.asarray(indices, int)
    if not len(ix):
        raise ValueError("An empty group is not a zero observation")
    out = np.zeros((size, size))
    out[np.ix_(ix, ix)] = np.eye(len(ix)) - 1 / len(ix)
    return out


def compile_bank(means, codes):
    """Compile target covariance contrasts without fitting an outcome predictor."""
    x, c = validate_source(means, codes)
    n = len(x)
    h = np.zeros((3, 6, n, n))
    groups = [(0, np.arange(n), 1)]
    counts = np.empty((5, 2), int)
    for q in range(5):
        for a in (0, 1):
            ix = np.flatnonzero(((c >> q) & 1) == a)
            counts[q, a] = len(ix)
            if len(ix) < 2:
                raise ValueError("Each binary target group needs at least two authors")
            groups.append((q + 1, ix, 2 * a - 1))
    for q, ix, sign in groups:
        weight = sign / (len(ix) - 1)
        h[0, q] += weight * centered_block(ix, n)
        for g in np.unique(c[ix]):
            sub = ix[c[ix] == g]
            h[1, q] += weight * centered_block(sub, n)
    h[2] = h[0] - h[1]
    return {
        "means": x,
        "codes": c,
        "dual": h,
        "reference": x.mean(axis=0),
        "binary_counts": counts,
        "profile_counts": np.bincount(c, minlength=32),
    }


def operator_gram(left, right):
    """Keep all signed Hilbert-Schmidt contractions without ambient operators."""
    a, b = left["means"], right["means"]
    if a.shape[2] != b.shape[2]:
        raise ValueError("Both operators need the same ambient vector coordinates")
    ka, kb = a[:, 0] @ b[:, 0].T, a[:, 1] @ b[:, 1].T
    hl, hr = left["dual"], right["dual"]
    result = np.empty((3, 6, 3, 6))
    for j in range(3):
        for r in range(6):
            transported = ka @ hr[j, r] @ kb.T
            result[:, :, j, r] = np.einsum("mqij,ij->mq", hl, transported)
    return result


def cosine_gram(cross, self_left, self_right):
    """Normalize descriptive operator products; absent norms remain missing."""
    nl = np.array([[self_left[m, q, m, q] for q in range(6)] for m in range(3)])
    nr = np.array([[self_right[m, q, m, q] for q in range(6)] for m in range(3)])
    if np.any(nl < -1e-12) or np.any(nr < -1e-12):
        raise ValueError("Negative squared operator norm beyond numerical tolerance")
    denominator = np.sqrt(np.maximum(nl, 0)[:, :, None, None]) * np.sqrt(
        np.maximum(nr, 0)[None, None]
    )
    return np.divide(
        cross, denominator, out=np.full_like(cross, np.nan), where=denominator > 0
    )


def cyclic_relink(codes):
    """Exchange B ownership only inside the same complete inherited profile."""
    c = np.asarray(codes)
    permutation = np.arange(len(c))
    for g in np.unique(c):
        ix = np.flatnonzero(c == g)
        permutation[ix] = np.roll(ix, 1)
    return permutation


def score_points(bank, points):
    """Evaluate fixed-reference pair witnesses, not personality probabilities."""
    p = np.asarray(points, float)
    if (
        p.ndim < 2
        or p.shape[-2:] != bank["reference"].shape
        or not np.isfinite(p).all()
    ):
        raise ValueError(
            "Finite paired window means in the source coordinates required"
        )
    flat = p.reshape(-1, *p.shape[-2:])
    centered = flat - bank["reference"]
    left = centered[:, 0] @ bank["means"][:, 0].T
    right = centered[:, 1] @ bank["means"][:, 1].T
    result = np.einsum("ai,mqij,aj->amq", left, bank["dual"], right, optimize=True)
    return result.reshape(*p.shape[:-2], 3, 6)


def cold_point(record, dimension):
    """Aggregate disjoint addressed events without outcomes or batch context."""
    if (
        set(record) != {"frame_key", "windows"}
        or not isinstance(record["frame_key"], str)
        or len(record["windows"]) != 2
    ):
        raise ValueError("Only frame identity and two numeric event windows allowed")
    seen, points, addresses = set(), [], []
    for window in record["windows"]:
        if not window:
            raise ValueError("An unobserved window is not a zero vector")
        vectors, ids = [], []
        for event in window:
            if set(event) != {"event_id", "vector", "condition"}:
                raise ValueError(
                    "Only addressed vector and inherited condition allowed"
                )
            identity, vector = event["event_id"], np.asarray(event["vector"], float)
            if (
                not isinstance(identity, str)
                or identity in seen
                or vector.shape != (dimension,)
                or not np.isfinite(vector).all()
                or type(event["condition"]) is not int
                or event["condition"] not in (0, 1)
            ):
                raise ValueError("Distinct finite evidence events required")
            seen.add(identity)
            vectors.append(vector)
            ids.append(identity)
        points.append(np.mean(vectors, axis=0))
        addresses.append(ids)
    return np.asarray(points), addresses


def cold_record(banks, record):
    """Return both source instruments and all components, with event identity."""
    point, addresses = cold_point(record, banks[0]["means"].shape[2])
    return {
        "frame_key": record["frame_key"],
        "event_ids": addresses,
        "targets": list(TARGETS),
        "components": list(COMPONENTS),
        "scores": [score_points(bank, point).tolist() for bank in banks],
        "units": "bilinear_covariance_witness_NOT_probability_or_trait_level",
    }
