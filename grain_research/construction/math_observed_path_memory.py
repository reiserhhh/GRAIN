#!/usr/bin/env python3
"""Qualified finite path laws distinguish mixture from target-specific memory."""

from __future__ import annotations

import numpy as np

from grain_research.construction.math_joint_meaning import homogeneous

MODELS = (
    "profile_markov",
    "author_markov_mixture",
    "common_path_association",
    "observed_path",
)


def triple_mass(path, states=8):
    """Count overlapping triples; overlap never creates independent subjects."""
    x = np.asarray(path)
    if (
        x.ndim != 1
        or len(x) < 3
        or not np.issubdtype(x.dtype, np.integer)
        or np.any((x < 0) | (x >= states))
    ):
        raise ValueError("At least three in-range integer observations required")
    out = np.zeros((states, states, states))
    np.add.at(out, (x[:-2], x[1:-1], x[2:]), 1 / (len(x) - 2))
    return out


def markov_table(table):
    """Preserve left/right adjacent margins under conditional A,C independence."""
    f = np.asarray(table, float)
    if f.ndim != 4 or not np.isfinite(f).all() or np.any(f < 0):
        raise ValueError("Nonnegative group-by-A-by-B-by-C table required")
    left, right, middle = f.sum(3), f.sum(1), f.sum((1, 3))
    return np.divide(
        left[..., None] * right[:, None],
        middle[:, None, :, None],
        out=np.zeros_like(f),
        where=middle[:, None, :, None] > 0,
    )


def aggregate(mass, targets):
    """Average equally weighted authors within their measured full profiles."""
    p, y = np.asarray(mass, float), np.asarray(targets)
    if (
        p.ndim != 4
        or len(p) == 0
        or len(set(p.shape[1:])) != 1
        or not np.isfinite(p).all()
        or np.any(p < 0)
        or not np.allclose(p.sum((1, 2, 3)), 1, rtol=0, atol=1e-12)
        or y.shape != (len(p), 5)
        or not np.isin(y, (0, 1)).all()
    ):
        raise ValueError(
            "Aligned normalized author masses and five binary targets required"
        )
    profiles, ix = np.unique(y, axis=0, return_inverse=True)
    full = np.zeros((len(profiles), *p.shape[1:]))
    np.add.at(full, ix, p / len(p))
    return profiles.astype(int), ix, full


def construct(mass, targets, tolerance=1e-10, max_sweeps=10000):
    """Build four fixed observation laws without fitting answers or smoothing."""
    profiles, ix, full = aggregate(mass, targets)
    individual = markov_table(mass)
    mixture = np.zeros_like(full)
    np.add.at(mixture, ix, individual / len(mass))
    common = np.zeros_like(full)
    receipts = []
    for b in range(full.shape[2]):
        common[:, :, b, :], receipt = homogeneous(
            full[:, :, b, :], tolerance, max_sweeps
        )
        receipts.append({"middle_state": b, **receipt})
    return profiles, np.stack([markov_table(full), mixture, common, full]), receipts


def fields(profiles, laws):
    """Contract all five targets,leaving unsupported observations missing."""
    profiles, laws = np.asarray(profiles), np.asarray(laws, float)
    denominator = laws.sum(1)
    positive = np.stack([laws[:, profiles[:, q] == 1].sum(1) for q in range(5)], axis=1)
    return np.divide(
        positive,
        denominator[:, None],
        out=np.full_like(positive, np.nan),
        where=denominator[:, None] > 0,
    )


def divergence(p, q):
    """Compute a finite KL in nats without clipping arithmetic residues."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    active = p > 0
    if np.any(q[active] <= 0):
        raise ValueError("Reference omits positive observed mass")
    return float(np.sum(p[active] * np.log(p[active] / q[active])))


def margin_errors(laws):
    """Audit the actual adjacent-profile and common pooled-path contracts."""
    full, common = laws[3], laws[2]
    adjacent = max(
        float(np.max(abs(laws[k].sum(axis) - full.sum(axis))))
        for k in range(4)
        for axis in (1, 3)
    )
    pooled = float(np.max(abs(common.sum(0) - full.sum(0))))
    return {"adjacent_profile_max": adjacent, "common_pooled_triple_max": pooled}


def recurrence(left, right, weights):
    """Retain signed field products and actual support,not a prediction metric."""
    left, right, weights = map(np.asarray, (left, right, weights))
    active = np.isfinite(left).all(0) & np.isfinite(right).all(0) & (weights > 0)
    if not active.any():
        return np.full(5, np.nan), np.full(5, np.nan), 0.0, 0
    w = weights[active]
    x, y = left[:, active], right[:, active]
    inner = np.sum(w * x * y, axis=1)
    norm = np.sqrt(np.sum(w * x * x, axis=1) * np.sum(w * y * y, axis=1))
    cosine = np.divide(inner, norm, out=np.full(5, np.nan), where=norm > 0)
    return inner, cosine, float(w.sum()), int(active.sum())


def cold_record(readout, record):
    """Read identified ordered events without labels,answers or source identities."""
    if set(record) != {"author_key", "events"}:
        raise ValueError("Only author key and observed events allowed")
    e = record["events"]
    if len(e) < 3 or any(set(v) != {"event_id", "cell", "timestamp"} for v in e):
        raise ValueError("At least three qualified events required")
    ids = [v["event_id"] for v in e]
    times = np.asarray([v["timestamp"] for v in e], float)
    if (
        any(not isinstance(v, str) or not v for v in ids)
        or len(set(ids)) != len(ids)
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError("Distinct identities and strict chronology required")
    cells = np.asarray([v["cell"] for v in e])
    triple_mass(cells, readout.shape[-1])
    values = readout[..., cells[:-2], cells[1:-1], cells[2:]]
    supported = np.isfinite(values).all(-1)
    with np.errstate(invalid="ignore"):
        score = values.mean(-1)
    score[~supported] = np.nan
    return {
        "author_key": record["author_key"],
        "unique_events": len(e),
        "overlapping_triples": len(e) - 2,
        "supported": supported.tolist(),
        "coverage": np.isfinite(values).mean(-1).tolist(),
        "scores": np.where(np.isfinite(score), score, None).tolist(),
    }
