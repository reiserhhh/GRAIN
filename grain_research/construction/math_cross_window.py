#!/usr/bin/env python3
"""Author-qualified two-window laws and their retained signed dependence."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

VERSION = "HM176_cross_window_observation_v1"
MODELS = ("profile_product", "other_author_same_profile", "same_author")


def code_hash():
    """Bind the frozen observation constructor rather than an optimized probe."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def validate(p, y):
    """Require two distributions and observed binary labels only in SOURCE."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    if (
        p.ndim != 3
        or p.shape[1:] != (2, 8)
        or y.shape != (len(p), 5)
        or not np.isfinite(p).all()
        or np.any(p < 0)
        or not np.isfinite(y).all()
        or not np.allclose(p.sum(2), 1, rtol=0, atol=2e-12)
        or np.any((y != 0) & (y != 1))
        or np.any(y.sum(0) == 0)
        or np.any(y.sum(0) == len(y))
    ):
        raise ValueError("Invalid SOURCE observation law")
    _, inverse, number = np.unique(y, axis=0, return_inverse=True, return_counts=True)
    if number.min() < 2:
        raise ValueError("Distinct-author comparison needs two authors per profile")
    return p, y.astype(int), inverse, number


def source_laws(p, y):
    """Construct all three empirical laws without assuming within-window iid."""
    p, y, c, number = validate(p, y)
    means = np.array([p[c == i].mean(0) for i in range(len(number))])
    other = np.empty((len(p), 8))
    for u in range(len(p)):
        idx = (c == c[u]) & (np.arange(len(p)) != u)
        other[u] = p[idx, 1].mean(0)
    laws = np.empty((3, 5, 2, 8, 8))
    for q in range(5):
        for g in (0, 1):
            idx = y[:, q] == g
            laws[0, q, g] = means[c[idx], 0].T @ means[c[idx], 1] / len(p)
            laws[1, q, g] = p[idx, 0].T @ other[idx] / len(p)
            laws[2, q, g] = p[idx, 0].T @ p[idx, 1] / len(p)
    return laws


def signed(x, prior):
    """Contrast target groups in their own units, retaining negative entries."""
    prior = np.asarray(prior)
    return x[:, 1] / prior[:, None, None] - x[:, 0] / (1 - prior[:, None, None])


def centered_accounts(p, y):
    """Separate common-window configuration and change by exact covariance."""
    p, y, c, number = validate(p, y)
    means = np.array([p[c == i].mean(0) for i in range(len(number))])
    x = p - means[c]
    m, d = x.mean(1), (x[:, 1] - x[:, 0]) / 2
    pairs = {
        "v0": (x[:, 0], x[:, 0]),
        "v1": (x[:, 1], x[:, 1]),
        "cross": (x[:, 0], x[:, 1]),
        "common": (m, m),
        "change": (d, d),
    }
    out = {}
    for name, (a, b) in pairs.items():
        out[name] = np.array(
            [
                [a[y[:, q] == g].T @ b[y[:, q] == g] / len(p) for g in (0, 1)]
                for q in range(5)
            ]
        )
    out["cross_symmetric"] = (out["cross"] + out["cross"].swapaxes(-1, -2)) / 2
    return out


def condition_accounts(same, other):
    """Keep choice, response and unsupported conditional blocks in DeltaJ."""
    same, other = np.asarray(same), np.asarray(other)
    if same.shape != other.shape or same.shape[-2:] != (8, 8):
        raise ValueError("Eight-cell joint laws required")
    a, b = (
        same.reshape(*same.shape[:-2], 2, 4, 2, 4),
        other.reshape(*other.shape[:-2], 2, 4, 2, 4),
    )
    ma, mb = a.sum(axis=(-3, -1)), b.sum(axis=(-3, -1))
    known = (ma > 0) & (mb > 0)
    qa = np.divide(
        a,
        ma[..., :, None, :, None],
        out=np.zeros_like(a),
        where=ma[..., :, None, :, None] > 0,
    )
    qb = np.divide(
        b,
        mb[..., :, None, :, None],
        out=np.zeros_like(b),
        where=mb[..., :, None, :, None] > 0,
    )
    mask = known[..., :, None, :, None]
    choice = np.where(mask, (ma - mb)[..., :, None, :, None] * (qa + qb) / 2, 0)
    response = np.where(mask, (ma + mb)[..., :, None, :, None] * (qa - qb) / 2, 0)
    unresolved = np.where(mask, 0, a - b)
    return {
        "choice": choice.reshape(same.shape),
        "response": response.reshape(same.shape),
        "unresolved": unresolved.reshape(same.shape),
        "conditional_supported": known,
    }


def matched_statistics(p, y):
    """Six frozen cross-window statistics, not selected traits or accuracy."""
    p, y, _, _ = validate(p, y)
    cross = centered_accounts(p, y)["cross"]
    overall = cross[0].sum(0)
    return np.r_[
        np.linalg.norm(overall), np.linalg.norm(signed(cross, y.mean(0)), axis=(1, 2))
    ]


def matched_reference(p, y, repetitions, seed):
    """Permute late windows inside the exact full measured profile once."""
    p, y, c, number = validate(p, y)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(repetitions):
        perm = np.arange(len(p))
        for i in range(len(number)):
            idx = np.flatnonzero(c == i)
            perm[idx] = rng.permutation(idx)
        mixed = np.stack([p[:, 0], p[perm, 1]], axis=1)
        null.append(matched_statistics(mixed, y))
    observed = matched_statistics(p, y)
    null = np.asarray(null)
    return observed, null, (1 + np.sum(null >= observed, axis=0)) / (repetitions + 1)


def build_bundle(p, y):
    """Seal source-supported two-event target functions, not history posteriors."""
    p, y, _, number = validate(p, y)
    laws = source_laws(p, y)
    den = laws.sum(2)
    psi = np.divide(laws[:, :, 1], den, out=np.full_like(den, np.nan), where=den > 0)
    return {
        "version": VERSION,
        "code_hash": code_hash(),
        "models": list(MODELS),
        "source_window_mass": p.tolist(),
        "source_binary_targets": y.tolist(),
        "profile_counts": number.tolist(),
        "joint_laws": laws.tolist(),
        "pair_readout": np.where(np.isfinite(psi), psi, None).tolist(),
        "readout_supported": (den > 0).tolist(),
        "interpretation": "mean_two_event_target_readout;not_whole_history_posterior",
        "smoothing": 0,
    }


def validate_bundle(bundle):
    """Reject altered source functions, unsupported fills or extra outcome fields."""
    if bundle.get("version") != VERSION or bundle.get("code_hash") != code_hash():
        raise ValueError("Unknown cross-window implementation")
    other = build_bundle(bundle["source_window_mass"], bundle["source_binary_targets"])
    if other != bundle:
        raise ValueError("Changed source law or extra outcome field")
    return bundle


def cold_record(bundle, record):
    """Average supported pair functions on distinct early/late observations."""
    if set(record) != {"author_key", "events"} or not isinstance(
        record["author_key"], str
    ):
        raise ValueError("Cold input excludes outcomes")
    events = record["events"]
    ids, counts = [], np.zeros((2, 8), int)
    for e in events:
        if set(e) != {"event_id", "window", "cell"}:
            raise ValueError("Only identified window/cell observations allowed")
        if (
            not isinstance(e["event_id"], str)
            or not e["event_id"]
            or type(e["window"]) is not int
            or type(e["cell"]) is not int
            or e["window"] not in (0, 1)
            or not 0 <= e["cell"] < 8
        ):
            raise ValueError("Invalid event identity/window/cell")
        ids.append(e["event_id"])
        counts[e["window"], e["cell"]] += 1
    if len(ids) != len(set(ids)):
        raise ValueError("Distinct event identities required")
    size = counts.sum(1)
    result = np.full((3, 5), np.nan)
    support = np.zeros((3, 5), bool)
    if np.all(size > 0):
        weight = np.outer(counts[0] / size[0], counts[1] / size[1])
        active = weight > 0
        psi = np.asarray(bundle["pair_readout"], float)
        support = np.isfinite(psi[..., active]).all(-1)
        for k, q in np.argwhere(support):
            result[k, q] = np.sum(weight[active] * psi[k, q][active])
    return {
        "author_key": record["author_key"],
        "events_per_window": size.tolist(),
        "cross_pairs": int(np.prod(size)),
        "status": "SCORED" if np.all(size > 0) else "MISSING_WINDOW",
        "pair_target_readout": np.where(np.isfinite(result), result, None).tolist(),
        "supported": support.tolist(),
        "interpretation": bundle["interpretation"],
    }
