#!/usr/bin/env python3
"""Positive paired evidence and target-valued dependence, without answer fitting."""

import numpy as np
from scipy.special import logsumexp

from grain_research.construction.math_joint_target_atlas import characters

TARGETS = ("E", "N", "A", "C", "O")
BITS = ((np.arange(32)[:, None] >> np.arange(5)) & 1).astype(float)
LAWS = ("same_author_pair", "profile_marginal_product")


def compile_bank(prototypes, codes, variance):
    """Freeze two source windows and the full profile reference, without a fit."""
    z, g = np.asarray(prototypes, float), np.asarray(codes)
    if (
        z.ndim != 3
        or z.shape[1] != 2
        or z.shape[2] == 0
        or not np.isfinite(z).all()
        or g.shape != (len(z),)
        or not np.isin(g, np.arange(32)).all()
        or not np.isfinite(variance)
        or variance <= 0
    ):
        raise ValueError(
            "Finite two-window sources, full profiles and fixed scale required"
        )
    counts = np.bincount(g.astype(int), minlength=32)
    if counts.min() == 0:
        raise ValueError("Missing source profiles are not neutral observations")
    return {"prototypes": z, "codes": g.astype(int), "variance": np.array(variance)}


def update(log_independent, log_factor):
    """Normalize profile evidence and retain the common dependence factor."""
    li, d = np.asarray(log_independent, float), np.asarray(log_factor, float)
    if li.ndim != 2 or li.shape[-1] != 32 or li.shape != d.shape:
        raise ValueError("All32 finite profile likelihoods required")
    if not np.isfinite(li).all() or not np.isfinite(d).all():
        raise ValueError("Finite log evidence required")
    logs = np.stack((li + d, li), axis=1)
    normalizers = logsumexp(logs, axis=-1)
    log_profile = logs - normalizers[..., None]
    probabilities = np.exp(log_profile)
    odds = np.stack(
        [
            logsumexp(logs[..., BITS[:, q] == 1], axis=-1)
            - logsumexp(logs[..., BITS[:, q] == 0], axis=-1)
            for q in range(5)
        ],
        axis=-1,
    )
    return {
        "log_profile": log_profile,
        "profile": probabilities,
        "targets": probabilities @ BITS,
        "log_odds": odds,
        "log_factor": d,
        "factor_characters": d @ characters() / 32,
        "log_evidence_ratio": normalizers[:, 0] - normalizers[:, 1],
        "target_log_odds_update": odds[:, 0] - odds[:, 1],
    }


def read(bank, points, *, fields=False):
    """Read a common paired observation under both positive source laws."""
    bank = compile_bank(bank["prototypes"], bank["codes"], bank["variance"])
    z, g, s2 = bank["prototypes"], bank["codes"], float(bank["variance"])
    x = np.asarray(points, float)
    if x.ndim != 3 or x.shape[1:] != z.shape[1:] or not np.isfinite(x).all():
        raise ValueError(
            "Finite two-window observations in registered coordinates required"
        )
    # Query-only Gaussian factors cancel in both the ratio and profile posterior.
    k = np.array(
        [x[:, v] @ z[:, v].T / s2 - np.sum(z[:, v] ** 2, 1) / (2 * s2) for v in (0, 1)]
    )
    joint, independent = np.empty((len(x), 32)), np.empty((len(x), 32))
    means = np.empty((len(x), 2, 32, 2, z.shape[-1])) if fields else None
    for code in range(32):
        ix = g == code
        a, b = k[:, :, ix]
        aj, bj, pj = logsumexp(a, 1), logsumexp(b, 1), logsumexp(a + b, 1)
        nlog = np.log(ix.sum())
        joint[:, code] = pj - nlog
        independent[:, code] = aj + bj - 2 * nlog
        if fields:
            wj = np.exp(a + b - pj[:, None])
            means[:, 0, code, 0] = wj @ z[ix, 0]
            means[:, 0, code, 1] = wj @ z[ix, 1]
            means[:, 1, code, 0] = np.exp(a - aj[:, None]) @ z[ix, 0]
            means[:, 1, code, 1] = np.exp(b - bj[:, None]) @ z[ix, 1]
    result = update(independent, joint - independent)
    if fields:
        gradients = np.empty((len(x), 2, 5, 2, z.shape[-1]))
        for q in range(5):
            hi = BITS[:, q] == 1
            weights = result["profile"]
            high = weights[..., hi] / weights[..., hi].sum(-1, keepdims=True)
            low = weights[..., ~hi] / weights[..., ~hi].sum(-1, keepdims=True)
            gradients[:, :, q] = (
                np.einsum("nmg,nmgvd->nmvd", high, means[:, :, hi])
                - np.einsum("nmg,nmgvd->nmvd", low, means[:, :, ~hi])
            ) / s2
        result["target_fields"] = gradients
        result["relational_target_field"] = gradients[:, 0] - gradients[:, 1]
    return result


def literal(bank, point):
    """Enumerate direct-distance Gaussian components, not vectorized dot kernels."""
    z, g, s2 = bank["prototypes"], bank["codes"], float(bank["variance"])
    kernels = np.array(
        [
            [np.exp(-(np.linalg.norm(point[v] - row[v]) ** 2) / (2 * s2)) for row in z]
            for v in (0, 1)
        ]
    )
    joint, product = [], []
    for code in range(32):
        a, b = kernels[:, g == code]
        joint.append(sum(float(aa * bb) for aa, bb in zip(a, b, strict=True)) / len(a))
        product.append(sum(float(aa * bb) for aa in a for bb in b) / len(a) ** 2)
    laws = np.array((joint, product))
    if np.any(laws <= 0):
        raise ValueError("Literal audit outside representable exponential range")
    p = laws / laws.sum(1, keepdims=True)
    return {"profile": p, "targets": p @ BITS, "log_factor": np.log(laws[0] / laws[1])}


def cold_point(record, dimensions):
    """Bind two disjoint event sets; do not accept target labels or model answers."""
    if (
        set(record) != {"frame_key", "windows"}
        or not isinstance(record["frame_key"], str)
        or not record["frame_key"]
        or not isinstance(record["windows"], list)
        or len(record["windows"]) != 2
    ):
        raise ValueError("Two addressed event windows required")
    seen, means = set(), []
    for window in record["windows"]:
        if not isinstance(window, list) or not window:
            raise ValueError("Missing window cannot be a zero vector")
        vectors = []
        for e in window:
            if set(e) != {"event_id", "vector", "condition"}:
                raise ValueError(
                    "Only event IDs, vectors and observed conditions permitted"
                )
            uid, v = e["event_id"], np.asarray(e["vector"], float)
            if not isinstance(uid, str) or not uid or uid in seen:
                raise ValueError("Repeated evidence is not an independent window")
            if v.shape != (dimensions,) or not np.isfinite(v).all():
                raise ValueError("Finite registered coordinates required")
            if type(e["condition"]) is not int or e["condition"] not in (0, 1):
                raise ValueError("Registered observed condition required")
            seen.add(uid)
            vectors.append(v)
        means.append(np.mean(vectors, 0))
    return np.asarray(means)


def cold(bank, record):
    """Return all five diagnostic updates, not a fitted or psychological probability."""
    r = read(bank, cold_point(record, bank["prototypes"].shape[-1])[None])
    return {
        "frame_key": record["frame_key"],
        "targets_by_law": r["targets"][0].tolist(),
        "joint_profiles_by_law": r["profile"][0].tolist(),
        "target_log_odds_update": r["target_log_odds_update"][0].tolist(),
        "common_log_factor": float(r["factor_characters"][0, 0]),
        "observation_log_evidence_ratio": float(r["log_evidence_ratio"][0]),
    }
