#!/usr/bin/env python3
"""Observed paired-vector laws distinguished from author/profile products."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from grain_research.construction.math_relational_diagnostic_update import BITS

LAWS = (
    "observed_pair",
    "within_author_product",
    "within_profile_product",
    "condition_matched_pair_relink",
)
COMPONENTS = ("pair_component", "author_component", "total_component", "pair_identity")


def pair_indices(times, conditions):
    """Bind22 strict chronological edges and one condition-matched right relink."""
    t, c = np.asarray(times, float), np.asarray(conditions)
    if (
        t.ndim != 2
        or t.shape[1] != 24
        or c.shape != t.shape
        or not np.isfinite(t).all()
        or not np.isin(c, (0, 1)).all()
        or np.any(np.diff(t.reshape(-1, 2, 12), axis=-1) <= 0)
    ):
        raise ValueError(
            "All strict times/conditions in two registered12-event windows required"
        )
    left = np.r_[np.arange(11), np.arange(12, 23)]
    right = left + 1
    relink = np.tile(np.arange(22), (len(t), 1))
    for u in range(len(t)):
        for w in range(2):
            for a in (0, 1):
                for b in (0, 1):
                    ix = np.flatnonzero(
                        (left // 12 == w) & (c[u, left] == a) & (c[u, right] == b)
                    )
                    relink[u, ix] = np.roll(ix, 1)
    return left, right, relink


def event_logs(points, prototypes, variance):
    """Evaluate unchanged finite Gaussian kernels; no fit or state discretization."""
    x, z = np.asarray(points, float), np.asarray(prototypes, float)
    if (
        x.ndim != 3
        or x.shape[1] != 2
        or z.ndim != 4
        or z.shape[2:] != x.shape[1:]
        or min(z.shape) < 1
        or not np.isfinite(x).all()
        or not np.isfinite(z).all()
        or not np.isfinite(variance)
        or variance <= 0
    ):
        raise ValueError(
            "Finite aligned ordered-pair vectors/positive frozen variance required"
        )
    result = []
    for v in (0, 1):
        flat = z[:, :, v].reshape(-1, z.shape[-1])
        distance = (
            np.sum(x[:, v] ** 2, -1)[:, None]
            + np.sum(flat**2, -1)[None]
            - 2 * x[:, v] @ flat.T
        )
        result.append((-distance / (2 * variance)).reshape(len(x), len(z), z.shape[1]))
    return np.stack(result, axis=2)


def profile_readings(logs, codes, relink):
    """Return four positive marginal-matched laws and all normalized target readings."""
    k, g, r = np.asarray(logs, float), np.asarray(codes), np.asarray(relink)
    if (
        k.ndim != 4
        or k.shape[2] != 2
        or g.shape != (k.shape[1],)
        or g.dtype.kind not in "iu"
        or r.shape != (k.shape[1], k.shape[3])
        or r.dtype.kind not in "iu"
        or not np.isfinite(k).all()
        or not np.isin(g, np.arange(32)).all()
        or not np.array_equal(
            np.sort(r, axis=1), np.tile(np.arange(k.shape[3]), (len(g), 1))
        )
    ):
        raise ValueError(
            "Complete finite pair logs/profiles and per-author permutation required"
        )
    counts = np.bincount(g, minlength=32)
    if counts.min() == 0:
        raise ValueError("Every declared source profile needs observed support")
    a, b = k[:, :, 0], k[:, :, 1]
    br = np.take_along_axis(b, r[None], axis=2)
    nlog = np.log(k.shape[3])
    ma, mb = logsumexp(a, -1) - nlog, logsumexp(b, -1) - nlog
    f, j, rr = logsumexp(a + b, -1) - nlog, ma + mb, logsumexp(a + br, -1) - nlog
    log_laws = np.empty((len(k), 4, 32))
    for code in range(32):
        ix = g == code
        ni = np.log(counts[code])
        log_laws[:, 0, code] = logsumexp(f[:, ix], -1) - ni
        log_laws[:, 1, code] = logsumexp(j[:, ix], -1) - ni
        log_laws[:, 2, code] = (
            logsumexp(ma[:, ix], -1) + logsumexp(mb[:, ix], -1) - 2 * ni
        )
        log_laws[:, 3, code] = logsumexp(rr[:, ix], -1) - ni
    prior = counts / counts.sum()
    weighted = log_laws + np.log(prior)
    log_profile = weighted - logsumexp(weighted, -1, keepdims=True)
    probabilities = np.exp(log_profile)
    odds = np.stack(
        [
            logsumexp(weighted[..., BITS[:, q] == 1], -1)
            - logsumexp(weighted[..., BITS[:, q] == 0], -1)
            for q in range(5)
        ],
        -1,
    )
    return {
        "profile": probabilities,
        "targets": probabilities @ BITS,
        "log_odds": odds,
        "log_laws": log_laws,
        "profile_prior": prior,
    }


def components(values):
    """Retain pair/author/total and one paired-identity contrast with signed accounts."""
    v = np.asarray(values, float)
    if v.ndim < 2 or v.shape[-2:] != (4, 5) or not np.isfinite(v).all():
        raise ValueError("All four complete five-target laws required")
    return {
        COMPONENTS[0]: v[..., 0, :] - v[..., 1, :],
        COMPONENTS[1]: v[..., 1, :] - v[..., 2, :],
        COMPONENTS[2]: v[..., 0, :] - v[..., 2, :],
        COMPONENTS[3]: v[..., 0, :] - v[..., 3, :],
    }


def cold(bundle, record):
    """Read a qualified numerical adjacent pair without labels/answers or an LLM."""
    if (
        set(record) != {"frame_key", "sampling_regime", "events"}
        or not isinstance(record["frame_key"], str)
        or not record["frame_key"]
        or record["sampling_regime"] != "next_cached_record_within_registered_window"
        or not isinstance(record["events"], list)
        or len(record["events"]) != 2
    ):
        raise ValueError("Exactly one qualified adjacent-pair record required")
    seen, vectors, times, windows = set(), [], [], []
    for e in record["events"]:
        if set(e) != {"event_id", "vector", "time", "window", "condition"}:
            raise ValueError(
                "Only addressed numeric evidence permitted, not labels/answers"
            )
        uid = e["event_id"]
        if not isinstance(uid, str) or not uid or uid in seen:
            raise ValueError("Two distinct evidence identities required")
        if type(e["condition"]) is not int or e["condition"] not in (0, 1):
            raise ValueError("Registered coarse observed condition required")
        if type(e["window"]) is not int or e["window"] not in (0, 1):
            raise ValueError("Registered source window required")
        seen.add(uid)
        vectors.append(e["vector"])
        times.append(e["time"])
        windows.append(e["window"])
    times = np.asarray(times, float)
    if not np.isfinite(times).all() or times[1] <= times[0] or windows[0] != windows[1]:
        raise ValueError(
            "Strict same-window order required; other sampling regimes need their own law"
        )
    logs = event_logs(
        np.asarray(vectors, float)[None],
        bundle["prototypes"],
        float(bundle["variance"]),
    )
    result = profile_readings(logs, bundle["codes"], bundle["relink"])
    return {
        "frame_key": record["frame_key"],
        "sampling_regime": record["sampling_regime"],
        "adjacency_provenance": "caller-declared next cached pair, not inferred from timestamps alone",
        "laws": {
            name: {
                k: result[k][0, i].tolist() for k in ("profile", "targets", "log_odds")
            }
            for i, name in enumerate(LAWS)
        },
        "components": {
            k: v[0].tolist() for k, v in components(result["targets"]).items()
        },
        "overlapping_pairs_not_multiplied_as_independent_events": True,
    }
