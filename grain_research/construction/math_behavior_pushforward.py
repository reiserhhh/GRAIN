#!/usr/bin/env python3
"""Map a frozen SOURCE evidence law to observed publication-behavior functions."""

import numpy as np
from scipy.special import logsumexp

from . import math_author_belief as B

FEATURES = (
    "mean_log_length",
    "sd_log_length",
    "mean_log_cached_gap",
    "sd_log_cached_gap",
    "venue_entropy",
    "venue_switch_fraction",
    "distinct_venue_fraction",
)


def behavioral_observation(lengths, times, venues):
    """Observe fixed metadata functions; cached gaps are not response latencies."""
    length, time, venue = map(np.asarray, (lengths, times, venues))
    if (
        length.ndim != 1
        or len(length) < 2
        or time.shape != length.shape
        or venue.shape != length.shape
        or not np.isfinite(length).all()
        or not np.isfinite(time).all()
        or np.any(length < 0)
        or np.any(np.diff(time) < 0)
        or venue.dtype.kind not in "iu"
    ):
        raise ValueError(
            "Aligned nonnegative lengths/chronological numeric venues required"
        )
    ll, gap = np.log1p(length), np.log1p(np.diff(time))
    _, counts = np.unique(venue, return_counts=True)
    p = counts / len(venue)
    return np.array(
        [
            ll.mean(),
            ll.std(),
            gap.mean(),
            gap.std(),
            -p @ np.log(p),
            np.mean(venue[1:] != venue[:-1]),
            len(counts) / len(venue),
        ]
    )


def profile_table(values, codes):
    """Retain all32 profiles; absent groups remain explicitly unsupported."""
    v, c = np.asarray(values, float), np.asarray(codes)
    if v.ndim != 2 or c.shape != (len(v),) or not np.isin(c, np.arange(32)).all():
        raise ValueError(
            "Aligned finite source values and full five-bit codes required"
        )
    if not np.isfinite(v).all():
        raise ValueError("Finite source observation functions required")
    count = np.bincount(c, minlength=32)
    table = np.zeros((32, v.shape[1]))
    np.add.at(table, c, v)
    np.divide(table, count[:, None], out=table, where=count[:, None] > 0)
    return table, count > 0


def pushforward(bundle, weights):
    """Read full mode and both old-profile expectations without application outcomes."""
    w = np.asarray(weights, float)
    values, codes = bundle["behavior"], bundle["codes"]
    if (
        w.shape != (len(values),)
        or np.any(w < 0)
        or not np.isfinite(w).all()
        or not np.isclose(w.sum(), 1)
    ):
        raise ValueError("Normalized aligned source posterior required")
    tables = [profile_table(values, row) for row in codes]
    profiles = B.profile(w, codes)
    for profile, (_table, support) in zip(profiles, tables, strict=True):
        if np.any(profile[~support] > 0):
            raise ArithmeticError("Unsupported SOURCE profile is not a neutral score")
    mediated = np.array([p @ t for p, (t, _) in zip(profiles, tables, strict=True)])
    full = w @ values
    return {
        "full_mode": full,
        "actual_profile": mediated[0],
        "old_permuted_profile": mediated[1],
        "within_profile": full - mediated,
        "profiles": profiles,
        "marginals": profiles @ ((np.arange(32)[:, None] >> np.arange(5)) & 1),
        "effective_sources": float(1 / (w @ w)),
    }


def infer(bundle, episode):
    """Use the old exact shared-kernel likelihood; no new estimator or tuning."""
    ll = B.component_log_likelihood(bundle, [np.asarray(episode)], True)
    return pushforward(bundle, np.exp(ll - logsumexp(ll)))


def cold(bundle, record):
    """Read one qualified early episode, never accept future truth or trait labels."""
    episodes = B.validate_record(record, 8)
    if len(episodes) != 1:
        raise ValueError("Only one early evidence episode is accepted")
    return infer(bundle, episodes[0])


def risk_gain(truth, prediction, prior, variance):
    """Return person/feature paired gain; constant SOURCE dimensions must be reported."""
    y, p, mu, v = map(np.asarray, (truth, prediction, prior, variance))
    if y.shape != p.shape or mu.shape != (y.shape[-1],) or v.shape != mu.shape:
        raise ValueError(
            "Aligned outcomes, predictions and frozen SOURCE units required"
        )
    if not all(np.isfinite(a).all() for a in (y, p, mu, v)) or np.any(v <= 0):
        raise ValueError(
            "Missing or constant SOURCE variance cannot be silently dropped"
        )
    return ((y - mu) ** 2 - (y - p) ** 2) / v
