#!/usr/bin/env python3
"""Cross-fitted author functionals of an existing paired observation law."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from scipy.stats import rankdata

from grain_research.construction import math_source_event_coupling as P

SERIES = P.LAWS + P.COMPONENTS
WEIGHT_LAWS = (0, 1, 3)
WINDOWS = ("window0", "window1", "all22")


def source_author_weights(logs, relink):
    """Return label-independent F/J/R masses on source authors for each pair."""
    k = np.asarray(logs, float)
    r = np.asarray(relink)
    if (
        k.ndim != 4
        or k.shape[2] != 2
        or r.shape != (k.shape[1], k.shape[3])
        or r.dtype.kind not in "iu"
        or not np.isfinite(k).all()
        or not np.array_equal(
            np.sort(r, axis=1), np.tile(np.arange(k.shape[3]), (k.shape[1], 1))
        )
    ):
        raise ValueError("Finite paired logs and source-author relinks required")
    a, b = k[:, :, 0], k[:, :, 1]
    nlog = np.log(k.shape[-1])
    f = logsumexp(a + b, -1) - nlog
    j = logsumexp(a, -1) + logsumexp(b, -1) - 2 * nlog
    rr = logsumexp(a + np.take_along_axis(b, r[None], axis=2), -1) - nlog
    mass = np.stack((f, j, rr), axis=1)
    return np.exp(mass - logsumexp(mass, axis=-1, keepdims=True))


def aggregate_profiles(pair_profiles):
    """Average dependent pair readouts; never form a likelihood product."""
    p = np.asarray(pair_profiles, float)
    if (
        p.ndim < 3
        or p.shape[-3:] != (22, 4, 32)
        or not np.isfinite(p).all()
        or np.any(p < 0)
        or not np.allclose(p.sum(-1), 1, atol=1e-12, rtol=0)
    ):
        raise ValueError("All22 normalized four-law profile readings required")
    halves = p.reshape(*p.shape[:-3], 2, 11, 4, 32).mean(-3)
    return np.concatenate((halves, halves.mean(-3, keepdims=True)), axis=-3)


def all_series(targets):
    """Keep probability laws and all signed pair/author/identity components."""
    t = np.asarray(targets, float)
    components = P.components(t)
    return np.concatenate((t, np.stack(list(components.values()), axis=-2)), axis=-2)


def corr(x, y, banks=None, ranks=False):
    """Column correlations, optionally removing bank offsets in the statistic only."""
    x, y = np.asarray(x, float).copy(), np.asarray(y, float).copy()
    if x.shape != y.shape or x.ndim != 2 or not np.isfinite([x, y]).all():
        raise ValueError("Matching finite n-by-target arrays required")
    groups = [np.arange(len(x))]
    if banks is not None:
        b = np.asarray(banks)
        if b.shape != (len(x),):
            raise ValueError("One bank identity per author required")
        groups = [np.flatnonzero(b == k) for k in np.unique(b)]
    for ix in groups:
        if ranks:
            x[ix], y[ix] = rankdata(x[ix], axis=0), rankdata(y[ix], axis=0)
        x[ix] -= x[ix].mean(0)
        y[ix] -= y[ix].mean(0)
    denominator = np.sqrt(np.sum(x * x, 0) * np.sum(y * y, 0))
    return np.divide(
        np.sum(x * y, 0),
        denominator,
        out=np.full(x.shape[1], np.nan),
        where=denominator > 0,
    )


def permutation_inference(pair_weights, codes, banks, count, seed):
    """Rebuild source AND held criteria under whole-profile within-bank permutations."""
    d = np.asarray(pair_weights, float)
    g, b = np.asarray(codes), np.asarray(banks)
    if d.ndim != 3 or d.shape[1:] != (len(g), len(g)) or b.shape != g.shape:
        raise ValueError("Space-by-author cross-fit weights required")
    if np.any(d[:, b[:, None] == b[None, :]] != 0):
        raise ValueError("Own-bank sources cannot be used in a criterion reading")
    labels = P.BITS[g]
    observed = np.stack([corr(w @ labels, labels, b) for w in d])
    rng = np.random.default_rng(seed)
    null, permutations = [], []
    for _ in range(count):
        gp = g.copy()
        for bank in np.unique(b):
            ix = np.flatnonzero(b == bank)
            gp[ix] = rng.permutation(g[ix])
        lp = P.BITS[gp]
        null.append(np.stack([corr(w @ lp, lp, b) for w in d]))
        permutations.append(gp)
    null = np.asarray(null)
    if not np.isfinite(observed).all() or not np.isfinite(null).all():
        raise ValueError("Undefined primary correlation; retain failure, do not rescue")
    nominal = (1 + np.sum(abs(null) >= abs(observed), axis=0)) / (count + 1)
    maximum = np.max(abs(null), axis=(1, 2))
    family = (1 + np.sum(maximum[:, None, None] >= abs(observed), axis=0)) / (count + 1)
    return observed, null, nominal, family, np.asarray(permutations)


def bootstrap_primary(scores, labels, banks, count, seed):
    """Bank-stratified author CI conditional on already-frozen cross-fitted scores."""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(banks == k) for k in np.unique(banks)]
    values, samples = [], []
    for _ in range(count):
        ix = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        values.append(np.stack([corr(s[ix], labels[ix], banks[ix]) for s in scores]))
        samples.append(ix)
    a = np.asarray(values)
    return np.nanquantile(a, (0.025, 0.975), axis=0), a, np.asarray(samples)


def matched_strangers(conditions, banks):
    """Deterministic same-bank nearest stranger using coarse conditions, not labels."""
    c = np.asarray(conditions)
    if c.shape != (len(banks), 24) or not np.isin(c, (0, 1)).all():
        raise ValueError("Two complete12-event coarse condition windows required")
    halves = c.reshape(-1, 2, 12)
    transitions = 2 * halves[:, :, :-1] + halves[:, :, 1:]
    feature = np.concatenate(
        (
            halves.mean(-1),
            np.stack([(transitions == k).mean(-1) for k in range(4)], -1).reshape(
                len(c), -1
            ),
        ),
        axis=1,
    )
    distance = np.sum((feature[:, None] - feature[None, :]) ** 2, axis=-1)
    distance[np.asarray(banks)[:, None] != np.asarray(banks)[None, :]] = np.inf
    np.fill_diagonal(distance, np.inf)
    match = np.argmin(distance, axis=1)
    return match, feature, distance[np.arange(len(c)), match]


def cold_author(bundle, pairs, space):
    """Score22 ordered numeric pairs from SOURCE only, no evaluated-author label."""
    x = np.asarray(pairs, float)
    z = np.asarray(bundle["prototypes"], float)
    if x.shape != (22, 2, z.shape[-1]) or space not in ("E5", "GRAIN15"):
        raise ValueError("All22 registered numeric pairs and declared space required")
    if space == "GRAIN15":
        x, z = x @ bundle["projection"], z @ bundle["projection"]
    values = P.profile_readings(
        P.event_logs(x, z, float(bundle["variance"])), bundle["codes"], bundle["relink"]
    )
    profiles = aggregate_profiles(values["profile"])
    return {"profiles": profiles, "series": all_series(profiles @ P.BITS)}
