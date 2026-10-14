#!/usr/bin/env python3
"""Source-defined path beliefs distinguish author learning from kernel learning."""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln, logsumexp

PROGRAMS = ((False, False), (True, False), (False, True), (True, True))
NAMES = ("fixed_reset", "fixed_shared", "kernel_reset", "kernel_shared")


def start_matched_donors(early):
    """Rotate within the first-state group; singleton donors stay unavailable."""
    x = np.asarray(early)
    if x.ndim != 2 or not len(x) or x.dtype.kind not in "iu":
        raise ValueError("Aligned early numeric paths required, no target labels")
    donors = np.full(len(x), -1, int)
    for start in np.unique(x[:, 0]):
        ix = np.flatnonzero(x[:, 0] == start)
        if len(ix) > 1:
            donors[ix] = np.roll(ix, 1)
    return donors


def counts(episodes, states):
    """Count starts and transitions, never a transition between two episodes."""
    initial, edges = np.zeros(states, int), np.zeros((states, states), int)
    for episode in episodes:
        x = np.asarray(episode)
        if (
            x.ndim != 1
            or not len(x)
            or x.dtype.kind not in "iu"
            or np.any((x < 0) | (x >= states))
        ):
            raise ValueError("Nonempty integer episode in the declared state alphabet")
        initial[x[0]] += 1
        np.add.at(edges, (x[:-1], x[1:]), 1)
    return initial, edges


def compile_source(paths, targets, source_indices, prior_mass=1.0, states=8):
    """Compile SOURCE-only Dirichlet rows; prior mass is not observed support."""
    x, y, ids = np.asarray(paths), np.asarray(targets), np.asarray(source_indices)
    if (
        x.ndim != 3
        or not len(x)
        or y.shape != (2, len(x), 5)
        or not np.isin(y, (0, 1)).all()
        or ids.shape != (len(x),)
        or len(np.unique(ids)) != len(ids)
        or not np.isfinite(prior_mass)
        or prior_mass <= 0
    ):
        raise ValueError("Aligned source identities, episodes and two full assignments")
    c0, c = zip(*(counts(row, states) for row in x), strict=True)
    c0, c = np.array(c0), np.array(c)
    total0, total = c0.sum(0), c.sum(0)
    if np.any(total0 <= 0) or np.any(total <= 0):
        raise ValueError("Globally missing SOURCE cells: no epsilon support rescue")
    base0 = total0 / total0.sum()
    base = total / total.sum(1, keepdims=True)
    return {
        "alpha0": c0 + prior_mass * base0,
        "alpha": c + prior_mass * base,
        "count0": c0,
        "count": c,
        "base0": prior_mass * base0,
        "base": prior_mass * base,
        "codes": y.astype(int) @ (1 << np.arange(5)),
        "source_indices": ids,
    }


def profile(weights, codes):
    """Keep both assignments and all32 SOURCE profiles instead of marginal products."""
    return np.array([np.bincount(row, weights=weights, minlength=32) for row in codes])


def component_log_likelihood(bundle, episodes, learn_kernel):
    """Calculate exact fixed-kernel or integrated Dirichlet likelihoods."""
    a0, a = bundle["alpha0"], bundle["alpha"]
    n0, n = counts(episodes, a.shape[-1])
    if not learn_kernel:
        return np.sum(n0 * np.log(a0 / a0.sum(1, keepdims=True)), 1) + np.sum(
            n * np.log(a / a.sum(2, keepdims=True)), (1, 2)
        )

    def beta_ratio(alpha, added):
        return (
            gammaln(alpha.sum(-1))
            - gammaln(alpha.sum(-1) + added.sum(-1))
            + np.sum(gammaln(alpha + added) - gammaln(alpha), -1)
        )

    return beta_ratio(a0, n0) + beta_ratio(a, n).sum(1)


def observe(bundle, episode, learn_author=True, learn_kernel=True, carry=None):
    """Predict before each observation; only shared programs give author posteriors."""
    a0, a = bundle["alpha0"], bundle["alpha"]
    n_sources, states = a0.shape
    counts([episode], states)
    if carry is not None and (
        not learn_author or carry["learn_kernel"] != learn_kernel
    ):
        raise ValueError("Carry requires the same shared-author observation law")
    logw = (
        np.full(n_sources, -np.log(n_sources))
        if carry is None
        else carry["logw"].copy()
    )
    n0 = np.zeros(states, int) if carry is None else carry["count0"].copy()
    n = np.zeros((states, states), int) if carry is None else carry["count"].copy()
    fields = [profile(np.exp(logw), bundle["codes"])]
    effective = [float(1 / np.sum(np.exp(2 * logw)))]
    predictions, logs, shares = [], [], []
    x = np.asarray(episode)
    for t, b in enumerate(x):
        if t == 0:
            alpha, source, base = a0, bundle["count0"], bundle["base0"]
            added = n0 if learn_kernel else np.zeros(states, int)
        else:
            origin = x[t - 1]
            alpha, source, base = (
                a[:, origin],
                bundle["count"][:, origin],
                bundle["base"][origin],
            )
            added = n[origin] if learn_kernel else np.zeros(states, int)
        denominator = (alpha + added).sum(1)
        q = (alpha + added) / denominator[:, None]
        weights = np.exp(logw)
        predictive = weights @ q
        probability = predictive[b]
        if probability <= 0 or not np.isfinite(predictive).all():
            raise ArithmeticError("No missing probability is imputed as neutral")
        predictions.append(predictive)
        logs.append(float(np.log(probability)))
        shares.append(
            [
                float(np.sum(weights * part / denominator) / probability)
                for part in (
                    source[:, b],
                    np.full(n_sources, base[b]),
                    np.full(n_sources, added[b]),
                )
            ]
        )
        if t == 0 or learn_author:
            logw += np.log(q[:, b])
            logw -= logsumexp(logw)
        if t == 0:
            n0[b] += 1
        else:
            n[x[t - 1], b] += 1
        fields.append(profile(np.exp(logw), bundle["codes"]))
        effective.append(float(1 / np.sum(np.exp(2 * logw))))
    profiles = np.array(fields)
    bits = (np.arange(32)[:, None] >> np.arange(5)) & 1
    return {
        "predictions": np.array(predictions),
        "logp": np.array(logs),
        "probability_shares_source_prior_application": np.array(shares),
        "profiles": np.moveaxis(profiles, 0, 1),
        "marginals": np.moveaxis(profiles @ bits, 0, 1),
        "effective_sources": np.array(effective),
        "profile_is_author_posterior": learn_author,
        "carry": {"logw": logw, "count0": n0, "count": n, "learn_kernel": learn_kernel},
    }


def validate_record(record, states):
    """Reject hidden labels, duplicate events and nonchronological episode boundaries."""
    if set(record) != {"author_key", "episodes"} or not record["episodes"]:
        raise ValueError("Only opaque author key and observed episodes are allowed")
    ids, times, episodes = [], [], []
    for row in record["episodes"]:
        if set(row) != {"events"} or not row["events"]:
            raise ValueError("Episode must contain identified events")
        cells = []
        for event in row["events"]:
            if set(event) != {"event_id", "timestamp", "cell"}:
                raise ValueError(
                    "No answer, psychological label or source ID in cold input"
                )
            ids.append(event["event_id"])
            times.append(event["timestamp"])
            cells.append(event["cell"])
        episodes.append(np.array(cells))
    if (
        any(not isinstance(v, str) or not v for v in ids)
        or len(set(ids)) != len(ids)
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError("Unique event identities and strict chronology required")
    counts(episodes, states)
    return episodes


def cold(bundle, record, learn_kernel=True):
    """Return a complete source-anchor trajectory without a neural/answer channel."""
    episodes = validate_record(record, bundle["alpha"].shape[-1])
    state, rows = None, []
    for episode in episodes:
        row = observe(bundle, episode, True, learn_kernel, state)
        state = row["carry"]
        rows.append(
            {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in row.items()
                if k != "carry"
            }
        )
    return {
        "author_key": record["author_key"],
        "episodes": rows,
        "not_human_trait_posterior": True,
    }
