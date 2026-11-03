#!/usr/bin/env python3
"""Collapsed partial-path kernel learning with fixed-budget numerical realizations."""

import numpy as np
from scipy.special import logsumexp

from . import math_author_belief as B
from . import math_conditional_memory as C


def parameters(bundle):
    """Validate positive unchanged finite SOURCE Dirichlet parameters."""
    a0, a = np.asarray(bundle["alpha0"], float), np.asarray(bundle["alpha"], float)
    if (
        a0.ndim != 2
        or not len(a0)
        or a0.shape[1] < 2
        or a.shape != (len(a0), a0.shape[1], a0.shape[1])
    ):
        raise ValueError("Aligned source start/transition laws required")
    if (
        not np.isfinite(a0).all()
        or not np.isfinite(a).all()
        or np.any(a0 <= 0)
        or np.any(a <= 0)
    ):
        raise ValueError("Strict positive SOURCE support required;no epsilon rescue")
    return a0, a


def categorical(probability, rng):
    """Sample rowwise normalized categories without interpreting label or author ID."""
    p = np.asarray(probability, float)
    cdf = p.cumsum(-1)
    cdf[..., -1] = 1.0
    return (rng.random(p.shape[:-1])[..., None] > cdf).sum(-1)


def ancestors(weights, rng):
    """Systematically resample each SOURCE's count particles without cross-source mixing."""
    n, k = weights.shape
    offsets = 2.0 * np.arange(n)
    cdf = weights.cumsum(1)
    cdf[:, -1] = 1.0
    positions = offsets[:, None] + (rng.random((n, 1)) + np.arange(k)) / k
    selected = np.searchsorted(
        (cdf + offsets[:, None]).ravel(), positions.ravel(), side="left"
    ).reshape(n, k)
    if np.any(selected // k != np.arange(n)[:, None]):
        raise ArithmeticError("Count ancestors crossed SOURCE identity")
    return selected % k


def filter_masks(bundle, masks, particles=64, seed=29020261014):
    """Integrate shared unknown rows approximately using only allowed-state masks."""
    a0, a = parameters(bundle)
    mask = np.asarray(masks)
    n, states = a0.shape
    if (
        mask.dtype != np.bool_
        or mask.ndim != 2
        or mask.shape[1] != states
        or not len(mask)
        or not mask.any(1).all()
    ):
        raise ValueError("Nonempty boolean observation masks required")
    if type(particles) is not int or particles < 1:
        raise ValueError("Fixed positive particle count required")
    rng = np.random.default_rng(seed)
    count0 = np.zeros((n, particles, states), np.int32)
    count = np.zeros((n, particles, states, states), np.int32)
    last = np.zeros((n, particles), int)
    log_evidence = np.zeros(n)
    prefixes = [log_evidence.copy()]
    src = np.arange(n)[:, None]
    part = np.arange(particles)[None, :]
    for t, allowed in enumerate(mask):
        if t == 0:
            q = np.broadcast_to(
                (a0 / a0.sum(1, keepdims=True))[:, None, :], (n, particles, states)
            ).copy()
        else:
            row = a[src, last] + count[src, part, last]
            q = row / row.sum(-1, keepdims=True)
        restricted = q * allowed
        mass = restricted.sum(-1)
        if np.any(mass <= 0) or not np.isfinite(mass).all():
            raise ArithmeticError("Unobserved support is not imputed as neutral")
        log_evidence += np.log(mass.mean(1))
        selected = ancestors(mass / mass.sum(1, keepdims=True), rng)
        count0, count, last = (
            count0[src, selected],
            count[src, selected],
            last[src, selected],
        )
        conditional = restricted[src, selected] / mass[src, selected, None]
        new = categorical(conditional, rng)
        if t == 0:
            count0[src, part, new] += 1
        else:
            count[src, part, last, new] += 1
        last = new
        prefixes.append(log_evidence.copy())
    return {
        "log_evidence": np.array(prefixes),
        "count0": count0,
        "count": count,
        "last": last,
    }


def exact_state(bundle, path):
    """Supply deterministic sufficient statistics for full observations,without particles."""
    a0, a = parameters(bundle)
    n0, count = B.counts([path], a.shape[-1])
    return {
        "log_evidence": np.array(
            [
                B.component_log_likelihood(bundle, [path[:t]], True)
                if t
                else np.zeros(len(a0))
                for t in range(len(path) + 1)
            ]
        ),
        "count0": np.broadcast_to(n0, (len(a0), 1, len(n0))).copy(),
        "count": np.broadcast_to(count, (len(a0), 1, *count.shape)).copy(),
    }


def early_contracts(bundle, path, particles, seed):
    """Retain six observation laws and the correct hidden-count state for each."""
    a0, _a = parameters(bundle)
    n, states = a0.shape
    full = exact_state(bundle, path)
    condition = filter_masks(
        bundle, C.observation_masks(path, "condition"), particles, seed
    )
    expression = filter_masks(
        bundle, C.observation_masks(path, "expression"), particles, seed + 1
    )
    fields = np.array([s["log_evidence"] for s in (full, condition, expression)])
    likelihoods = C.contract_log_likelihoods(fields)
    logw = C.posterior_logweights(likelihoods[:, -1])
    none = {
        "count0": np.zeros((n, 1, states), int),
        "count": np.zeros((n, 1, states, states), int),
    }
    states_by_contract = (none, full, condition, expression, full, full)
    return {
        "fields": fields,
        "logweights": logw,
        "states": states_by_contract,
        "approximate_coarse_likelihood_exceeds_exact_full": fields[0]
        > fields[1:] + 1e-12,
    }


def late(bundle, path, logweights, early_state):
    """Predict actual future vector-states before observing them,with no boundary edge."""
    a0, a = parameters(bundle)
    n, states = a0.shape
    B.counts([path], states)
    n0, count = early_state["count0"], early_state["count"]
    k = n0.shape[1]
    if (
        n0.shape != (n, k, states)
        or count.shape != (n, k, states, states)
        or logweights.shape != (n,)
        or abs(logsumexp(logweights)) > 1e-10
    ):
        raise ValueError("Aligned source beliefs and count particles required")
    lw = np.broadcast_to(logweights[:, None] - np.log(k), (n, k)).copy()
    extra0, extra = np.zeros(states, int), np.zeros((states, states), int)
    logs, predictions = [], []
    profile_rows = [B.profile(np.exp(lw).sum(1), bundle["codes"])]
    neff = [float(1 / np.sum(np.exp(lw).sum(1) ** 2))]
    for t, b in enumerate(path):
        row = (
            a0[:, None, :] + n0 + extra0
            if t == 0
            else a[:, path[t - 1], None, :]
            + count[:, :, path[t - 1], :]
            + extra[path[t - 1]]
        )
        q = row / row.sum(-1, keepdims=True)
        predictive = np.einsum("ir,irb->b", np.exp(lw), q)
        if not np.isfinite(predictive).all() or np.any(predictive <= 0):
            raise ArithmeticError("Predictive support failed;no smoothing rescue")
        predictions.append(predictive)
        logs.append(float(np.log(predictive[b])))
        lw += np.log(q[..., b])
        lw -= logsumexp(lw)
        if t == 0:
            extra0[b] += 1
        else:
            extra[path[t - 1], b] += 1
        w = np.exp(lw).sum(1)
        profile_rows.append(B.profile(w, bundle["codes"]))
        neff.append(float(1 / np.sum(w**2)))
    profiles = np.moveaxis(np.array(profile_rows), 0, 1)
    bits = (np.arange(32)[:, None] >> np.arange(5)) & 1
    return {
        "predictions": np.asarray(predictions),
        "logp": np.asarray(logs),
        "profiles": profiles,
        "marginals": profiles @ bits,
        "effective_sources": np.asarray(neff),
    }


def cold(bundle, record, particles=64, seed=29020261014):
    """Run all six approximate observation laws without neural or author-label inputs."""
    episodes = B.validate_record(record, 8)
    if len(episodes) != 2:
        raise ValueError("Exactly two qualified episodes required")
    result = early_contracts(bundle, episodes[0], particles, seed)
    return {
        "author_key": record["author_key"],
        "contracts": {
            name: {
                key: value.tolist()
                for key, value in late(bundle, episodes[1], w, state).items()
            }
            for name, w, state in zip(
                C.CONTRACTS, result["logweights"], result["states"], strict=True
            )
        },
        "early_log_likelihoods": result["fields"].tolist(),
        "coarse_monotonicity_violation_mask": result[
            "approximate_coarse_likelihood_exceeds_exact_full"
        ].tolist(),
        "approximate_coarse_integration_not_exact": True,
        "not_human_trait_posterior": True,
    }
