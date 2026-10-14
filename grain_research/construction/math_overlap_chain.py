#!/usr/bin/env python3
"""Compatible SOURCE path laws with identity-bound, masked observations."""

from __future__ import annotations

import numpy as np


def bits(dimensions):
    """Return all fixed binary profiles, without inferred psychological classes."""
    return (np.arange(1 << dimensions)[:, None] >> np.arange(dimensions)) & 1


def entropy(mass):
    """Shannon entropy of finite nonnegative mass, with exact zero convention."""
    x = np.asarray(mass, float)
    positive = x > 0
    return float(-np.sum(x[positive] * np.log(x[positive])))


def propagate(initial, factors, observed):
    """Sum hidden positions in one factor chain, retaining unsupported evidence."""
    initial, factors = np.asarray(initial, float), np.asarray(factors, float)
    g, k, _ = initial.shape
    obs = np.asarray(observed)
    if (
        initial.shape != (g, k, k)
        or factors.shape != (len(obs) - 2, g, k, k, k)
        or obs.dtype.kind not in "iu"
        or np.any((obs < -1) | (obs >= k))
        or not np.isfinite(initial).all()
        or not np.isfinite(factors).all()
        or np.any(initial < 0)
        or np.any(factors < 0)
    ):
        raise ValueError("Compatible nonnegative path factors/observations required")
    masks = np.ones((len(obs), k))
    for t in np.flatnonzero(obs >= 0):
        masks[t] = np.eye(k)[obs[t]]
    state = initial * masks[0][None, :, None] * masks[1][None, None, :]
    total = state.sum()
    if total <= 0:
        return np.zeros(g), float("-inf")
    state /= total
    log_mass = np.log(total)
    for t, factor in enumerate(factors):
        state = np.einsum("gab,gabc,c->gbc", state, factor, masks[t + 2])
        total = state.sum()
        if total <= 0:
            return np.zeros(g), float("-inf")
        state /= total
        log_mass += np.log(total)
    return state.sum((1, 2)), float(log_mass)


def compile_source(paths, targets, states=8):
    """Create compatible positioned cliques; never fit application answers."""
    x, y = np.asarray(paths), np.asarray(targets)
    if (
        x.ndim != 3
        or len(x) < 2
        or x.shape[2] < 3
        or x.dtype.kind not in "iu"
        or np.any((x < 0) | (x >= states))
        or y.ndim != 2
        or len(y) != len(x)
        or not np.isin(y, (0, 1)).all()
    ):
        raise ValueError("Author-aligned finite path windows/binary targets required")
    profiles = bits(y.shape[1])
    codes = y.astype(int) @ (1 << np.arange(y.shape[1]))
    n, w, length = x.shape
    prior = np.bincount(codes, minlength=len(profiles)) / n
    if np.any(prior == 0):
        raise ValueError("All registered profiles must have SOURCE support")
    cliques = np.zeros((length - 2, len(profiles), states, states, states))
    for t in range(length - 2):
        np.add.at(
            cliques[t],
            (
                np.repeat(codes, w),
                x[:, :, t].ravel(),
                x[:, :, t + 1].ravel(),
                x[:, :, t + 2].ravel(),
            ),
            1 / (n * w),
        )
    pairs = cliques.sum(-1)
    consistency = float(np.max(abs(cliques[:-1].sum(-3) - pairs[1:])))
    if consistency > 1e-12:
        raise ValueError("Adjacent SOURCE clique margins disagree")
    initial = pairs[0]
    transitions = np.divide(
        cliques,
        pairs[..., None],
        out=np.zeros_like(cliques),
        where=pairs[..., None] > 0,
    )
    clique_given_profile = cliques / prior[None, :, None, None, None]
    unobserved = np.full(length, -1, int)
    # Each profile's product partition is computed before any application use.
    z = []
    for g in range(len(profiles)):
        _, logz = propagate(
            np.ones((1, states, states)), clique_given_profile[:, g : g + 1], unobserved
        )
        z.append(np.exp(logz))
    partitions = np.asarray(z)
    if np.any(partitions <= 0) or not np.isfinite(partitions).all():
        raise ValueError("Unsupported overlap-product profile; no rescue")
    product_initial = np.broadcast_to(
        (prior / partitions)[:, None, None], initial.shape
    ).copy()
    return {
        "profiles": profiles,
        "prior": prior,
        "initial": initial,
        "transitions": transitions,
        "cliques": cliques,
        "product_initial": product_initial,
        "product_factors": clique_given_profile,
        "partitions": partitions,
        "consistency_error": np.array(consistency),
    }


def observed_vector(record, states, length):
    """Bind one observed state per original position; aliases never add evidence."""
    if (
        set(record) != {"case_key", "author_key", "window_size", "events"}
        or record["window_size"] != length
        or not isinstance(record["case_key"], str)
        or not isinstance(record["author_key"], str)
    ):
        raise ValueError("Only addressed observed events allowed; no labels/outcomes")
    observations, seen, positions = np.full(length, -1, int), {}, {}
    for event in record["events"]:
        if set(event) != {"event_id", "position", "cell"}:
            raise ValueError("Observed event schema only")
        key, pos, cell = event["event_id"], event["position"], event["cell"]
        if (
            not isinstance(key, str)
            or not key
            or type(pos) is not int
            or not 0 <= pos < length
            or type(cell) is not int
            or not 0 <= cell < states
        ):
            raise ValueError("Distinct valid identity/position/state required")
        if key in seen:
            if seen[key] != (pos, cell):
                raise ValueError("Conflicting alias")
            continue
        if pos in positions:
            raise ValueError("Different identities claim the same event position")
        seen[key], positions[pos], observations[pos] = (pos, cell), key, cell
    return observations


def score(bundle, record):
    """Return two whole-path readings from an aggregate frozen law, no LLM."""
    length = len(bundle["transitions"]) + 2
    observed = observed_vector(record, bundle["initial"].shape[-1], length)
    result = {
        "case_key": record["case_key"],
        "observed_events": int((observed >= 0).sum()),
    }
    for name, initial, factors in (
        ("chain", bundle["initial"], bundle["transitions"]),
        ("normalized_product", bundle["product_initial"], bundle["product_factors"]),
    ):
        posterior, logmass = propagate(initial, factors, observed)
        supported = np.isfinite(logmass)
        result[name] = {
            "supported": bool(supported),
            "log_evidence": float(logmass) if supported else None,
            "profile": posterior.tolist() if supported else None,
            "targets": (posterior @ bundle["profiles"]).tolist() if supported else None,
        }
    return result


def clique_marginals(initial, factors):
    """Contract each clique using forward/backward sums under its one joint law."""
    initial, factors = np.asarray(initial), np.asarray(factors)
    backward = [None] * (len(factors) + 1)
    backward[-1] = np.ones_like(initial)
    for t in range(len(factors) - 1, -1, -1):
        backward[t] = np.einsum("gabc,gbc->gab", factors[t], backward[t + 1])
    normalizer = np.sum(initial * backward[0])
    state, out = initial.copy(), []
    for t, factor in enumerate(factors):
        out.append(state[..., None] * factor * backward[t + 1][:, None] / normalizer)
        state = np.einsum("gab,gabc->gbc", state, factor)
    return np.asarray(out)


def source_entropy_accounts(paths, targets, bundle):
    """Retain exact empirical-to-chain completion cost, not psych information."""
    x, y = np.asarray(paths), np.asarray(targets)
    codes = y.astype(int) @ (1 << np.arange(y.shape[1]))
    rows = np.column_stack((np.repeat(codes, x.shape[1]), x.reshape(-1, x.shape[-1])))
    _, counts = np.unique(rows, axis=0, return_counts=True)
    empirical_entropy = entropy(counts / counts.sum())
    conditional_entropy = sum(
        entropy(c) - entropy(c.sum(-1)) for c in bundle["cliques"]
    )
    chain_entropy = entropy(bundle["initial"]) + conditional_entropy
    log_probabilities = []
    for row in rows:
        g, path = int(row[0]), row[1:]
        value = np.log(bundle["initial"][g, path[0], path[1]])
        for t in range(len(path) - 2):
            value += np.log(bundle["transitions"][t, g, *path[t : t + 3]])
        log_probabilities.append(value)
    kl = -float(np.mean(log_probabilities)) - empirical_entropy
    return {
        "empirical_entropy": empirical_entropy,
        "chain_entropy": chain_entropy,
        "KL_empirical_chain": kl,
        "entropy_identity_error": abs(kl - chain_entropy + empirical_entropy),
    }
