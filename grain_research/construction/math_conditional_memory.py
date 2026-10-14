#!/usr/bin/env python3
"""Separate natural history, coarsened observations and fixed-condition designs."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from grain_research.construction import math_author_belief as B

VIEWS = ("full", "condition", "expression")
CONTRACTS = (
    "none",
    "full",
    "condition",
    "expression",
    "expression_given_condition",
    "condition_given_expression",
)


def fixed_rows(bundle):
    """Use the unchanged SOURCE posterior means, not a newly fitted kernel."""
    a0, a = np.asarray(bundle["alpha0"]), np.asarray(bundle["alpha"])
    if (
        a0.ndim != 2
        or a0.shape[1] != 8
        or a.shape != (len(a0), 8, 8)
        or not np.isfinite(a0).all()
        or not np.isfinite(a).all()
        or np.any(a0 <= 0)
        or np.any(a <= 0)
    ):
        raise ValueError("Positive frozen eight-state SOURCE law required")
    return a0 / a0.sum(1, keepdims=True), a / a.sum(2, keepdims=True)


def observation_masks(path, view):
    """Observe only declared bits; hidden old states are never transition inputs."""
    B.counts([path], 8)
    x, alphabet = np.asarray(path), np.arange(8)
    if view == "full":
        values, labels = x, alphabet
    elif view == "condition":
        values, labels = x // 4, alphabet // 4
    elif view == "expression":
        values, labels = x % 4, alphabet % 4
    else:
        raise ValueError("Unknown observation contract")
    return values[:, None] == labels[None, :]


def forward_log_likelihood(bundle, path, view):
    """Sum over complete hidden paths by a scaled finite HMM forward recursion."""
    initial, transition = fixed_rows(bundle)
    masks = observation_masks(path, view)
    log_total = np.zeros(len(initial))
    prefixes = [log_total.copy()]
    filtered = initial.copy()
    for t, mask in enumerate(masks):
        if t:
            filtered = np.einsum("ia,iab->ib", filtered, transition)
        filtered *= mask
        mass = filtered.sum(1)
        if np.any(mass <= 0) or not np.isfinite(mass).all():
            raise ArithmeticError("Unobserved support is not imputed as neutral")
        log_total += np.log(mass)
        filtered /= mass[:, None]
        prefixes.append(log_total.copy())
    return np.array(prefixes)


def evidence_fields(bundle, path):
    """Retain all three likelihoods and all prefixes of the SAME source model."""
    return np.array([forward_log_likelihood(bundle, path, view) for view in VIEWS])


def contract_log_likelihoods(fields):
    """Conditional ratios change the design prior; they are not natural posteriors."""
    full, condition, expression = np.asarray(fields)
    return np.array(
        [
            np.zeros_like(full),
            full,
            condition,
            expression,
            full - condition,
            full - expression,
        ]
    )


def posterior_logweights(log_likelihood):
    """Normalize under the declared uniform reference, never an application batch."""
    ell = np.asarray(log_likelihood)
    if not np.isfinite(ell).all():
        raise ValueError("Finite supported evidence likelihood required")
    return ell - logsumexp(ell, axis=-1, keepdims=True)


def observe_late(bundle, path, logweights):
    """Carry reference belief only; late episodes restart without a boundary edge."""
    weights = np.asarray(logweights)
    if weights.shape != (len(bundle["alpha0"]),) or abs(logsumexp(weights)) > 1e-10:
        raise ValueError("Normalized aligned reference-author log weights required")
    return B.observe(
        bundle,
        path,
        learn_author=True,
        learn_kernel=False,
        carry={
            "logw": weights,
            "count0": np.zeros(8, int),
            "count": np.zeros((8, 8), int),
            "learn_kernel": False,
        },
    )


def cold(bundle, record):
    """Score every frozen contract from qualified events, with no LLM or labels."""
    episodes = B.validate_record(record, 8)
    if len(episodes) != 2:
        raise ValueError("This finite contract requires one early and one late episode")
    fields = evidence_fields(bundle, episodes[0])
    weights = posterior_logweights(contract_log_likelihoods(fields))[:, -1]
    rows = {}
    for name, logw in zip(CONTRACTS, weights, strict=True):
        late = observe_late(bundle, episodes[1], logw)
        rows[name] = {
            "early_profiles": B.profile(np.exp(logw), bundle["codes"]).tolist(),
            "late_logp": late["logp"].tolist(),
            "late_profiles": late["profiles"].tolist(),
            "conditional_design_not_natural_posterior": "given" in name,
        }
    return {
        "author_key": record["author_key"],
        "contracts": rows,
        "fixed_source_rows_no_kernel_learning": True,
        "condition_is_coarse_venue_bit_not_full_situation": True,
        "not_human_trait_posterior": True,
    }
