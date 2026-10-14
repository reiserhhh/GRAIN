#!/usr/bin/env python3
"""Positive forward/reverse observation compatibility, without answer fitting."""

import numpy as np
from scipy.special import expit, logsumexp


def finite_forward(log_likelihood):
    """Preserve raw prior odds and normalize the declared finite outcome support."""
    a = np.asarray(log_likelihood, float)
    if a.ndim != 2 or a.shape[1] != 2 or len(a) < 2 or not np.isfinite(a).all():
        raise ValueError("At least two finite outcomes by H/L likelihoods required")
    log_z = logsumexp(a, axis=0)
    log_forward = a - log_z
    raw_odds = a[:, 0] - a[:, 1]
    finite_odds = log_forward[:, 0] - log_forward[:, 1]
    return {
        "log_forward": log_forward,
        "log_normalizer": log_z,
        "raw_odds": raw_odds,
        "finite_odds": finite_odds,
        "raw_equal_prior": expit(raw_odds),
        "finite_equal_prior": expit(finite_odds),
    }


def inverse_orders(logits):
    """Decode high-first and low-first without deleting label/order effects."""
    z = np.asarray(logits, float)
    if z.ndim != 4 or z.shape[1:] != (2, 2, 2) or not np.isfinite(z).all():
        raise ValueError("Outcomes by two contracts/two orders/two digits required")
    odds = (z[..., 1] - z[..., 0]) * np.array([-1.0, 1.0])
    return {
        "orders": odds,
        "mean": odds.mean(-1),
        "order": (odds[..., 0] - odds[..., 1]) / 2,
    }


def compatibility(raw_odds, reverse_odds):
    """Compute all prior-free four-cycles and the exact constant-error lower bound."""
    l, r = np.asarray(raw_odds, float), np.asarray(reverse_odds, float)
    if l.ndim != 1 or len(l) < 2 or r.shape != l.shape:
        raise ValueError("Matching complete forward and reverse outcomes required")
    if not np.isfinite(l).all() or not np.isfinite(r).all():
        raise ValueError("Nonfinite law; do not clip or repair")
    difference = r - l
    cycles = difference[:, None] - difference[None, :]
    return {
        "difference": difference,
        "cycles": cycles,
        "prior_offset_description": float(difference.mean()),
        "centered_rms": float(np.sqrt(np.mean((difference - difference.mean()) ** 2))),
        "constant_uniform_error_bound": float(np.ptp(difference) / 2),
        "cycle_rms": float(np.sqrt(np.mean(cycles**2))),
        "cycle_max": float(np.max(abs(cycles))),
    }


def literal_cycle(log_forward, high_probability, i, j):
    """Compare the positive four-cycle products independently of odds subtraction."""
    a, r = np.asarray(log_forward, float), np.asarray(high_probability, float)
    if a.ndim != 2 or a.shape[1] != 2 or r.shape != (len(a),):
        raise ValueError("Complete positive forward and reverse table required")
    if np.any((r <= 0) | (r >= 1)):
        raise ValueError("Strictly positive inverse law required; no clipping")
    left = np.log(r[i]) + np.log1p(-r[j]) + a[j, 0] + a[i, 1]
    right = np.log1p(-r[i]) + np.log(r[j]) + a[i, 0] + a[j, 1]
    return float(left - right)


def splice_passage(tokenizer, prefix, passage, suffix, system, max_tokens):
    """Insert exact historical token IDs into an explicit inverse-read template."""
    marker = "GRAIN_EXACT_PASSAGE_HM256"
    text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": prefix + marker + suffix},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    if text.count(marker) != 1:
        raise ValueError("Ambiguous inverse-read passage boundary")
    before, after = text.split(marker)
    start = tokenizer.encode(before, add_special_tokens=False)
    end = tokenizer.encode(after, add_special_tokens=False)
    ids = start + list(passage) + end
    if (
        not passage
        or len(ids) > max_tokens
        or ids[len(start) : len(start) + len(passage)] != list(passage)
    ):
        raise ValueError("Exact passage changed or exceeded declared support")
    return ids, len(start)
