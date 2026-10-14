#!/usr/bin/env python3
"""Keep choice semantics, answer order and output-link effects distinct."""

from __future__ import annotations

import numpy as np


def sigmoid(values):
    """Evaluate a binary observation link without overflowing at large logits."""
    x = np.asarray(values, float)
    if not np.isfinite(x).all():
        raise ValueError("Finite log odds required")
    return np.exp(-np.logaddexp(0.0, -x))


def nine_table(values):
    """Require the three policy states by three actually observed history states."""
    x = np.asarray(values, float)
    if x.shape != (3, 3) or not np.isfinite(x).all():
        raise ValueError("All nine finite observations are required")
    return x


def contrast_basis():
    """Use constant, high-minus-low and nonneutral-minus-neutral contrasts."""
    return np.column_stack(
        (
            np.ones(3) / np.sqrt(3),
            [-1 / np.sqrt(2), 0, 1 / np.sqrt(2)],
            [1 / np.sqrt(6), -2 / np.sqrt(6), 1 / np.sqrt(6)],
        )
    )


def orthogonal_accounts(values):
    """Preserve all nine contrast coordinates; they are not personality factors."""
    x = nine_table(values)
    b = contrast_basis()
    coefficients = b.T @ x @ b
    recovered = b @ coefficients @ b.T
    return coefficients, float(np.max(abs(x - recovered)))


def link_accounts(log_odds):
    """Separate actual log-odds interaction from curvature of the binary link."""
    ell = nine_table(log_odds)
    q = sigmoid(ell)
    separated = ell[:, 1, None] + ell[None, 1, :] - ell[1, 1]
    q_separated = sigmoid(separated)
    q_additive = q[:, 1, None] + q[None, 1, :] - q[1, 1]
    link_curvature = q_separated - q_additive
    history_policy_remainder = q - q_separated
    return {
        "q": q,
        "separated_log_odds": separated,
        "log_odds_interaction": ell - separated,
        "q_interaction": q - q_additive,
        "link_curvature": link_curvature,
        "response_remainder": history_policy_remainder,
        "q_reconstruction_error": float(
            np.max(abs(q - q_additive - link_curvature - history_policy_remainder))
        ),
    }


def aligned_order_accounts(order_zero, order_one):
    """Retain the order-even action field AND order-odd protocol susceptibility."""
    x, y = nine_table(order_zero), nine_table(order_one)
    even, odd = (x + y) / 2, (x - y) / 2
    return {
        "even": even,
        "odd": odd,
        "reconstruction_error": float(
            max(np.max(abs(x - even - odd)), np.max(abs(y - even + odd)))
        ),
    }
