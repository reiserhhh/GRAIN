#!/usr/bin/env python3
"""Fixed-body context accounts with response nonlinearity kept separate."""

from __future__ import annotations

import numpy as np


def context_characters(values):
    """Decompose a (..., body-low/high, context-original/common) response table."""
    y = np.asarray(values, float)
    if y.shape[-2:] != (2, 2) or not np.isfinite(y).all():
        raise ValueError("Four finite body/context corners required")
    mean = y.mean(axis=(-2, -1))
    body = (y[..., 1, :].sum(-1) - y[..., 0, :].sum(-1)) / 4
    context = (y[..., :, 1].sum(-1) - y[..., :, 0].sum(-1)) / 4
    joint = (y[..., 1, 1] - y[..., 0, 1] - y[..., 1, 0] + y[..., 0, 0]) / 4
    return mean, body, context, joint


def sigmoid(values):
    """Evaluate a stable binary link without fitting or clipping an effect."""
    x = np.asarray(values, float)
    if not np.isfinite(x).all():
        raise ValueError("Finite log odds required")
    return (1 + np.tanh(x / 2)) / 2


def context_pair_account(old_log_odds, common_log_odds):
    """Separate an average log-odds offset from body-dependent context response."""
    a, b = np.asarray(old_log_odds, float), np.asarray(common_log_odds, float)
    if a.shape != (2,) or b.shape != (2,) or not np.isfinite([a, b]).all():
        raise ValueError("Aligned low/high log odds under both contexts required")
    mean, body, context, joint = context_characters(np.stack((a, b), -1))
    shift = float((b - a).mean())
    predicted = sigmoid(a + shift)
    actual, old = sigmoid(b), sigmoid(a)
    return {
        "mean_log_odds": float(mean),
        "body_character": float(body),
        "context_character": float(context),
        "joint_character": float(joint),
        "old_log_odds_gap": float(a[1] - a[0]),
        "common_log_odds_gap": float(b[1] - b[0]),
        "log_odds_gap_change": float((b[1] - b[0]) - (a[1] - a[0])),
        "old_probability_gap": float(old[1] - old[0]),
        "common_probability_gap": float(actual[1] - actual[0]),
        "probability_gap_change": float((actual[1] - actual[0]) - (old[1] - old[0])),
        "offset_only_probability_gap_change": float(
            (predicted[1] - predicted[0]) - (old[1] - old[0])
        ),
        "joint_probability_remainder": float(
            (actual[1] - actual[0]) - (predicted[1] - predicted[0])
        ),
        "body_only_two_context_floor": float(np.mean((actual - old) ** 2) / 4),
    }
