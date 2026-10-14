#!/usr/bin/env python3
"""Opportunity-relative evidence; standard choice math, not human trait truth."""

from __future__ import annotations

import numpy as np


def choice_log_likelihood(features, theta, chosen, available):
    """Log probability in a declared finite exponential-family choice law."""
    f, t = np.asarray(features, dtype=float), np.asarray(theta, dtype=float)
    mask = np.asarray(available)
    if (
        f.ndim != 2
        or t.shape != (f.shape[1],)
        or mask.shape != (f.shape[0],)
        or mask.dtype != np.bool_
        or not isinstance(chosen, (int, np.integer))
        or not 0 <= chosen < f.shape[0]
        or not mask[chosen]
        or not np.isfinite(f).all()
        or not np.isfinite(t).all()
    ):
        raise ValueError(
            "Finite features, a valid chosen option and Boolean availability required"
        )
    utility = f @ t
    offered = utility[mask]
    shift = offered.max()
    return float(utility[chosen] - shift - np.log(np.exp(offered - shift).sum()))


def choice_log_bayes_factor(features, low, high, chosen, available):
    """Return a likelihood contrast; neither parameter is inferred from text."""
    return choice_log_likelihood(
        features, high, chosen, available
    ) - choice_log_likelihood(features, low, chosen, available)


def opportunity_effects(corners):
    """Contrast unchosen-option polarity with and without actual availability."""
    x = np.asarray(corners, dtype=float)
    if x.shape[:2] != (2, 2) or not np.isfinite(x).all():
        raise ValueError("Complete alternative-by-availability corners required")
    contrasts = x[1] - x[0]
    level = (x[:, 1] - x[:, 0]).mean(axis=0)
    return np.stack((contrasts[0], contrasts[1], level, contrasts[1] - contrasts[0]))


def validate_material_spec(spec):
    """Require the performed action to remain fixed across unchosen alternatives."""
    if spec.get("packet") != "HM135" or len(spec.get("banks", [])) != 2:
        raise ValueError("Registered HM135 two-bank material required")
    for bank in spec["banks"]:
        if len(bank["behaviours"]) != 6 or len(bank["causes"]) != 2:
            raise ValueError("Six families and two availability conditions required")
        for pair in bank["behaviours"]:
            if len(pair) != 2:
                raise ValueError("Both unchosen alternatives required")
            pieces = [s.split(". Unused option: ") for s in pair]
            if any(len(p) != 2 for p in pieces) or pieces[0][0] != pieces[1][0]:
                raise ValueError("Performed action changed across alternatives")
            if pieces[0][1] == pieces[1][1]:
                raise ValueError("Alternative polarity became constant")
        closed, opened = bank["causes"]
        if closed.replace("closed", "open") != opened or "not performed" not in opened:
            raise ValueError("Availability must be the only clarification edit")
