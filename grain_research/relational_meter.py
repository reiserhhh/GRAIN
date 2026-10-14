#!/usr/bin/env python3
"""Affine evidence mixtures and fractional-linear scores of a fixed finite reader.

These are population expectations under balanced known opportunities, not a
learned natural-language world model. Source controls calibrate the observation
operator; evaluated people or psychological labels are not constructor inputs.
"""

from __future__ import annotations

import itertools

import numpy as np

from grain_research.relational_evidence import SHAPE, query_kernel


def controlled_operator(controls):
    """Compile T(mu)=offset+operator@mu from a 4x2x2 factorial response bank.

    Axes before the flattened output are group=actor/channel, question pole,
    true response direction. Each group/pole is equally available; only response
    direction has probability (1 + sign*mu[group])/2. Every output sums to one.
    """
    a = np.asarray(controls, float)
    if a.shape != (4, 2, 2, 16) or not np.isfinite(a).all() or np.any(a < 0):
        raise ValueError("Complete finite 4x2x2 factorial controls required")
    if not np.allclose(a.sum(axis=-1), 1):
        raise ValueError("Every control must allocate unit candidate mass")
    offset = a.mean(axis=(0, 1, 2))
    operator = ((a[:, :, 1] - a[:, :, 0]).sum(axis=1) / 16).T
    return offset, operator


def expected_joint(offset, operator, response_means):
    """Apply the factorial expectation to four bounded group response means."""
    a, b, mu = np.asarray(offset), np.asarray(operator), np.asarray(response_means)
    if a.shape != (16,) or b.shape != (16, 4) or mu.shape != (4,):
        raise ValueError("Finite-reader dimensions disagree")
    if not all(np.isfinite(v).all() for v in (a, b, mu)) or np.any(abs(mu) > 1):
        raise ValueError("Response means must be finite and bounded")
    return (a + b @ mu).reshape(SHAPE)


def fractional_map(offset, operator, *, actor=0, channel=1, at=None):
    """Return numerator/denominator and exact derivative of a conditional score.

    The derivative includes changes in membership mass. Using only the numerator
    is incorrect when the reader's actor/channel attribution depends on content.
    """
    mu = np.zeros(4) if at is None else np.asarray(at, float)
    expected_joint(offset, operator, mu)
    numerator_kernel = query_kernel(actor, channel).ravel()
    membership = np.zeros(SHAPE)
    membership[actor, channel] = 1
    denominator_kernel = membership.ravel()
    n0, d0 = float(numerator_kernel @ offset), float(denominator_kernel @ offset)
    n, d = numerator_kernel @ operator, denominator_kernel @ operator
    value_n, value_d = n0 + n @ mu, d0 + d @ mu
    return {
        "numerator_offset": n0,
        "numerator_response": n,
        "denominator_offset": d0,
        "denominator_response": d,
        "score": float(value_n / value_d) if value_d > 0 else None,
        "derivative": (n * value_d - value_n * d) / value_d**2 if value_d > 0 else None,
    }


def enumerate_expectation(controls, response_means):
    """Direct factorial sum, independent of compiled coefficients, for checking."""
    result = np.zeros(16)
    for group, pole, sign in itertools.product(range(4), range(2), range(2)):
        probability = (1 + (2 * sign - 1) * response_means[group]) / 16
        result += probability * controls[group, pole, sign]
    return result.reshape(SHAPE)
