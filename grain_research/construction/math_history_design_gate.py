#!/usr/bin/env python3
"""Exact observation-law counterexamples to automatic information doubling."""

from __future__ import annotations

import itertools
from fractions import Fraction as F


def event_kernel(a, b, covariance=F(0)):
    """Return (C,Y) probabilities for Bernoulli marginals a,b and Y=C xor R."""
    probabilities = [
        (1 - a) * (1 - b) + covariance,
        (1 - a) * b - covariance,
        a * b + covariance,
        a * (1 - b) - covariance,
    ]
    if min(probabilities) < 0 or sum(probabilities) != 1:
        raise ValueError("Invalid conditional event kernel")
    return probabilities


def history_law(support, mass, mode):
    """Integrate all three-event histories; independent episodes given p."""
    kernels = []
    for p in support:
        if mode == "shared_independent":
            a, b, covariance = p, p, F(0)
        elif mode == "distinct_parameters":
            a, b, covariance = F(1, 2), p, F(0)
        elif mode == "copied_channel":
            a, b, covariance = p, p, p * (1 - p)
        elif mode == "balanced_dependence":
            a, b, covariance = p, p, F(-1, 50) if p < F(1, 2) else F(1, 50)
        else:
            raise ValueError("Unknown declared observation law")
        kernels.append(event_kernel(a, b, covariance))
    return [
        sum(
            w * kernel[i] * kernel[j] * kernel[k]
            for w, kernel in zip(mass, kernels, strict=True)
        )
        for i, j, k in itertools.product(range(4), repeat=3)
    ]


def source_observable_checks(law):
    """Compute gates from observed histories, without latent source parameters.

    Zero low-order checks are necessary, not sufficient, for conditional
    independence. The weighted check retains a concrete cancellation witness.
    """
    diagonal, covariance, weighted = F(0), F(0), F(0)
    for sequence, probability in zip(
        itertools.product(range(4), repeat=3), law, strict=True
    ):
        c = [address // 2 for address in sequence]
        r = [(address // 2) ^ (address % 2) for address in sequence]
        diagonal += probability * (c[0] - r[0]) * (c[1] - r[1])
        covariance += probability * c[0] * (r[0] - r[1])
        weighted += probability * c[0] * c[1] * (r[1] - r[2])
    return {
        "cross_event_parameter_gap": diagonal,
        "within_event_excess_covariance": covariance,
        "choice_weighted_excess_covariance": weighted,
    }
