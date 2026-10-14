#!/usr/bin/env python3
"""Exact binary-input ratios with explicit floating-point summation bounds."""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np


def exact_ratio_audit(rows, field, actual):
    """Audit only an MS ratio; raw physical comparisons retain their1e-8 ceiling."""
    if not rows or not math.isfinite(actual):
        raise ValueError("finite nonempty ratio data required")
    f = Fraction.from_float
    denominator = sum((f(float(r["total"])) ** 2 for r in rows), Fraction(0))
    if denominator == 0:
        raise ValueError("zero ratio denominator cannot be audited as a number")
    if field == "link_decoder_cross_over_total_ms":
        terms = [2 * f(float(r["link"])) * f(float(r["decoder"])) for r in rows]
    elif field in ("link_ms_over_total_ms", "decoder_ms_over_total_ms"):
        name = field.split("_", 1)[0]
        terms = [f(float(r[name])) ** 2 for r in rows]
    else:
        raise ValueError("only the three declared dimensionless ratios allowed")
    numerator, magnitude = sum(terms, Fraction(0)), sum(map(abs, terms), Fraction(0))
    exact = numerator / denominator
    rounded = float(exact)
    eps, operations = np.finfo(float).eps, 2 * len(rows) + 8
    gamma = operations * eps / (1 - operations * eps)
    bound = gamma * (float(magnitude / denominator) + abs(rounded)) + math.ulp(rounded)
    error = float(abs(f(float(actual)) - exact))
    if error > bound:
        raise ValueError("ratio exceeds explicit double-reduction roundoff bound")
    rms = math.sqrt(float(denominator / len(rows)))
    return {
        "actual": float(actual),
        "exact_binary_input_ratio": rounded,
        "absolute_error": error,
        "roundoff_bound": bound,
        "total_rms": rms,
        "denominator_resolved_at_physical_1e8": rms > 1e-8,
        "operations_bound": operations,
    }
