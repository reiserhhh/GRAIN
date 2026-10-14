#!/usr/bin/env python3
"""Exact position ideals and source-identified token-gradient fields for HM28."""

from __future__ import annotations

from fractions import Fraction

import numpy as np


def positions(count):
    """Return exact normalized positions, retaining the singleton convention."""
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("positive integer token count required")
    return (
        (Fraction(0),)
        if count == 1
        else tuple(Fraction(2 * j, count - 1) - 1 for j in range(count))
    )


def multiply(a, b):
    """Multiply ascending rational coefficient vectors without floating rank tests."""
    c = [Fraction(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            c[i + j] += Fraction(x) * Fraction(y)
    return c


def remainder(a, modulus):
    """Compute the polynomial remainder in exact rational arithmetic."""
    a, p = list(map(Fraction, a)), list(map(Fraction, modulus))
    if len(p) < 2 or not p[-1]:
        raise ValueError(
            "nonconstant modulus with nonzero leading coefficient required"
        )
    while len(a) >= len(p):
        scale, shift = a[-1] / p[-1], len(a) - len(p)
        for j, value in enumerate(p):
            a[shift + j] -= scale * value
        a.pop()
    return a + [Fraction(0)] * (len(p) - 1 - len(a))


def certificate(counts, degree=3):
    """Certify the ideal of source positions and its truncated identifiable rank."""
    if not counts or not isinstance(degree, int) or degree < 0:
        raise ValueError("source counts and nonnegative degree required")
    points = sorted({p for count in counts for p in positions(count)})
    polynomial = [Fraction(1)]
    for point in points:
        polynomial = multiply(polynomial, [-point, Fraction(1)])
    reduction = [
        remainder([Fraction(0)] * k + [Fraction(1)], polynomial)
        for k in range(degree + 1)
    ]
    return {
        "positions": points,
        "polynomial": polynomial,
        "reduction": np.array(reduction, dtype=object).T,
        "rank": min(degree + 1, len(points)),
        "nullity": max(0, degree + 1 - len(points)),
    }


def interpolate(points, values, degree=3):
    """Choose the canonical degree<n representative, not a Euclidean extension."""
    points = tuple(map(Fraction, points))
    v = np.asarray(values, float)
    if (
        not points
        or len(set(points)) != len(points)
        or len(points) > degree + 1
        or v.ndim != 2
        or len(v) != len(points)
        or not np.isfinite(v).all()
    ):
        raise ValueError("distinct supported nodes and finite matching values required")
    basis = np.zeros((degree + 1, len(points)))
    for j, point in enumerate(points):
        coefficients, denominator = [Fraction(1)], Fraction(1)
        for k, other in enumerate(points):
            if k != j:
                coefficients = multiply(coefficients, [-other, Fraction(1)])
                denominator *= point - other
        basis[: len(coefficients), j] = [float(x / denominator) for x in coefficients]
    return basis @ v


def fit_node_field(gradients):
    """Group source constraints by position with exact original family/token weights."""
    gs = [np.asarray(g, float) for g in gradients]
    if not gs or any(
        g.ndim != 2
        or not np.isfinite(g).all()
        or not len(g)
        or g.shape[1] != gs[0].shape[1]
        for g in gs
    ):
        raise ValueError("matched finite source token gradients required")
    cert = certificate([len(g) for g in gs])
    if len(cert["positions"]) > 4:
        raise ValueError("independent node law would exceed the declared cubic model")
    lookup = {p: i for i, p in enumerate(cert["positions"])}
    numerator = np.zeros((len(lookup), gs[0].shape[1]))
    denominator = [Fraction(0)] * len(lookup)
    for gradient in gs:
        count = len(gradient)
        weight = Fraction(1, len(gs) * count**3)
        for p, g in zip(positions(count), gradient, strict=True):
            i = lookup[p]
            numerator[i] += float(weight * count) * g
            denominator[i] += weight
    values = numerator / np.array(list(map(float, denominator)))[:, None]
    return {
        "positions": cert["positions"],
        "values": values,
        "coefficients": interpolate(cert["positions"], values),
        "weights": denominator,
    }


def node_action(field, delta):
    """Evaluate the identified action only at source-observed token positions."""
    d = np.asarray(delta, float)
    if (
        d.ndim != 2
        or not np.isfinite(d).all()
        or d.shape[1] != field["values"].shape[1]
    ):
        raise ValueError("matched finite token displacement required")
    lookup = {p: i for i, p in enumerate(field["positions"])}
    total = 0.0
    for p, h in zip(positions(len(d)), d, strict=True):
        if p not in lookup:
            raise ValueError("new position is not source-identified")
        total += float(field["values"][lookup[p]] @ h) / len(d)
    return total


def exact_witness(count, polynomial):
    """Give rational source-null exposure for displacement h_j=p(t_j)."""
    points = positions(count)
    values = [sum(c * p**k for k, c in enumerate(polynomial)) for p in points]
    return {
        "positions": points,
        "values": values,
        "single_action": sum(v * v for v in values) / count,
        "paired_action_difference": 2 * sum(v * v for v in values) / count,
    }
