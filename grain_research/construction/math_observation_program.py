#!/usr/bin/env python3
"""Finite program/evidence contrasts; no predictor fitting or champion selection."""

from __future__ import annotations

import itertools

import numpy as np


def validate_cube(cube):
    """Require complete program/b/c support and finite primitive units."""
    x = np.asarray(cube, dtype=np.float64)
    if x.shape[:3] != (2, 2, 2) or not np.isfinite(x).all():
        raise ValueError("Complete two-program, b/c support required")
    return x


def effects(cube):
    """Keep both slopes, condition level and their signed interaction."""
    x = validate_cube(cube)
    slopes = x[:, 1] - x[:, 0]
    level = (x[:, :, 1] - x[:, :, 0]).mean(axis=1)
    return np.stack(
        (slopes[:, 0], slopes[:, 1], level, slopes[:, 1] - slopes[:, 0]), axis=1
    )


def characters(cube):
    """Retain all eight program/b/c characters, without averaging away program."""
    x = validate_cube(cube)
    signs = np.array(list(itertools.product((-1, 1), repeat=3)))
    flat = x.reshape(8, *x.shape[3:])
    return np.stack(
        [
            np.einsum(
                "i,i...->...",
                np.prod(signs[:, [j for j in range(3) if mask & (1 << j)]], axis=1),
                flat,
            )
            / 8
            for mask in range(8)
        ]
    )


def reconstruct(gamma):
    """Invert all character coefficients on the original registered corners."""
    x = np.asarray(gamma, dtype=np.float64)
    if x.shape[0] != 8 or not np.isfinite(x).all():
        raise ValueError("Eight finite coefficients required")
    corners = []
    for signs in itertools.product((-1, 1), repeat=3):
        corners.append(
            sum(
                x[mask] * np.prod([signs[j] for j in range(3) if mask & (1 << j)])
                for mask in range(8)
            )
        )
    return np.stack(corners).reshape(2, 2, 2, *x.shape[1:])


def field_comparison(old, full):
    """Describe the entire signed field without assuming improvement or causality."""
    a, b = np.asarray(old, dtype=float), np.asarray(full, dtype=float)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Matched finite fields required")
    aa, bb = float(np.sum(a * a)), float(np.sum(b * b))
    return {
        "H_rms": float(np.sqrt(np.mean(a * a))),
        "full_rms": float(np.sqrt(np.mean(b * b))),
        "difference_rms": float(np.sqrt(np.mean((b - a) ** 2))),
        "cosine": float(np.sum(a * b) / np.sqrt(aa * bb)) if aa and bb else None,
        "full_signed_projection_on_H": float(np.sum(a * b) / aa) if aa else None,
        "relative_difference_to_H": float(np.sqrt(np.sum((b - a) ** 2) / aa))
        if aa
        else None,
    }


def zero_value_carrier_example():
    """Known softmax example: a zero value still alters a conditional contrast."""
    cube = np.empty((2, 2, 2))
    for r, b, c in itertools.product(range(2), repeat=3):
        chosen = 2 * b - 1
        cube[r, b, c] = chosen if r == 0 else chosen / (1 + (1 if c == 0 else 3))
    return cube
