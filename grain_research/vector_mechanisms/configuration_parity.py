#!/usr/bin/env python3
"""Exact odd/even projections under a comment-centered sign reflection."""

import numpy as np


def parity_components(positive, negative, origin):
    """Split the same fixed-query function into reflection-odd and reflection-even parts."""
    p, n, o = (np.asarray(x, float) for x in (positive, negative, origin))
    if len({x.shape for x in (p, n, o)}) != 1 or not all(
        np.isfinite(x).all() for x in (p, n, o)
    ):
        raise ValueError("Finite same-shape endpoint scores required")
    plus, minus = p - o, n - o
    return {
        "plus": plus,
        "minus": minus,
        "odd": (plus - minus) / 2,
        "even": (plus + minus) / 2,
    }


def paired_even_error(positive, negative, even_estimate):
    """Account for the error of one even readout on two equally weighted directions."""
    p, n, h = (np.asarray(x, float) for x in (positive, negative, even_estimate))
    if len({x.shape for x in (p, n, h)}) != 1:
        raise ValueError("Matched paired effects and even readout required")
    odd, even = (p - n) / 2, (p + n) / 2
    return {
        "paired_mse": float(np.mean(((p - h) ** 2 + (n - h) ** 2) / 2)),
        "even_mse": float(np.mean((even - h) ** 2)),
        "odd_energy": float(np.mean(odd**2)),
    }
