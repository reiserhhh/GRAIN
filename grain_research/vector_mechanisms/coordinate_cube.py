#!/usr/bin/env python3
"""Fixed-context endpoint cube for query mean, configuration and final state."""

import itertools

import numpy as np

BITS = tuple(itertools.product((0, 1), repeat=3))
NAMES = {
    "100": "mean",
    "010": "configuration",
    "001": "final",
    "110": "mean_configuration",
    "101": "mean_final",
    "011": "configuration_final",
    "111": "triple",
}


def query_cube(p, q, edited_p, edited_q):
    """Split a paired natural displacement, preserving token identity and mean."""
    p, q, pe, qe = (np.asarray(x, float) for x in (p, q, edited_p, edited_q))
    if (
        p.ndim != 2
        or q.ndim != 1
        or p.shape != pe.shape
        or q.shape != qe.shape
        or p.shape[1] != len(q)
        or min(p.shape) < 1
        or not all(np.isfinite(v).all() for v in (p, q, pe, qe))
    ):
        raise ValueError("Require finite same-template P/Q endpoint pairs")
    delta = pe - p
    mean = np.broadcast_to(delta.mean(0), p.shape)
    configuration = delta - mean
    return {
        "".join(map(str, bits)): (
            p + bits[0] * mean + bits[1] * configuration,
            q + bits[2] * (qe - q),
        )
        for bits in BITS
    }


def anchored_effects(values):
    """Mobius inversion on a three-block Boolean cube; no independence assumed."""
    keys = {"".join(map(str, bits)) for bits in BITS}
    if set(values) != keys or not np.isfinite(list(values.values())).all():
        raise ValueError("Require eight finite cube values")
    effects = {}
    for bits in BITS[1:]:
        label = "".join(map(str, bits))
        value = 0.0
        for sub in BITS:
            if all(a <= b for a, b in zip(sub, bits, strict=True)):
                sign = (-1) ** (sum(bits) - sum(sub))
                value += sign * values["".join(map(str, sub))]
        effects[NAMES[label]] = float(value)
    if abs(sum(effects.values()) - (values["111"] - values["000"])) > 1e-9:
        raise ValueError("Cube accounting failed")
    return effects
