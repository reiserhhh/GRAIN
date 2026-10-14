#!/usr/bin/env python3
"""Exchange sectors and ordered observations, with no fit or target labels."""

from __future__ import annotations

import itertools

import numpy as np

from suica_core.suica import TOKEN_RE


def four_slices(text):
    """Reproduce the inherited four lexical cuts, returning missing as empty."""
    spans = list(TOKEN_RE.finditer(text))
    if len(spans) < 48:
        return []
    edges = [0, *[spans[int(k * len(spans) / 4)].start() for k in (1, 2, 3)], len(text)]
    parts = [text[a:b] for a, b in itertools.pairwise(edges)]
    if "".join(parts) != text or min(len(TOKEN_RE.findall(p)) for p in parts) < 12:
        raise ValueError("Inherited lexical cut lost text or support")
    return parts


def event_slices(text):
    """Use the three existing public record boundaries, without parsing roles."""
    parts = text.splitlines(keepends=True)
    if len(parts) != 3 or "".join(parts) != text:
        raise ValueError("Exactly three lossless designed events required")
    if min(len(TOKEN_RE.findall(p)) for p in parts) < 12:
        raise ValueError("Event support below frozen instrument minimum")
    return parts


def event_observations(path):
    """Return mean, directed step, and signed triangular area of three points."""
    x = np.asarray(path, float)
    if x.shape != (3, 4) or not np.isfinite(x).all():
        raise ValueError("Three finite four-factor event points required")
    a, b = x[0] - x[2], x[1] - x[2]
    area = np.array(
        [
            (a[i] * b[j] - a[j] * b[i]) / 2
            for i, j in itertools.combinations(range(4), 2)
        ]
    )
    return x.mean(0), x[1] - x[0], area


def parity(kernel):
    """Project a full three-state kernel onto exchange-even and -odd parts."""
    f = np.asarray(kernel, float)
    if f.ndim < 2 or f.shape[:2] != (3, 3) or not np.isfinite(f).all():
        raise ValueError("Complete finite three-state observer required")
    reverse = f.swapaxes(0, 1)
    return (f + reverse) / 2, (f - reverse) / 2


def cycle_action(kernel):
    """Return the fixed plus-minus 45-edge law contrast in observer units."""
    f = np.asarray(kernel, float)
    parity(f)
    return (2 / 15) * (f[0, 1] + f[1, 2] + f[2, 0] - f[1, 0] - f[2, 1] - f[0, 2])


def integrate(path, kernel):
    """Read an existing closed numerical author ledger, not a trait label."""
    p = np.asarray(path)
    f = np.asarray(kernel, float)
    parity(f)
    if (
        p.ndim != 1
        or len(p) < 2
        or not np.equal(p, np.floor(p)).all()
        or np.min(p) < 0
        or np.max(p) > 2
    ):
        raise ValueError("Finite integer three-state closed path required")
    p = p.astype(int)
    return f[p, np.roll(p, -1)].mean(0)
