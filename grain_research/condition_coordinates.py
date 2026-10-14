#!/usr/bin/env python3
"""Affine changes of condition coordinates and their response parameters.

This is standard linear algebra, not a new psychometric theorem. It gives the
transport law only when the coordinate transformation is common and invertible.
"""

from __future__ import annotations

import numpy as np


def reexpress_response(level, response, shift, transform):
    """Transport m(z)=a+Bz under z_new=c+Tz without changing m or its target.

    Rows of B correspond to response functions. A stochastic condition reader or
    an outcome-dependent transformation does not satisfy this exact transport.
    """
    a, b, c, t = [np.asarray(v, float) for v in (level, response, shift, transform)]
    if (
        b.ndim != 2
        or a.shape != (b.shape[0],)
        or c.shape != (b.shape[1],)
        or t.shape != (b.shape[1], b.shape[1])
    ):
        raise ValueError("Aligned response-by-condition arrays required")
    if any(not np.isfinite(v).all() for v in (a, b, c, t)):
        raise ValueError("Finite coordinate map required")
    if np.linalg.matrix_rank(t) < len(c):
        raise ValueError("Condition transformation is not invertible")
    b_new = np.linalg.solve(t.T, b.T).T
    return a - b_new @ c, b_new


def selective_repeat_expectation(mean, events=6, extra_copies=18):
    """Expected signed report mean when the most positive event is repeated.

    All events are iid binary signs with expectation mean, every event is still
    observed, and selection affects copies only. This is not a missing-event model.
    """
    m = np.asarray(mean, float)
    if not np.isfinite(m).all() or np.any(abs(m) > 1):
        raise ValueError("Binary-sign expectation must be in [-1, 1]")
    if (
        not isinstance(events, int)
        or events < 1
        or not isinstance(extra_copies, int)
        or extra_copies < 0
    ):
        raise ValueError("Integer event/copy counts required")
    selected = 1 - 2 * ((1 - m) / 2) ** events
    return (events * m + extra_copies * selected) / (events + extra_copies)
