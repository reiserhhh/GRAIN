#!/usr/bin/env python3
"""Exact two-context tangent recursion for specified, possibly noncommuting operators."""

import numpy as np


def symmetric_step(a, b, x, y):
    """Separate a propagated existing difference from the context-operator difference."""
    a, b, x, y = (np.asarray(v, float) for v in (a, b, x, y))
    if (
        a.ndim != 2
        or a.shape[0] != a.shape[1]
        or b.shape != a.shape
        or x.ndim != 2
        or y.shape != x.shape
        or x.shape[1] != a.shape[0]
    ):
        raise ValueError("Matched square operators and row-vector responses required")
    mean = (a + b) / 2
    carried = (x - y) @ mean.T
    injected = (x + y) / 2 @ (a - b).T
    return carried, injected


def terminal_parts(operators, initial):
    """Transport each injected difference through later mean operators in order.

    No inverses or commutativity are assumed. This is an identity for supplied
    local Jacobians; it neither estimates them nor replaces a nonlinear model.
    """
    a, d = np.asarray(operators, float), np.asarray(initial, float)
    if (
        a.ndim != 4
        or a.shape[0] != 2
        or a.shape[2] != a.shape[3]
        or d.ndim != 3
        or d.shape[0] != 2
        or d.shape[2] != a.shape[2]
        or not np.isfinite(a).all()
        or not np.isfinite(d).all()
    ):
        raise ValueError(
            "Finite paired operator sequences and initial directions required"
        )
    x, y = d[0].copy(), d[1].copy()
    parts = [x - y]
    local = []
    for left, right in zip(a[0], a[1], strict=True):
        _, injected = symmetric_step(left, right, x, y)
        mean = (left + right) / 2
        parts = [p @ mean.T for p in parts] + [injected]
        local.append(injected)
        x, y = x @ left.T, y @ right.T
    return {
        "difference": x - y,
        "terminal_parts": np.asarray(parts),
        "local_injections": np.asarray(local),
    }
