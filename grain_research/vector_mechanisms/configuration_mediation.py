#!/usr/bin/env python3
"""Finite two-factor allocation for an explicitly defined hidden-state game."""

from contextlib import contextmanager

import numpy as np


def factorial_allocation(cells):
    """Allocate F11-F00 by averaging the two orders; retain interaction."""
    values = np.asarray(cells, dtype=float)
    if values.ndim < 1 or values.shape[-1] != 4 or not np.isfinite(values).all():
        raise ValueError("Finite cells in F11,F00,F10,F01 order required")
    f11, f00, f10, f01 = np.moveaxis(values, -1, 0)
    native_head = f11 - f10
    projected_head = f01 - f00
    head = 0.5 * (native_head + projected_head)
    other = 0.5 * ((f11 - f01) + (f10 - f00))
    contrast = f11 - f00
    return {
        "configuration": contrast,
        "head_native_body": native_head,
        "head_projected_body": projected_head,
        "head_allocation": head,
        "other_allocation": other,
        "interaction": f11 - f10 - f01 + f00,
        "additive_error": head + other - contrast,
    }


def endpoint_errors(cells, native, projected, tolerance=2e-5):
    """Require the native/projected cells to reproduce frozen source endpoints."""
    values = np.asarray([*cells, native, projected], dtype=float)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise ValueError("Four finite cells and two source endpoints required")
    errors = np.abs(values[:2] - values[4:])
    if max(errors) > tolerance:
        raise ValueError("Source native/projected endpoint replay failed")
    return {"native": float(errors[0]), "projected": float(errors[1])}


@contextmanager
def projected_body(layer, groups):
    """Apply the exact S9 block-output mean projection and always remove its hook."""
    from scripts.run_grain_mean_null_path import mean_null_direction

    groups = [list(group) for group in groups]

    def boundary(_module, _inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        state = h + mean_null_direction(h[0], groups, list(range(len(groups))))[None]
        return (state, *output[1:]) if isinstance(output, tuple) else state

    handle = layer.register_forward_hook(boundary)
    try:
        yield
    finally:
        handle.remove()
