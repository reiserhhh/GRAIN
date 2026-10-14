#!/usr/bin/env python3
"""Distinguish natural actor exchanges from arbitrary partial-cache actions."""

from __future__ import annotations

import numpy as np


def exchange_parts(coefficients, axis):
    """Split complete finite actions by the parity required of a natural exchange."""
    x = np.asarray(coefficients, float)
    if axis not in ("e", "n") or x.shape[-1:] != (8,) or not np.isfinite(x).all():
        raise ValueError(
            "Eight complete finite characters and a registered axis required"
        )
    bit = 1 if axis == "e" else 2
    odd = np.array([bool(mask & bit) for mask in range(8)])
    total = float(np.sum(x**2))
    even_energy = float(np.sum(x[..., ~odd] ** 2))
    return {
        "even": np.where(odd, 0, x),
        "odd": np.where(odd, x, 0),
        "even_coefficient_rms": float(np.sqrt(np.mean(x[..., ~odd] ** 2))),
        "odd_coefficient_rms": float(np.sqrt(np.mean(x[..., odd] ** 2))),
        "even_energy_fraction": even_energy / total if total else None,
        "total_energy": total,
    }
