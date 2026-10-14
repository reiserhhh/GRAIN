#!/usr/bin/env python3
"""The observable quotient of a fixed categorical linear output head.

The kernel is invisible to this head on its normalized hidden-state domain.
It is not psychological noise, all-model invisibility, or an intervention
claim about unnormalized upstream activations.
"""

from __future__ import annotations

import numpy as np


def quotient_basis(head, tolerance=1e-10):
    """Return a fixed Euclidean row-space basis after the exact logit gauge."""
    w = np.asarray(head, float)
    if w.ndim != 2 or min(w.shape) < 2 or not np.isfinite(w).all() or tolerance <= 0:
        raise ValueError("finite categorical head and positive tolerance required")
    centered = w - w.mean(axis=0)
    u, singular, _ = np.linalg.svd(centered.T, full_matrices=False)
    rank = int((singular > tolerance * singular[0]).sum()) if singular[0] else 0
    return {
        "head": centered,
        "basis": u[:, :rank],
        "rank": rank,
        "singular_values": singular,
    }


def split_displacement(displacement, quotient):
    """Keep both observable and head-null displacement; do not discard either."""
    d = np.asarray(displacement, float)
    basis = quotient["basis"]
    if d.ndim != 1 or d.shape != (basis.shape[0],) or not np.isfinite(d).all():
        raise ValueError(
            "finite hidden displacement in the declared head space required"
        )
    visible = basis @ (basis.T @ d)
    hidden = d - visible
    return {
        "observable": visible,
        "head_null": hidden,
        "full_energy": float(d @ d),
        "observable_energy": float(visible @ visible),
        "head_null_energy": float(hidden @ hidden),
        "null_action_max": float(np.max(abs(quotient["head"] @ hidden))),
        "energy_closure_error": abs(float(d @ d - visible @ visible - hidden @ hidden)),
    }
