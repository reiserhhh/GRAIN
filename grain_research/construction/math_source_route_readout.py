#!/usr/bin/env python3
"""Reference-declared relative word evidence and calibration on saved routes."""

from __future__ import annotations

import numpy as np

from grain_research.construction import math_late_condition as L


def reference_covectors(token_ids, covectors, bias):
    """Build both predeclared observed-support references without target fitting."""
    ids = np.asarray(token_ids)
    w, b = np.asarray(covectors, float), np.asarray(bias, float)
    if (
        ids.ndim != 1
        or ids.dtype.kind not in "iu"
        or not len(ids)
        or w.ndim != 2
        or len(w) != len(ids)
        or b.shape != ids.shape
        or not np.isfinite(w).all()
        or not np.isfinite(b).all()
    ):
        raise ValueError("Complete actual-token output basis required")
    _, first, inverse = np.unique(ids, return_index=True, return_inverse=True)
    if not np.array_equal(w, w[first][inverse]) or not np.array_equal(
        b, b[first][inverse]
    ):
        raise ValueError("One token ID must have exactly one frozen covector")
    return {
        "occurrence": (w.mean(0), float(b.mean())),
        "unique": (w[first].mean(0), float(b[first].mean())),
    }


def relative_parts(chosen, partition, reference_logits):
    """Keep gauge-invariant selected relative evidence AND scalar calibration."""
    z, a, m = [np.asarray(v, float) for v in (chosen, partition, reference_logits)]
    if z.shape != m.shape or not np.isfinite(m).all():
        raise ValueError("Reference logits must address all actual observations")
    account = L.accounts(z - m, a - m)
    return {
        "shape": account["numerator"],
        "calibration": -account["normalizer"],
        "total": account["effect"],
    }


def state_difference(states):
    """Take own/foreign by high/low contrasts at every prediction position."""
    h = np.asarray(states, float)
    if h.ndim != 4 or h.shape[:2] != (2, 11) or not np.isfinite(h).all():
        raise ValueError("Complete own/foreign x eleven-voice state family required")
    return (h[0, 1::2] - h[0, 2::2]) - (h[1, 1::2] - h[1, 2::2])


def support_oscillation(delta, covectors):
    """Measure alternative-word field range on observed tokens, not full vocab."""
    d, w = np.asarray(delta, float), np.asarray(covectors, float)
    if d.ndim != 3 or w.ndim != 2 or d.shape[-1] != w.shape[-1]:
        raise ValueError("Aligned target/position/hidden and token basis required")
    if not np.isfinite(d).all() or not np.isfinite(w).all() or not len(w):
        raise ValueError("Finite nonempty observation support required")
    fields = d.reshape(-1, d.shape[-1]) @ w.T
    return np.ptp(fields, axis=1).reshape(d.shape[:2])
