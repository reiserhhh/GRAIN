#!/usr/bin/env python3
"""HM25: finite decoder path integrals and readout/decoder error accounts."""

from __future__ import annotations

import numpy as np


def simpson_weights(nodes):
    """Return fixed composite Simpson weights on an odd uniform node count."""
    x = np.asarray(nodes, float)
    if x.ndim != 1 or len(x) < 3 or len(x) % 2 != 1 or not np.isfinite(x).all():
        raise ValueError("finite odd uniform nodes required")
    step = np.diff(x)
    if step[0] <= 0 or not np.allclose(step, step[0], atol=1e-14, rtol=1e-12):
        raise ValueError("strictly increasing uniform nodes required")
    w = np.ones(len(x))
    w[1:-1:2], w[2:-1:2] = 4, 2
    return w * step[0] / 3


def integrate(nodes, values):
    """Integrate matched scalar or vector rows without outcome-selected quadrature."""
    y = np.asarray(values, float)
    if y.ndim < 1 or len(y) != len(nodes) or not np.isfinite(y).all():
        raise ValueError("matched finite node values required")
    return np.tensordot(simpson_weights(nodes), y, axes=(0, 0))


def candidate_score_gradient(logits, alphabet):
    """Compute the finite candidate readout and its zero-sum logit covector."""
    z, a = np.asarray(logits, float), np.asarray(alphabet, float)
    if (
        z.ndim != 1
        or z.shape != a.shape
        or not np.isfinite(z).all()
        or not np.isfinite(a).all()
    ):
        raise ValueError("matched finite logits and alphabet required")
    p = np.exp(z - z.max())
    p /= p.sum()
    score = float(p @ a)
    return score, p * (a - score)


def finite_account(logits_0, logits_1, alphabet, tangent_0):
    """Separate decoder nonlinearity from known candidate-readout nonlinearity."""
    f0, g0 = candidate_score_gradient(logits_0, alphabet)
    f1, _ = candidate_score_gradient(logits_1, alphabet)
    actual = f1 - f0
    frozen = float(g0 @ (np.asarray(logits_1) - np.asarray(logits_0)))
    return {
        "native_action": actual,
        "start_tangent_action": float(tangent_0),
        "frozen_readout_logit_action": frozen,
        "decoder_nonlinear_increment": frozen - tangent_0,
        "readout_nonlinear_increment": actual - frozen,
        "full_nonlinear_increment": actual - tangent_0,
    }
