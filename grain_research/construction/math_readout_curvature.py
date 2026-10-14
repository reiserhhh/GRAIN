#!/usr/bin/env python3
"""Exact scalar softmax-link curvature and finite physical mixed actions."""

from __future__ import annotations

import numpy as np

COMPONENTS = ("link", "decoder", "roundoff")


def link_derivatives(logits, values):
    """Return a scalar expectation, gradient and indefinite logit Hessian."""
    l, a = np.asarray(logits, float), np.asarray(values, float)
    if l.ndim != 1 or l.shape != a.shape or not len(l):
        raise ValueError("one aligned nonempty logit/value alphabet required")
    if not np.isfinite(l).all() or not np.isfinite(a).all():
        raise ValueError("finite alphabet required")
    p = np.exp(l - l.max())
    p /= p.sum()
    score = float(p @ a)
    g = p * (a - score)
    h = np.diag(g) - np.outer(p, g) - np.outer(g, p)
    return score, p, g, h


def hessian_from_actual(probabilities, gradient):
    """Keep the actual float32 forward probabilities/weights as numerical inputs."""
    p, g = np.asarray(probabilities, float), np.asarray(gradient, float)
    if p.ndim != 1 or p.shape != g.shape or not np.isfinite(p + g).all():
        raise ValueError("finite aligned readout derivatives required")
    return np.diag(g) - np.outer(p, g) - np.outer(g, p)


def link_covectors(hessian, h_directional, p_jacobian):
    """Contract J_P^T Hess(m) J_H d without a full upstream Hessian."""
    h, dh, jp = (np.asarray(x, float) for x in (hessian, h_directional, p_jacobian))
    if h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError("square output Hessian required")
    if dh.ndim != 2 or dh.shape[1] != len(h) or jp.shape[0] != len(h):
        raise ValueError("aligned physical directions and digit Jacobians required")
    if not all(np.isfinite(x).all() for x in (h, dh, jp)):
        raise ValueError("finite Jacobians required")
    return np.tensordot(dh @ h.T, jp, axes=(1, 0))


def project_components(covectors, basis, delta_p, h_norms):
    """Separate source-supported coefficients and complements in physical units."""
    v, u, dp, norms = (
        np.asarray(x, float) for x in (covectors, basis, delta_p, h_norms)
    )
    if v.ndim != 3 or v.shape[0] != 2 or v.shape[1:] != dp.shape:
        raise ValueError("two physical H arms with the original P layout required")
    if u.ndim != 2 or u.shape[0] != dp.size or norms.shape != (2,):
        raise ValueError("aligned frozen basis and physical norms required")
    if not (norms > 0).all() or not all(
        np.isfinite(x).all() for x in (v, u, dp, norms)
    ):
        raise ValueError("finite physical arrays and positive norms required")
    coefficients = v.reshape(2, -1) @ u
    z = dp.ravel() @ u
    theta = coefficients / norms[:, None]
    eta = norms[:, None] * z[None, :]
    full = v.reshape(2, -1) @ dp.ravel()
    supported = np.sum(theta * eta, axis=1)
    return {
        "theta": theta,
        "eta": eta,
        "full": full,
        "supported": supported,
        "complement": full - supported,
        "coefficients": coefficients,
    }


def validate_layout(row, h_shape, p_shape):
    """Reject mismatched token identities, overlap and changed physical spans."""
    hp, pp = row["spans"]["A_property"]["positions"], row["early_query_positions"]
    if len(hp) != h_shape[0] or len(pp) != p_shape[0]:
        raise ValueError("token/physical layout mismatch")
    if (
        len(set(hp + pp)) != len(hp) + len(pp)
        or max(hp + pp) >= len(row["input_ids"]) - 1
    ):
        raise ValueError("distinct pre-endpoint coordinates required")
    if len(row["early_query_token_ids"]) != len(pp):
        raise ValueError("ordered query token identity required")


def metrics(rows):
    """Retain link/decoder energies and signed cross terms, including cancellation."""
    x = np.array([[r[k] for k in (*COMPONENTS, "total")] for r in rows], float)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError("nonempty finite addressed actions required")
    result = {"count": len(rows)}
    for i, k in enumerate((*COMPONENTS, "total")):
        result[k + "_rms"] = float(np.sqrt(np.mean(x[:, i] ** 2)))
        result[k + "_mean"] = float(x[:, i].mean())
    for i, k in enumerate(COMPONENTS):
        for j in range(i + 1, 3):
            result[k + "__" + COMPONENTS[j] + "_cross"] = float(
                2 * np.mean(x[:, i] * x[:, j])
            )
    result["sum_identity_error"] = float(np.max(abs(x[:, :3].sum(axis=1) - x[:, 3])))
    denominator = float(np.mean(x[:, 3] ** 2))
    result["link_ms_over_total_ms"] = (
        float(np.mean(x[:, 0] ** 2) / denominator) if denominator else None
    )
    result["decoder_ms_over_total_ms"] = (
        float(np.mean(x[:, 1] ** 2) / denominator) if denominator else None
    )
    result["link_decoder_cross_over_total_ms"] = (
        result["link__decoder_cross"] / denominator if denominator else None
    )
    return result
