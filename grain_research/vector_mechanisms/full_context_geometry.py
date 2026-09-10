#!/usr/bin/env python3
"""Full hidden-state distance modulation of the frozen scalar token dictionary."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist

from . import hermite_token_potential as hermite
from .context_modulated_potential import multiplier, view_weights
from .position_reader import position_design
from .token_potential import last_gradient as additive_last_gradient

ARM = "full_context_geometry"


def source_context(lasts, additive):
    """Keep original source last states and a single source-only distance bandwidth."""
    x = np.asarray(lasts, float)
    if x.ndim != 2 or len(x) != len(additive["members"]) or not np.isfinite(x).all():
        raise ValueError("Aligned finite source context states required")
    if len(additive["landmarks"]) % len(x):
        raise ValueError("Equal source landmark ownership required")
    d = pdist(x)
    positive = d[d > 0]
    if not len(positive):
        raise ValueError("Undefined positive source context bandwidth")
    return {
        "raw_contexts": x.copy(),
        "context_mean": x.mean(0),
        "context_bandwidth": np.array(np.median(positive)),
    }


def context_terms(last, model):
    """Owner-level centered kernels and analytic full last-state derivatives."""
    x, sigma = model["raw_contexts"], float(model["context_bandwidth"])
    last = np.asarray(last, float)
    if last.shape != x.shape[1:] or not np.isfinite(last).all():
        raise ValueError("Finite full last state required")
    delta = last - x
    k = np.exp(-np.sum(delta**2, axis=1) / (2 * sigma**2))
    k0 = np.exp(-np.sum((model["context_mean"] - x) ** 2, axis=1) / (2 * sigma**2))
    return k - k0, -k[:, None] * delta / sigma**2


def design(body, last, model):
    """Add the same number of conditional features without changing the prefix."""
    z, l = hermite.coordinates(body, last, model)
    p = position_design(len(z), 3)[:, 1:]
    phi = hermite.scalar_design(z, l, p, model)
    gamma, _ = context_terms(last, model)
    gamma = np.repeat(gamma, len(model["landmarks"]) // len(gamma))
    extra = phi[1 + 2 * z.shape[1] :] * multiplier(gamma, z.shape[1])
    return z, p, np.r_[phi, extra]


def density_blocks(z, p, last, model):
    """Stream BODY derivatives; no full hidden-width by landmark tensors."""
    gamma, _ = context_terms(last, model)
    gamma = np.repeat(gamma, len(model["landmarks"]) // len(gamma))
    weights = multiplier(gamma, z.shape[1])
    for rows, d in hermite.density_blocks(z, p, model, chunk_size=64):
        yield rows, np.concatenate((d, d[:, :, 1 + 2 * z.shape[1] :] * weights), axis=2)


def scalar(body, last, model):
    """Evaluate the single compatible scalar used for all derivatives."""
    return float(design(body, last, model)[2] @ model["coef"])


def predict(body, last, model):
    """Read score and BODY density from current inputs and a source-frozen function."""
    z, p, phi = design(body, last, model)
    if phi.shape != model["coef"].shape:
        raise ValueError("Conditional layout differs")
    return float(phi @ model["coef"]), np.concatenate(
        [d @ model["coef"] for _, d in density_blocks(z, p, last, model)]
    )


def last_gradient(body, last, model):
    """Differentiate original last term plus full-distance conditional modulation."""
    z, l = hermite.coordinates(body, last, model)
    phi = hermite.scalar_design(z, l, position_design(len(z), 3)[:, 1:], model)
    rank, m = z.shape[1], len(model["landmarks"])
    e = model["coef"][len(phi) :]
    prefix = 1 + 2 * rank
    per_center = phi[prefix : prefix + m] * e[:m]
    per_center += np.sum(
        phi[prefix + m :].reshape(m, rank) * e[m:].reshape(m, rank), axis=1
    )
    _, derivative = context_terms(last, model)
    per_owner = per_center.reshape(len(derivative), -1).sum(1)
    return additive_last_gradient(model) + per_owner @ derivative


def geometry_account(last, model):
    """Diagnostic-only span decomposition and exact common radial factorization."""
    x = model["raw_contexts"] - model["context_mean"]
    _, s, vt = np.linalg.svd(x, full_matrices=False)
    basis = vt[s > s[0] * 1e-10].T
    delta = last - model["context_mean"]
    inside = (delta @ basis) @ basis.T
    outside = delta - inside
    sigma = float(model["context_bandwidth"])
    radial = float(np.exp(-(outside @ outside) / (2 * sigma**2)))
    kin = np.exp(-np.sum((inside - x) ** 2, axis=1) / (2 * sigma**2))
    full = np.exp(-np.sum((last - model["raw_contexts"]) ** 2, axis=1) / (2 * sigma**2))
    return {
        "inside_norm": float(np.linalg.norm(inside)),
        "outside_norm": float(np.linalg.norm(outside)),
        "radial_factor": radial,
        "in_span_kernel_mean": float(kin.mean()),
        "full_kernel_mean": float(full.mean()),
        "factorization_error": float(np.max(abs(full - radial * kin))),
    }


def fit(bodies, lasts, gradients, scores, author_ids, original_lasts, additive):
    """One conditional solve; independently verify the frozen additive normal block."""
    if not bodies or any(
        len(v) != len(bodies) for v in (lasts, gradients, scores, author_ids)
    ):
        raise ValueError("Aligned source views required")
    if set(author_ids) != set(additive["members"]):
        raise ValueError("Source membership differs")
    model = {k: v for k, v in additive.items() if not k.startswith("context_")}
    model |= source_context(original_lasts, additive) | {"arm": np.array(ARM)}
    u, sd = model["basis"], model["scale"]
    sl, sk = float(model["score_scale"]), float(model["density_scale"])
    alpha = float(model["alpha"])
    oldwidth = len(additive["coef"])
    width = oldwidth + len(model["landmarks"]) * (u.shape[1] + 1)
    fg, fr = np.zeros((width, width)), np.zeros(width)
    vg, vr = np.zeros_like(fg), np.zeros(width)
    fc, vc = 0.0, 0.0
    for h, last, g, y, w in zip(
        bodies, lasts, gradients, scores, view_weights(author_ids), strict=True
    ):
        z, p, phi = design(h, last, model)
        if g.shape != h.shape:
            raise ValueError("Native BODY shape differs")
        k = len(h) * (g @ u) * sd
        vg += w * np.outer(phi, phi) / sl**2
        vr += w * phi * y / sl**2
        vc += w * y**2 / sl**2
        fc += w * np.mean(k**2) / sk**2
        norm = np.sqrt(w / k.size) / sk
        for rows, d in density_blocks(z, p, last, model):
            a, b = d.reshape(-1, width) * norm, k[rows].ravel() * norm
            fg += a.T @ a
            fr += a.T @ b
    gram, rhs = fg + vg, fr + vr
    block_error = float(
        max(
            np.max(abs(gram[:oldwidth, :oldwidth] - additive["gram"])),
            np.max(abs(rhs[:oldwidth] - additive["rhs"])),
        )
    )
    if block_error > 1e-9:
        raise ValueError("Frozen additive normal equations changed")
    penalty = np.eye(width) * alpha
    penalty[0, 0] = 0
    coef = np.linalg.solve(gram + penalty, rhs)
    normal_error = float(np.max(abs((gram + penalty) @ coef - rhs)))

    def losses(theta):
        return np.array(
            [
                vc - 2 * theta @ vr + theta @ vg @ theta,
                fc - 2 * theta @ fr + theta @ fg @ theta,
                alpha * np.sum(theta[1:] ** 2),
            ]
        )

    loss, oldloss = (
        losses(coef),
        losses(np.r_[additive["coef"], np.zeros(width - oldwidth)]),
    )
    if (
        not np.isfinite(coef).all()
        or normal_error > 1e-8
        or loss.sum() > oldloss.sum() + 1e-8
    ):
        raise ValueError("Scalar nested objective failure")
    model |= {"coef": coef, "gram": gram, "rhs": rhs}
    return model, {
        "candidate": ARM,
        "parameters": width,
        "source_authors": len(set(author_ids)),
        "source_views": len(bodies),
        "source_score_loss": float(loss[0]),
        "source_density_loss": float(loss[1]),
        "source_penalty": float(loss[2]),
        "source_objective": float(loss.sum()),
        "source_zero_density_loss": float(fc),
        "frozen_additive_objective": float(oldloss.sum()),
        "additive_normal_block_error": block_error,
        "normal_equation_error": normal_error,
        "context_bandwidth": float(model["context_bandwidth"]),
        "score_scale": sl,
        "density_scale": sk,
    }
