#!/usr/bin/env python3
"""A context-product scalar with compatible BODY, last and mixed derivatives."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist

from . import hermite_token_potential as hermite
from .position_reader import position_design
from .token_potential import last_gradient as additive_last_gradient

ARMS = ("additive_augmented", "context_product_augmented")


def context_reference(lasts, baseline):
    """Construct context coordinates from ORIGINAL source states, never targets."""
    x = np.asarray(lasts, float)
    members = np.asarray(baseline["members"])
    m = len(baseline["landmarks"])
    if x.ndim != 2 or len(x) != len(members) or not np.isfinite(x).all():
        raise ValueError("Aligned finite original source last states required")
    if len(set(members)) != len(x) or m % len(x):
        raise ValueError("Equal source landmark ownership required")
    mu = x.mean(0)
    _, s, vt = np.linalg.svd(x - mu, full_matrices=False)
    keep = np.flatnonzero(s > s[0] * 1e-10)[: min(3, len(x) - 1)]
    if not len(keep):
        raise ValueError("Rank-zero context reference")
    u = vt[keep].T.copy()
    u *= np.sign(u[np.argmax(abs(u), axis=0), np.arange(len(keep))])
    c = (x - mu) @ u
    sd = c.std(0)
    sd[sd < 1e-8] = 1
    c /= sd
    distances = pdist(c)
    positive = distances[distances > 0]
    if not len(positive):
        raise ValueError("Undefined context bandwidth")
    return {
        "context_mean": mu,
        "context_basis": u,
        "context_scale": sd,
        "context_centers": np.repeat(c, m // len(x), axis=0),
        "context_bandwidth": np.array(np.median(positive)),
        "context_members": members.copy(),
    }


def context_terms(last, model):
    """Return centered context modulation and its derivative in context coordinates."""
    c = ((last - model["context_mean"]) @ model["context_basis"]) / model[
        "context_scale"
    ]
    centers = model["context_centers"]
    sigma = float(model["context_bandwidth"])
    delta = c - centers
    k = np.exp(-np.sum(delta**2, axis=1) / (2 * sigma**2))
    k0 = np.exp(-np.sum(centers**2, axis=1) / (2 * sigma**2))
    return k - k0, -k[:, None] * delta / sigma**2


def multiplier(gamma, rank):
    """Match the inherited all-kernel, then center-major Hermite feature layout."""
    return np.r_[gamma, np.repeat(gamma, rank)]


def design(body, last, model):
    """Construct the scalar design, leaving the entire additive prefix unchanged."""
    z, l = hermite.coordinates(body, last, model)
    p = position_design(len(z), 3)[:, 1:]
    phi = hermite.scalar_design(z, l, p, model)
    if str(model["arm"]) == ARMS[1]:
        gamma, _ = context_terms(last, model)
        prefix = 1 + 2 * z.shape[1]
        phi = np.r_[phi, phi[prefix:] * multiplier(gamma, z.shape[1])]
    return z, p, phi


def density_blocks(z, p, last, model):
    """Stream coordinate gradient designs; context stays fixed for BODY derivatives."""
    conditional = str(model["arm"]) == ARMS[1]
    prefix = 1 + 2 * z.shape[1]
    if conditional:
        gamma, _ = context_terms(last, model)
        weights = multiplier(gamma, z.shape[1])
    for rows, d in hermite.density_blocks(z, p, model, chunk_size=64):
        yield (
            rows,
            np.concatenate((d, d[:, :, prefix:] * weights), axis=2)
            if conditional
            else d,
        )


def scalar(body, last, model):
    """Evaluate the same scalar used for every reported derivative."""
    return float(design(body, last, model)[2] @ model["coef"])


def predict(body, last, model):
    """Return complete scalar and BODY coordinate density without teacher inputs."""
    z, p, phi = design(body, last, model)
    coef = model["coef"]
    if coef.shape != phi.shape or not np.isfinite(coef).all():
        raise ValueError("Conditional coefficient layout differs")
    return float(phi @ coef), np.concatenate(
        [d @ coef for _, d in density_blocks(z, p, last, model)]
    )


def last_gradient(body, last, model):
    """Differentiate the original last term AND the context-dependent local scalar."""
    result = additive_last_gradient(model)
    if str(model["arm"]) != ARMS[1]:
        return result
    z, l = hermite.coordinates(body, last, model)
    p = position_design(len(z), 3)[:, 1:]
    phi = hermite.scalar_design(z, l, p, model)
    rank, m = z.shape[1], len(model["landmarks"])
    extra = model["coef"][len(phi) :]
    prefix = 1 + 2 * rank
    effect = phi[prefix : prefix + m] * extra[:m]
    effect += np.sum(
        phi[prefix + m :].reshape(m, rank) * extra[m:].reshape(m, rank), axis=1
    )
    _, dg = context_terms(last, model)
    return result + ((effect @ dg) / model["context_scale"]) @ model["context_basis"].T


def view_weights(author_ids):
    """Equal author mass, then equal mass among that author's available views."""
    ids = np.asarray(author_ids)
    unique, counts = np.unique(ids, return_counts=True)
    if not len(unique):
        raise ValueError("Nonempty view membership required")
    count = dict(zip(unique, counts, strict=True))
    return np.array([1 / (len(unique) * count[i]) for i in ids])


def fit_pair(bodies, lasts, gradients, scores, author_ids, original_lasts, baseline):
    """Solve two matched augmented-source objectives from streamed normal equations."""
    n = len(bodies)
    if not n or any(len(v) != n for v in (lasts, gradients, scores, author_ids)):
        raise ValueError("Aligned source views required")
    reference = context_reference(original_lasts, baseline)
    model = dict(baseline) | reference | {"arm": np.array(ARMS[1])}
    weights = view_weights(author_ids)
    if set(author_ids) != set(baseline["members"]):
        raise ValueError("Source membership differs")
    u, sd = model["basis"], model["scale"]
    sl, sk = float(model["score_scale"]), float(model["density_scale"])
    alpha, additive_width = float(model["alpha"]), len(baseline["coef"])
    width = additive_width + len(model["landmarks"]) * (u.shape[1] + 1)
    fg, fr = np.zeros((width, width)), np.zeros(width)
    vg, vr = np.zeros_like(fg), np.zeros(width)
    fc, vc = 0.0, 0.0
    for h, last, g, y, w in zip(bodies, lasts, gradients, scores, weights, strict=True):
        if np.asarray(g).shape != h.shape:
            raise ValueError("Aligned native BODY gradient required")
        z, p, phi = design(h, last, model)
        k = len(h) * (g @ u) * sd
        vg += w * np.outer(phi, phi) / sl**2
        vr += w * phi * y / sl**2
        vc += w * y**2 / sl**2
        fc += w * np.mean(k**2) / sk**2
        normalizer = np.sqrt(w / k.size) / sk
        for rows, d in density_blocks(z, p, last, model):
            a, b = d.reshape(-1, width) * normalizer, k[rows].ravel() * normalizer
            fg += a.T @ a
            fr += a.T @ b
    models, diagnostics = [], []
    for arm, size in zip(ARMS, (additive_width, width), strict=True):
        gram, rhs = (fg + vg)[:size, :size], (fr + vr)[:size]
        penalty = np.eye(size) * alpha
        penalty[0, 0] = 0
        coef = np.linalg.solve(gram + penalty, rhs)
        error = float(np.max(abs((gram + penalty) @ coef - rhs)))
        if error > 1e-8 or not np.isfinite(coef).all():
            raise ValueError("Normal equation solution failed")
        losses = np.array(
            [
                vc - 2 * coef @ vr[:size] + coef @ vg[:size, :size] @ coef,
                fc - 2 * coef @ fr[:size] + coef @ fg[:size, :size] @ coef,
                alpha * np.sum(coef[1:] ** 2),
            ]
        )
        models.append(
            model | {"arm": np.array(arm), "coef": coef, "gram": gram, "rhs": rhs}
        )
        diagnostics.append(
            {
                "candidate": arm,
                "parameters": size,
                "source_authors": len(set(author_ids)),
                "source_views": n,
                "source_score_loss": float(losses[0]),
                "source_density_loss": float(losses[1]),
                "source_penalty": float(losses[2]),
                "source_objective": float(losses.sum()),
                "normal_equation_error": error,
                "context_rank": reference["context_basis"].shape[1],
                "source_context_bandwidth": float(reference["context_bandwidth"]),
                "score_scale": sl,
                "density_scale": sk,
            }
        )
    if diagnostics[1]["source_objective"] > diagnostics[0]["source_objective"] + 1e-8:
        raise ValueError("Nested conditional objective increased")
    return models, diagnostics
