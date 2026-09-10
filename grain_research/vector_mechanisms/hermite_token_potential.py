#!/usr/bin/env python3
"""Derivative-enhanced scalar Gaussian token dictionary with compatible fields."""

from __future__ import annotations

import numpy as np

from .position_reader import position_design


def coordinates(body, last, model):
    """Use only the unchanged source coordinate reference and current H16."""
    h, last = np.asarray(body, float), np.asarray(last, float)
    u, mu, sd = (np.asarray(model[n], float) for n in ("basis", "center", "scale"))
    if (
        h.ndim != 2
        or len(h) < 2
        or last.shape != (h.shape[1],)
        or np.any(sd <= 0)
        or not np.isfinite(h).all()
        or not np.isfinite(last).all()
    ):
        raise ValueError("Finite aligned BODY and separate last token required")
    return (h @ u - mu) / sd, (last @ u - mu) / sd


def local_features(z, p, model):
    """Return Gaussian and source-center derivative scalar features for a token block."""
    z, p = np.asarray(z, float), np.asarray(p, float)
    sigma = float(model["bandwidth"])
    centers = np.asarray(model["landmarks"], float)
    if sigma <= 0 or p.shape != (len(z), 3) or centers.shape[1] != z.shape[1] + 3:
        raise ValueError("Frozen descriptor/positive bandwidth required")
    delta = np.column_stack((z, p))[:, None, :] - centers[None]
    k = np.exp(-np.sum(delta * delta, axis=2) / (2 * sigma**2))
    dz = delta[:, :, : z.shape[1]]
    psi = k[:, :, None] * dz / sigma
    return k, psi, dz


def scalar_design(z, last, p, model):
    """Mean local scalar features, with the old coefficient prefix unchanged."""
    k, psi, _ = local_features(z, p, model)
    return np.r_[1, z.mean(0), last, k.mean(0), psi.mean(0).ravel()]


def density_blocks(z, p, model, chunk_size=128):
    """Yield T*dL/dz designs in small token blocks, never token-hidden-center tensors."""
    rank = z.shape[1]
    m = len(model["landmarks"])
    oldwidth = 1 + 2 * rank + m
    width = oldwidth + m * rank
    sigma = float(model["bandwidth"])
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("Positive token block size required")
    for start in range(0, len(z), chunk_size):
        stop = min(start + chunk_size, len(z))
        k, _, diff = local_features(z[start:stop], p[start:stop], model)
        d = np.zeros((stop - start, rank, width))
        d[:, :, 1 : rank + 1] = np.eye(rank)
        d[:, :, 1 + 2 * rank : oldwidth] = (-k[:, :, None] * diff / sigma**2).transpose(
            0, 2, 1
        )
        hermite = (
            (
                np.eye(rank)[None, None]
                - diff[:, :, :, None] * diff[:, :, None, :] / sigma**2
            )
            * k[:, :, None, None]
            / sigma
        )
        d[:, :, oldwidth:] = hermite.transpose(0, 2, 1, 3).reshape(
            stop - start, rank, m * rank
        )
        yield slice(start, stop), d


def predict(body, last, model):
    """Score and BODY density from one scalar, without native teacher arguments."""
    z, l = coordinates(body, last, model)
    p = position_design(len(z), 3)[:, 1:]
    phi = scalar_design(z, l, p, model)
    coef = np.asarray(model["coef"], float)
    if coef.shape != phi.shape or not np.isfinite(coef).all():
        raise ValueError("Expanded scalar coefficient layout differs")
    density = np.concatenate([d @ coef for _, d in density_blocks(z, p, model)])
    return float(phi @ coef), density


def qr_capacity(density, target):
    """Compute fixed-cutoff source column-space capacity via a small augmented QR."""
    d, k = np.asarray(density, float), np.asarray(target, float)
    if (
        d.ndim != 2
        or k.shape != (len(d),)
        or not np.isfinite(d).all()
        or not np.isfinite(k).all()
    ):
        raise ValueError("Aligned finite source density system required")
    r = np.linalg.qr(np.column_stack((d, k)), mode="r")
    u, s, vt = np.linalg.svd(r[:, :-1], full_matrices=False)
    keep = s > s[0] * 1e-10 if len(s) and s[0] else np.zeros(len(s), bool)
    response = u[:, keep].T @ r[:, -1]
    coef = vt[keep].T @ (response / s[keep])
    residual = r[:, -1] - u[:, keep] @ response
    return {
        "source_field_capacity_loss": float(residual @ residual),
        "source_represented_energy": float(response @ response),
        "source_field_rank": int(keep.sum()),
        "source_field_inverse_norm": float(np.linalg.norm(coef)),
        "source_field_condition": float(s[0] / s[keep][-1])
        if keep.any()
        else float("nan"),
        "capacity_energy_error": abs(
            float(k @ k - response @ response - residual @ residual)
        ),
    }, s


def fit(bodies, lasts, gradients, scores, baseline):
    """Fit one nested scalar/density program with the unchanged NEXT60 objective."""
    n = len(bodies)
    if not n or any(len(a) != n for a in (lasts, gradients, scores)):
        raise ValueError("Aligned source membership required")
    u, sd = baseline["basis"], baseline["scale"]
    rank = u.shape[1]
    oldwidth = len(baseline["coef"])
    width = oldwidth + len(baseline["landmarks"]) * rank
    y = np.asarray(scores, float)
    sl, sk = float(baseline["score_scale"]), float(baseline["density_scale"])
    alpha = float(baseline["alpha"])
    ks = [len(h) * (np.asarray(g, float) @ u) * sd for h, g in zip(bodies, gradients)]
    np.testing.assert_allclose(max(float(y.std()), 1e-6), sl, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(
        max(float(np.sqrt(np.mean([np.mean(k * k) for k in ks]))), 1e-6),
        sk,
        atol=1e-12,
        rtol=1e-12,
    )
    value, dblocks, kblocks = [], [], []
    fieldgram, fieldrhs = np.zeros((width, width)), np.zeros(width)
    for h, last, k in zip(bodies, lasts, ks, strict=True):
        z, l = coordinates(h, last, baseline)
        p = position_design(len(h), 3)[:, 1:]
        value.append(scalar_design(z, l, p, baseline) / (np.sqrt(n) * sl))
        normalizer = np.sqrt(n * len(h) * rank) * sk
        for rows, d in density_blocks(z, p, baseline):
            a = d.reshape(-1, width) / normalizer
            b = k[rows].ravel() / normalizer
            fieldgram += a.T @ a
            fieldrhs += a.T @ b
            dblocks.append(a)
            kblocks.append(b)
    v = np.asarray(value)
    ell = y / (np.sqrt(n) * sl)
    gram = fieldgram + v.T @ v
    rhs = fieldrhs + v.T @ ell
    penalty = np.eye(width) * alpha
    penalty[0, 0] = 0
    coef = np.linalg.solve(gram + penalty, rhs)
    oldcoef = np.r_[baseline["coef"], np.zeros(width - oldwidth)]

    def losses(theta):
        return np.array(
            [
                np.sum((v @ theta - ell) ** 2),
                1 - 2 * theta @ fieldrhs + theta @ fieldgram @ theta,
                alpha * np.sum(theta[1:] ** 2),
            ]
        )

    newloss, oldloss = losses(coef), losses(oldcoef)
    error = float(np.max(abs((gram + penalty) @ coef - rhs)))
    if error > 1e-8 or newloss.sum() > oldloss.sum() + 1e-8:
        raise ValueError("Nested objective or normal equation contract failed")
    cap, singular = qr_capacity(np.concatenate(dblocks), np.concatenate(kblocks))
    diag = {
        "parameters": width,
        "baseline_parameters": oldwidth,
        "source_authors": n,
        "source_score_loss": float(newloss[0]),
        "source_field_loss": float(newloss[1]),
        "source_penalty": float(newloss[2]),
        "source_objective": float(newloss.sum()),
        "baseline_source_score_loss": float(oldloss[0]),
        "baseline_source_field_loss": float(oldloss[1]),
        "baseline_source_penalty": float(oldloss[2]),
        "baseline_source_objective": float(oldloss.sum()),
        "normal_equation_error": error,
        "coefficient_norm": float(np.linalg.norm(coef)),
        "baseline_coefficient_norm": float(np.linalg.norm(baseline["coef"])),
        **cap,
    }
    model = dict(baseline) | {
        "arm": np.array("hermite_kernel_then_mean"),
        "coef": coef,
        "gram": gram,
        "rhs": rhs,
        "field_gram": fieldgram,
        "field_rhs": fieldrhs,
        "source_value_design": v,
        "source_value_targets": ell,
        "source_field_singular_values": singular,
        "baseline_width": np.array(oldwidth),
    }
    return model, diag
