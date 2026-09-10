#!/usr/bin/env python3
"""A fixed 16-Gaussian token map before the compatible pooled quadratic."""

from __future__ import annotations

import numpy as np

from . import coupled_pooling as prior
from . import full_context_geometry as base

ARM = "nonlinear_pooled_quadratic"
LOSS_NAMES = (
    "source_score_loss",
    "source_density_loss",
    "source_last_loss",
    "source_cross_loss",
    "source_penalty",
)


def token_features(z, p, model):
    """Return the 16 declared Gaussians and their derivatives in frozen z coordinates."""
    centers = np.asarray(model["landmarks"])
    if centers.shape != (64, z.shape[1] + 3) or p.shape != (len(z), 3):
        raise ValueError(
            "Exactly64 frozen landmarks and three position features required"
        )
    sigma = float(model["bandwidth"])
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Positive frozen bandwidth required")
    delta = np.column_stack((z, p))[:, None] - centers[::4][None]
    phi = np.exp(-np.sum(delta**2, axis=2) / (2 * sigma**2))
    jac = -phi[:, :, None] * delta[:, :, : z.shape[1]] / sigma**2
    return phi, jac


def reference(records, local):
    """Use original source views only, never intervened/current author centering."""
    groups = prior.layout(records, local["members"])
    if local["landmarks"].shape != (64, local["basis"].shape[1] + 3):
        raise ValueError("Exactly64 frozen landmarks required")
    values = []
    for group in groups:
        r = records[group[0]]
        z, p, design = base.design(r["body"], r["last"], local)
        if len(design) != len(local["coef"]):
            raise ValueError("Frozen local prefix required")
        values.append(token_features(z, p, local)[0].mean(0))
    return dict(
        local,
        arm=np.array(ARM),
        pool_center=np.mean(values, axis=0),
        local_width=np.array(len(local["coef"])),
    )


def quadratic(z, p, model):
    """Return quadratic coefficient features and their pooled-coordinate derivatives."""
    features, jac = token_features(z, p, model)
    v = features.mean(0) - model["pool_center"]
    i, j = np.triu_indices(16)
    phi = v[i] * v[j]
    phi[i == j] *= 0.5
    dv = np.zeros((136, 16))
    rows = np.arange(136)
    dv[rows, i] = v[j]
    off = i != j
    dv[rows[off], j[off]] = v[i[off]]
    return phi, dv, jac


def design(body, last, model):
    """Keep the full local prefix; replace only the pooled feature map."""
    z, p, phi = base.design(body, last, model)
    return z, p, np.r_[phi, quadratic(z, p, model)[0]]


def density_blocks(z, p, last, model):
    """Stream T*dF/dz, including content-dependent derivatives of pooled features."""
    _, dv, jac = quadratic(z, p, model)
    for rows, local in base.density_blocks(z, p, last, model):
        extra = np.einsum("tkr,fk->trf", jac[rows], dv)
        yield rows, np.concatenate((local, extra), axis=2)


def predict(body, last, model):
    """Compute score and full BODY density from the same scalar."""
    z, p, phi = design(body, last, model)
    if phi.shape != model["coef"].shape:
        raise ValueError("Scalar coefficient layout differs")
    return float(phi @ model["coef"]), np.concatenate(
        [a @ model["coef"] for _, a in density_blocks(z, p, last, model)]
    )


def scalar(body, last, model):
    """Score-only evaluation for independent finite differences."""
    return float(design(body, last, model)[2] @ model["coef"])


def last_gradient(body, last, model):
    """The added pooled term has no last-state dependence."""
    return base.last_gradient(
        body, last, dict(model, coef=model["coef"][: int(model["local_width"])])
    )


def normal_equations(records, model, scales):
    """Same four source-normalized loss components as NEXT68, new design only."""
    groups = prior.layout(records, model["members"])
    ts = prior.targets(records, model)
    coords = [design(r["body"], r["last"], model) for r in records]
    width = len(coords[0][2])
    grams, rhs, constants = (
        np.zeros((4, width, width)),
        np.zeros((4, width)),
        np.zeros(4),
    )

    def add(channel, a, b, norm):
        a, b = a.reshape(-1, width) * norm, b.ravel() * norm
        grams[channel] += a.T @ a
        rhs[channel] += a.T @ b
        constants[channel] += b @ b

    for r, (z, p, phi), t in zip(records, coords, ts, strict=True):
        add(
            0,
            phi,
            np.asarray(r["score"]),
            1 / (np.sqrt(len(records)) * float(model["score_scale"])),
        )
        for rows, d in density_blocks(z, p, r["last"], model):
            add(
                1,
                d,
                t[rows],
                1 / (np.sqrt(len(records) * t.size) * float(model["density_scale"])),
            )
    for i, j, plus, minus in groups:
        for channel, left, right, cut, energy in (
            (2, i, j, 0, scales[0]),
            (3, minus, plus, len(ts[i]) // 2, scales[1]),
        ):
            delta = (ts[right][cut:] - ts[left][cut:]) / 2
            streams = [
                density_blocks(*coords[k][:2], records[k]["last"], model)
                for k in (left, right)
            ]
            for (s0, a), (s1, b) in zip(*streams, strict=True):
                if s0 != s1:
                    raise ValueError("Paired block layout differs")
                start, end = max(s0.start, cut), s0.stop
                if start >= end:
                    continue
                d = (b[start - s0.start :] - a[start - s0.start :]) / 2
                if channel == 3 and np.any(d[:, :, : int(model["local_width"])]):
                    raise ValueError("Local cross-token derivative must be zero")
                add(
                    channel,
                    d,
                    delta[start - cut : end - cut],
                    1 / np.sqrt(len(groups) * delta.size * energy),
                )
    return grams, rhs, constants


def objective(records, model, scales):
    """Recompute the objective from predictions, independently of normal equations."""
    ts = prior.targets(records, model)
    ps = [predict(r["body"], r["last"], model) for r in records]
    score = (
        np.mean([(p[0] - r["score"]) ** 2 for p, r in zip(ps, records, strict=True)])
        / float(model["score_scale"]) ** 2
    )
    field = (
        np.mean([np.mean((p[1] - t) ** 2) for p, t in zip(ps, ts, strict=True)])
        / float(model["density_scale"]) ** 2
    )
    paired = [[], []]
    for i, j, plus, minus in prior.layout(records, model["members"]):
        for c, left, right, cut in ((0, i, j, 0), (1, minus, plus, len(ts[i]) // 2)):
            e = (ps[right][1] - ps[left][1] - ts[right] + ts[left]) / 2
            paired[c].append(np.mean(e[cut:] ** 2))
    return np.array(
        [
            score,
            field,
            np.mean(paired[0]) / scales[0],
            np.mean(paired[1]) / scales[1],
            float(model["alpha"]) * np.sum(model["coef"][1:] ** 2),
        ]
    )


def fit(records, local):
    """One convex solve; verify same local normal prefix and both target energy units."""
    model = reference(records, local)
    scales = prior.energies(records, model)
    for observed, key in zip(
        scales, ("source_delta_energy", "source_cross_energy"), strict=True
    ):
        if not np.isclose(observed, float(local[key]), rtol=1e-10, atol=1e-14):
            raise ValueError("Frozen source energy normalization changed")
    gs, rs, cs = normal_equations(records, model, scales)
    gram, rhs = gs.sum(0), rs.sum(0)
    size = int(model["local_width"])
    prefix_error = float(
        max(
            np.max(abs(gram[:size, :size] - local["gram"])),
            np.max(abs(rhs[:size] - local["rhs"])),
        )
    )
    if prefix_error > 1e-8:
        raise ValueError("Frozen matched-local normal equations changed")
    penalty = np.eye(len(rhs)) * float(model["alpha"])
    penalty[0, 0] = 0
    coef = np.linalg.solve(gram + penalty, rhs)
    model |= {
        "coef": coef,
        "gram": gram,
        "rhs": rhs,
        "source_cross_energy": np.array(scales[1]),
    }
    loss = objective(records, model, scales)
    algebra = np.r_[
        cs - 2 * (rs @ coef) + np.einsum("i,kij,j->k", coef, gs, coef),
        coef @ penalty @ coef,
    ]
    error = float(np.max(abs(loss - algebra)))
    normal = float(np.max(abs((gram + penalty) @ coef - rhs)))
    local_loss = prior.objective(records, local, scales)
    if (
        error > 1e-7
        or normal > 1e-8
        or loss.sum() > local_loss.sum() + 1e-8
        or not np.isfinite(coef).all()
    ):
        raise ValueError("Compatible objective or nested reference failed")
    return model, {
        "candidate": ARM,
        "parameters": len(coef),
        "source_authors": len(model["members"]),
        "source_views": len(records),
        "source_last_energy": scales[0],
        "source_cross_energy": scales[1],
        "matched_prefix_error": prefix_error,
        "objective_reconstruction_error": error,
        "normal_equation_error": normal,
        "source_objective": float(loss.sum()),
        **dict(zip(LOSS_NAMES, map(float, loss), strict=True)),
    }
