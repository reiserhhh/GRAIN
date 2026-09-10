#!/usr/bin/env python3
"""Kernel-weighted content versus relative selection, both retaining log mass."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from . import coupled_pooling as objective_reference
from . import full_context_geometry as base
from .nonlinear_pooling import LOSS_NAMES

ARMS = ("unnormalized_mass", "normalized_mass")


def weighted_summary(z, logw):
    """Stable mass and relative content, without clipping an unrepresentable mass."""
    z, logw = np.asarray(z, float), np.asarray(logw, float)
    if z.ndim != 2 or logw.ndim != 2 or len(z) != len(logw) or len(z) < 2:
        raise ValueError("Aligned token vectors and kernel log weights required")
    if not np.isfinite(z).all() or not np.isfinite(logw).all():
        raise ValueError("Finite token vectors and log weights required")
    logsum = logsumexp(logw, axis=0)
    ell = logsum - np.log(len(z))
    a = np.exp(logw - logsum)
    mass = np.exp(ell)
    if not np.isfinite(mass).all() or np.any(mass <= 0):
        raise FloatingPointError("Kernel mass is not representable; no clipping")
    relative = a.T @ z
    moment = mass[:, None] * relative
    if not np.isfinite(moment).all() or not np.isfinite(relative).all():
        raise FloatingPointError("Nonfinite weighted content")
    return {"a": a, "ell": ell, "mass": mass, "r": relative, "m": moment}


def kernel_state(z, p, model):
    """Use the original64-center contract, selecting indices0,4,...,60 only."""
    centers = np.asarray(model["landmarks"])
    sigma = float(model["bandwidth"])
    if centers.shape != (64, z.shape[1] + 3) or p.shape != (len(z), 3):
        raise ValueError(
            "Exactly64 frozen landmarks and three position features required"
        )
    if sigma <= 0 or not np.isfinite(sigma):
        raise ValueError("Positive frozen bandwidth required")
    delta = np.column_stack((z, p))[:, None] - centers[::4][None]
    logw = -np.sum(delta**2, axis=2) / (2 * sigma**2)
    return weighted_summary(z, logw) | {"s": -delta[:, :, : z.shape[1]] / sigma**2}


def summary_features(z, p, model):
    """Return16 content vectors plus16 log masses in a declared arm."""
    state = kernel_state(z, p, model)
    arm = str(model["arm"])
    if arm not in ARMS:
        raise ValueError("Unknown evidence reader arm")
    content = state["m"] if arm == ARMS[0] else state["r"]
    return np.r_[content.ravel(), state["ell"]]


def reference(records, local, arm):
    """Freeze each feature center on original source authors only."""
    groups = objective_reference.layout(records, local["members"])
    model = dict(local, arm=np.array(arm), local_width=np.array(len(local["coef"])))
    if local["landmarks"].shape != (64, local["basis"].shape[1] + 3):
        raise ValueError("Exactly64 frozen landmarks required")
    values = []
    for g in groups:
        r = records[g[0]]
        z, p, phi = base.design(r["body"], r["last"], model)
        if len(phi) != len(local["coef"]):
            raise ValueError("Frozen local coefficient prefix required")
        values.append(summary_features(z, p, model))
    model["summary_center"] = np.mean(values, axis=0)
    return model


def design(body, last, model):
    """Local scalar plus centered mass-preserving summary coordinates."""
    z, p, phi = base.design(body, last, model)
    return z, p, np.r_[phi, summary_features(z, p, model) - model["summary_center"]]


def density_blocks(z, p, last, model):
    """Stream T*dF/dz with the direct-content, selection and mass derivatives."""
    state = kernel_state(z, p, model)
    rank, n = z.shape[1], len(z)
    for rows, local in base.density_blocks(z, p, last, model):
        s, a = state["s"][rows], state["a"][rows]
        if str(model["arm"]) == ARMS[0]:
            coefficient = n * a * state["mass"][None]
            contrast = z[rows, None, :]
        elif str(model["arm"]) == ARMS[1]:
            coefficient = n * a
            contrast = z[rows, None, :] - state["r"][None]
        else:
            raise ValueError("Unknown evidence reader arm")
        derivative = coefficient[:, :, None, None] * (
            np.eye(rank)[None, None] + s[:, :, :, None] * contrast[:, :, None, :]
        )
        content = derivative.transpose(0, 2, 1, 3).reshape(
            rows.stop - rows.start, rank, 16 * rank
        )
        logmass = (n * a[:, :, None] * s).transpose(0, 2, 1)
        extra = np.concatenate((content, logmass), axis=2)
        if not np.isfinite(extra).all():
            raise FloatingPointError("Nonfinite derivative; no clipping")
        yield rows, np.concatenate((local, extra), axis=2)


def predict(body, last, model):
    """One scalar supplies both complete score and BODY density."""
    z, p, phi = design(body, last, model)
    if phi.shape != model["coef"].shape:
        raise ValueError("Scalar coefficient layout differs")
    return float(phi @ model["coef"]), np.concatenate(
        [d @ model["coef"] for _, d in density_blocks(z, p, last, model)]
    )


def scalar(body, last, model):
    """Score-only computation for independent finite differences."""
    return float(design(body, last, model)[2] @ model["coef"])


def last_gradient(body, last, model):
    """Summary features do not depend on the separate last state."""
    return base.last_gradient(
        body, last, dict(model, coef=model["coef"][: int(model["local_width"])])
    )


def normal_equations(records, model, scales):
    """Keep the four source objectives and their normalization exactly matched."""
    groups = objective_reference.layout(records, model["members"])
    ts = objective_reference.targets(records, model)
    coords = [design(r["body"], r["last"], model) for r in records]
    width = len(coords[0][2])
    gs, rs, cs = np.zeros((4, width, width)), np.zeros((4, width)), np.zeros(4)

    def add(channel, a, b, norm):
        a, b = a.reshape(-1, width) * norm, b.ravel() * norm
        gs[channel] += a.T @ a
        rs[channel] += a.T @ b
        cs[channel] += b @ b

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
                    raise ValueError("Paired token blocks differ")
                start, end = max(s0.start, cut), s0.stop
                if start >= end:
                    continue
                d = (b[start - s0.start :] - a[start - s0.start :]) / 2
                if channel == 3 and np.any(d[:, :, : int(model["local_width"])]):
                    raise ValueError(
                        "Local prefix must have zero cross-region derivative"
                    )
                add(
                    channel,
                    d,
                    delta[start - cut : end - cut],
                    1 / np.sqrt(len(groups) * delta.size * energy),
                )
    return gs, rs, cs


def objective(records, model, scales):
    """Direct prediction loss, independent of Gram arithmetic."""
    ts = objective_reference.targets(records, model)
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
    for i, j, plus, minus in objective_reference.layout(records, model["members"]):
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
    """Two equally sized source-only solves, preserving the frozen local normal block."""
    scales = objective_reference.energies(records, local)
    for value, key in zip(
        scales, ("source_delta_energy", "source_cross_energy"), strict=True
    ):
        if not np.isclose(value, float(local[key]), rtol=1e-10, atol=1e-14):
            raise ValueError("Frozen energy normalization changed")
    oldloss = objective_reference.objective(records, local, scales).sum()
    models, diags = [], []
    for arm in ARMS:
        model = reference(records, local, arm)
        gs, rs, cs = normal_equations(records, model, scales)
        gram, rhs = gs.sum(0), rs.sum(0)
        size = int(model["local_width"])
        error = float(
            max(
                np.max(abs(gram[:size, :size] - local["gram"])),
                np.max(abs(rhs[:size] - local["rhs"])),
            )
        )
        if error > 1e-8:
            raise ValueError("Frozen matched-local normal equations changed")
        penalty = np.eye(len(rhs)) * float(model["alpha"])
        penalty[0, 0] = 0
        coef = np.linalg.solve(gram + penalty, rhs)
        model |= {"coef": coef, "gram": gram, "rhs": rhs}
        loss = objective(records, model, scales)
        algebra = np.r_[
            cs - 2 * (rs @ coef) + np.einsum("i,kij,j->k", coef, gs, coef),
            coef @ penalty @ coef,
        ]
        objerror = float(np.max(abs(loss - algebra)))
        normal = float(np.max(abs((gram + penalty) @ coef - rhs)))
        if (
            objerror > 1e-7
            or normal > 1e-8
            or not np.isfinite(coef).all()
            or loss.sum() > oldloss + 1e-8
        ):
            raise ValueError("Compatible/nested objective failed")
        models.append(model)
        diags.append(
            {
                "candidate": arm,
                "parameters": len(coef),
                "source_authors": len(local["members"]),
                "source_views": len(records),
                "source_last_energy": scales[0],
                "source_cross_energy": scales[1],
                "matched_prefix_error": error,
                "objective_reconstruction_error": objerror,
                "normal_equation_error": normal,
                "source_objective": float(loss.sum()),
                **dict(zip(LOSS_NAMES, map(float, loss), strict=True)),
            }
        )
    return models, diags
