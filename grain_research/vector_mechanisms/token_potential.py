#!/usr/bin/env python3
"""Source-trained scalar token potentials with compatible analytic derivatives."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist

from .position_reader import hidden_basis, position_design

ARMS = ("mean_then_kernel", "kernel_then_mean")


def equal_author_moments(values):
    """Give each author total weight one, irrespective of BODY token count."""
    xs = [np.asarray(x, float) for x in values]
    if not xs or any(x.ndim != 2 or not len(x) for x in xs):
        raise ValueError("Nonempty author matrices required")
    if len({x.shape[1] for x in xs}) != 1 or not all(np.isfinite(x).all() for x in xs):
        raise ValueError("Aligned finite coordinates required")
    mean = np.mean([x.mean(axis=0) for x in xs], axis=0)
    var = np.mean([np.mean((x - mean) ** 2, axis=0) for x in xs], axis=0)
    sd = np.sqrt(var)
    sd[sd < 1e-8] = 1
    return mean, sd


def source_descriptor(states, gradients, members, rank=8, landmarks=16):
    """Fit the fixed source gradient basis, normalization and token landmarks."""
    hs, gs = (
        [np.asarray(x, float) for x in states],
        [np.asarray(x, float) for x in gradients],
    )
    ids = np.asarray(members, int)
    if len(hs) != len(gs) or len(hs) != len(ids) or len(set(ids)) != len(ids):
        raise ValueError("Unique source membership required")
    if any(h.shape != g.shape or not np.isfinite(h).all() for h, g in zip(hs, gs)):
        raise ValueError("Bound source state/gradient matrices required")
    if any(len(h) < landmarks for h in hs):
        raise ValueError("Not enough BODY tokens for fixed distinct landmarks")
    result = hidden_basis(gs, rank=rank, seed=60)
    u = result["basis"].T.copy()
    signs = np.sign(u[np.argmax(np.abs(u), axis=0), np.arange(rank)])
    u *= signs
    projected = [h @ u for h in hs]
    mean, sd = equal_author_moments(projected)
    centers = []
    for h in projected:
        z = (h - mean) / sd
        v = np.column_stack((z, position_design(len(h), 3)[:, 1:]))
        idx = np.linspace(0, len(h) - 1, landmarks).astype(int)
        if len(set(idx)) != landmarks:
            raise ValueError("Repeated source landmarks")
        centers.append(v[idx])
    centers = np.concatenate(centers)
    distances = pdist(centers)
    positive = distances[distances > 0]
    if not len(positive):
        raise ValueError("Undefined positive source bandwidth")
    bandwidth = float(np.median(positive))
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("Invalid source bandwidth")
    return {
        "basis": u,
        "center": mean,
        "scale": sd,
        "landmarks": centers,
        "bandwidth": np.array(bandwidth),
        "members": ids,
        "singular_values": result["singular_values"],
        "weighted_energy": result["weighted_energy"],
    }


def validate_members(model, expected):
    """Reject different source references at fit/application, before reading targets."""
    if not np.array_equal(model["members"], np.asarray(expected, int)):
        raise ValueError("Source exclusion contract differs")


def coordinate_design(z, last, p, landmarks, bandwidth, arm):
    """Return scalar design and T*dL/dz design without token-hidden-center tensors."""
    z, last, p, landmarks = (np.asarray(x, float) for x in (z, last, p, landmarks))
    t, rank = z.shape
    if (
        arm not in ARMS
        or t < 2
        or last.shape != (rank,)
        or p.shape != (t, 3)
        or landmarks.ndim != 2
        or landmarks.shape[1] != rank + 3
        or not len(landmarks)
        or not np.isfinite(bandwidth)
        or bandwidth <= 0
        or not all(np.isfinite(v).all() for v in (z, last, p, landmarks))
    ):
        raise ValueError("Finite fixed scalar descriptor contract required")
    v = np.column_stack((z, p))
    if arm == ARMS[0]:
        v = v.mean(axis=0, keepdims=True)
    diff = v[:, None, :] - landmarks[None, :, :]
    kernel = np.exp(-np.sum(diff * diff, axis=2) / (2 * bandwidth**2))
    derivative = -kernel[:, :, None] * diff[:, :, :rank] / bandwidth**2
    phi = np.concatenate(([1.0], z.mean(axis=0), last, kernel.mean(axis=0)))
    density = np.zeros((t, rank, len(phi)))
    density[:, :, 1 : rank + 1] = np.eye(rank)
    density[:, :, 1 + 2 * rank :] = derivative.transpose(0, 2, 1)
    return phi, density


def state_design(body, last, descriptor, arm):
    """Project pure H16 values using only a frozen source descriptor."""
    u, mu, sd = (descriptor[n] for n in ("basis", "center", "scale"))
    body, last = np.asarray(body, float), np.asarray(last, float)
    if (
        body.ndim != 2
        or last.shape != (body.shape[1],)
        or u.shape[0] != body.shape[1]
        or np.any(sd <= 0)
    ):
        raise ValueError("Hidden state/reference alignment differs")
    return coordinate_design(
        (body @ u - mu) / sd,
        (last @ u - mu) / sd,
        position_design(len(body), 3)[:, 1:],
        descriptor["landmarks"],
        float(descriptor["bandwidth"]),
        arm,
    )


def fit_potential(bodies, lasts, gradients, scores, descriptor, arm, alpha=0.01):
    """Solve the frozen equal-author value plus density-field quadratic objective."""
    y = np.asarray(scores, float)
    n = len(bodies)
    if not n or any(len(v) != n for v in (lasts, gradients, y)):
        raise ValueError("Aligned source authors required")
    u, sd = descriptor["basis"], descriptor["scale"]
    rank = u.shape[1]
    ks = [len(h) * (np.asarray(g, float) @ u) * sd for h, g in zip(bodies, gradients)]
    sl = max(float(y.std()), 1e-6)
    sk = max(float(np.sqrt(np.mean([np.mean(k * k) for k in ks]))), 1e-6)
    records = [state_design(h, last, descriptor, arm) for h, last in zip(bodies, lasts)]
    width = len(records[0][0])
    gram, rhs = np.zeros((width, width)), np.zeros(width)
    for (phi, density), target, k in zip(records, y, ks, strict=True):
        d = density.reshape(-1, width)
        gram += np.outer(phi, phi) / (n * sl**2) + d.T @ d / (n * k.size * sk**2)
        rhs += phi * target / (n * sl**2) + d.T @ k.ravel() / (n * k.size * sk**2)
    penalty = np.eye(width) * alpha
    penalty[0, 0] = 0
    system = gram + penalty
    coef = np.linalg.solve(system, rhs)
    error = float(np.max(np.abs(system @ coef - rhs)))
    if not np.isfinite(coef).all() or error > 1e-8:
        raise ValueError("Unstable normal-equation solution")
    pred = np.array([phi @ coef for phi, _ in records])
    density_mse = float(
        np.mean([np.mean((d @ coef - k) ** 2) for (_, d), k in zip(records, ks)])
    )
    model = dict(descriptor) | {
        "arm": np.array(arm),
        "coef": coef,
        "origin": np.array(y.mean()),
        "score_scale": np.array(sl),
        "density_scale": np.array(sk),
        "alpha": np.array(alpha),
        "gram": gram,
        "rhs": rhs,
    }
    diag = {
        "parameters": width,
        "data_rank": int(np.linalg.matrix_rank(gram)),
        "regularized_condition": float(np.linalg.cond(system)),
        "normal_equation_error": error,
        "source_score_mse": float(np.mean((pred - y) ** 2)),
        "source_density_mse": density_mse,
        "normalized_score_mse": float(np.mean((pred - y) ** 2) / sl**2),
        "normalized_density_mse": density_mse / sk**2,
        "bandwidth": float(descriptor["bandwidth"]),
        "source_authors": n,
        "rank": rank,
    }
    return model, diag


def predict_potential(body, last, model):
    """Compute one scalar and its coordinate density, with no teacher arguments."""
    phi, density = state_design(body, last, model, str(model["arm"]))
    return float(phi @ model["coef"]), density @ model["coef"]


def hidden_gradient(density, model):
    """Chain rule: dL/dH = (Khat/s_z) U.T / T for BODY-only perturbations."""
    return (
        (np.asarray(density, float) / model["scale"]) @ model["basis"].T / len(density)
    )


def last_gradient(model):
    """Return the separate last-token derivative of the same scalar."""
    rank = model["basis"].shape[1]
    beta = model["coef"][1 + rank : 1 + 2 * rank]
    return (beta / model["scale"]) @ model["basis"].T
