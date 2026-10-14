#!/usr/bin/env python3
"""Source-frozen Gaussian response fields over author-specific condition cells."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform

from grain_research.conditional_reference import cell_means, fit_reference


def fit_response_reference(conditions, scores, grain, authors, source, seed=913):
    """Fit all condition moments, geometry and working noise on source authors."""
    c, x, g = (
        np.asarray(conditions, str),
        np.asarray(scores, float),
        np.asarray(grain, float),
    )
    a, source = np.asarray(authors), np.asarray(source)
    if g.ndim != 2 or len(g) != len(c) or len(a) != len(c):
        raise ValueError("Aligned author, condition, evidence and GRAIN units required")
    ref = fit_reference(c, x, a, source)
    descriptors, sse, dof = {}, np.zeros(x.shape[1]), 0
    for uid in source:
        ix = np.flatnonzero(a == uid)
        if not np.isfinite(g[ix]).all():
            raise ValueError("Nonfinite source geometry")
        for key, (_, mean) in cell_means(c[ix], g[ix]).items():
            descriptors.setdefault(key, []).append(mean)
        for key, (n, mean) in cell_means(c[ix], x[ix]).items():
            delta = x[ix[c[ix] == key]] - mean
            sse += np.sum(delta**2, axis=0)
            dof += n - 1
    if dof == 0:
        raise ValueError("No repeated source cell for the working noise estimate")
    raw = np.vstack([np.mean(descriptors[k], axis=0) for k in ref["keys"]])
    center, scale = raw.mean(0), raw.std(0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    z = (raw - center) / scale
    sample = np.sort(
        np.random.default_rng(seed).choice(len(z), min(2048, len(z)), replace=False)
    )
    distances = pdist(z[sample])
    positive = distances[distances > 1e-12]
    bandwidth = float(np.median(positive)) if len(positive) else 1.0
    ref.update(
        descriptor=z,
        descriptor_mean=center,
        descriptor_scale=scale,
        descriptor_raw=raw,
        bandwidth=bandwidth,
        noise=np.maximum(sse / dof, 1e-12),
        noise_dof=dof,
        seed=seed,
    )
    return ref


def condition_kernel(reference, alpha, beta, permutation=None):
    """Return a PSD random-intercept plus RBF condition covariance (unit scale)."""
    if not np.isfinite([alpha, beta]).all() or min(alpha, beta) < 0:
        raise ValueError("Nonnegative finite variance ratios required")
    z = np.asarray(reference["descriptor"], float)
    n = len(z)
    if permutation is not None:
        p = np.asarray(permutation, int)
        if p.shape != (n,) or set(p.tolist()) != set(range(n)):
            raise ValueError("Descriptor permutation must be a bijection")
        z = z[p]
    if beta == 0:
        return np.full((n, n), alpha, float)
    distances = squareform(pdist(z, metric="sqeuclidean"))
    return alpha + beta * np.exp(-distances / (2 * reference["bandwidth"] ** 2))


def observed_cells(conditions, scores, reference):
    """Expose source-known cells without discarding the unknown-cell accounting."""
    cells = cell_means(conditions, scores)
    lookup = {key: i for i, key in enumerate(reference["keys"])}
    known = [key for key in cells if key in lookup]
    return {
        "indices": np.array([lookup[k] for k in known], int),
        "means": np.vstack([cells[k][1] for k in known])
        if known
        else np.empty((0, np.shape(scores)[1])),
        "counts": np.array([cells[k][0] for k in known], int),
        "unknown_cells": len(cells) - len(known),
        "unknown_units": sum(n for k, (n, _) in cells.items() if k not in lookup),
        "all_cells": len(cells),
    }


def predict_cells(reference, kernel, indices, means, counts, targets):
    """Predict disjoint missing cells; return prior, personal adjustment and variance."""
    indices, targets = np.asarray(indices, int), np.asarray(targets, int)
    means, counts = np.asarray(means, float), np.asarray(counts, float)
    mu, k = np.asarray(reference["mean"]), np.asarray(kernel, float)
    if (
        indices.ndim != 1
        or targets.ndim != 1
        or len(set(indices)) != len(indices)
        or len(set(targets)) != len(targets)
        or set(indices) & set(targets)
        or means.shape != (len(indices), mu.shape[1])
        or counts.shape != (len(indices),)
        or np.any(counts <= 0)
        or not np.isfinite(means).all()
        or not np.isfinite(counts).all()
        or k.shape != (len(mu), len(mu))
        or any(j < 0 or j >= len(mu) for j in [*indices, *targets])
    ):
        raise ValueError(
            "Finite disjoint unique observed/target cells and positive counts required"
        )
    prior = mu[targets]
    adjustment = np.zeros_like(prior)
    variance = np.diag(k)[targets].copy()
    if len(indices):
        system = k[np.ix_(indices, indices)] + np.diag(1 / counts)
        cross = k[np.ix_(targets, indices)]
        weights = np.linalg.solve(system, cross.T).T
        adjustment = weights @ (means - mu[indices])
        variance -= np.sum(weights * cross, axis=1)
    if np.min(variance, initial=0) < -1e-9:
        raise ValueError("Negative posterior variance")
    return prior, adjustment, np.maximum(variance, 0)[:, None] * reference["noise"]


def complete_functional(reference, kernel, indices, means, counts, nu):
    """Keep observed cells empirical; label both prior and personal borrowed terms."""
    indices, means, counts = (
        np.asarray(indices, int),
        np.asarray(means, float),
        np.asarray(counts, float),
    )
    nu, k, mu = (
        np.asarray(nu, float),
        np.asarray(kernel, float),
        np.asarray(reference["mean"]),
    )
    if (
        nu.shape != (len(mu),)
        or not np.isfinite(nu).all()
        or np.any(nu < 0)
        or not np.isclose(nu.sum(), 1)
        or len(set(indices)) != len(indices)
        or any(j < 0 or j >= len(mu) for j in indices)
        or means.shape != (len(indices), mu.shape[1])
        or counts.shape != (len(indices),)
        or np.any(counts <= 0)
        or not np.isfinite(means).all()
        or not np.isfinite(counts).all()
        or k.shape != (len(mu), len(mu))
    ):
        raise ValueError("Valid fixed target and observed cell packet required")
    missing = nu.copy()
    missing[indices] = 0
    observed = nu[indices] @ means
    source_prior = missing @ mu
    personal = np.zeros(mu.shape[1])
    k_missing = k @ missing
    variance = float(missing @ k_missing)
    if len(indices):
        system = k[np.ix_(indices, indices)] + np.diag(1 / counts)
        cross = k_missing[indices]
        weights = np.linalg.solve(system, cross)
        personal = weights @ (means - mu[indices])
        variance -= float(weights @ cross)
    if variance < -1e-9:
        raise ValueError("Negative functional variance")
    return {
        "observed": observed,
        "missing_source_prior": source_prior,
        "missing_personal_adjustment": personal,
        "model_completed": observed + source_prior + personal,
        "model_latent_sd": np.sqrt(max(variance, 0) * reference["noise"]),
        "coverage": float(nu[indices].sum()),
        "missing_mass": float(missing.sum()),
        "empirical_point": observed.copy() if missing.sum() <= 1e-14 else None,
    }


def evaluate_hidden_cells(conditions, scores, authors, panel, reference, kernel):
    """Leave each entire known author-condition out, keeping equal-author addresses."""
    c, x, a = (
        np.asarray(conditions, str),
        np.asarray(scores, float),
        np.asarray(authors),
    )
    rows, actual, prior, personal, model_sd = [], [], [], [], []
    for uid in panel:
        ix = a == uid
        cells = observed_cells(c[ix], x[ix], reference)
        for j, target in enumerate(cells["indices"]):
            keep = np.arange(len(cells["indices"])) != j
            p, d, v = predict_cells(
                reference,
                kernel,
                cells["indices"][keep],
                cells["means"][keep],
                cells["counts"][keep],
                [target],
            )
            rows.append(
                {
                    "author_index": int(uid),
                    "condition_index": int(target),
                    "target_units": int(cells["counts"][j]),
                    "other_known_cells": int(keep.sum()),
                    "unknown_cells": cells["unknown_cells"],
                    "unknown_units": cells["unknown_units"],
                }
            )
            actual.append(cells["means"][j])
            prior.append(p[0])
            personal.append(d[0])
            model_sd.append(np.sqrt(v[0]))
    if not rows:
        raise ValueError("No reference-known targets in evaluation panel")
    return (
        rows,
        np.asarray(actual),
        np.asarray(prior),
        np.asarray(personal),
        np.asarray(model_sd),
    )
