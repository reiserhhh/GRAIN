#!/usr/bin/env python3
"""Outcome-blind conditional attention moments and fixed grouped readout."""

from __future__ import annotations

import hashlib

import numpy as np
import torch

from grain_research.vector_mechanisms.prefix import rotate_half, rotate_query
from grain_research.vector_mechanisms.query_tilt import tilted_readout


def project_boundary(model, boundary, layer=17):
    """Compute next-layer QKV only, retaining exact normalization and rotary rules."""
    parameter = next(model.parameters())
    state = torch.as_tensor(boundary, device=parameter.device, dtype=parameter.dtype)
    block = model.model.layers[layer]
    if state.ndim != 2 or not len(state) or not torch.isfinite(state).all():
        raise ValueError("Finite token-by-channel boundary required")
    if block.self_attn.sliding_window is not None:
        raise ValueError("Unwindowed compact-prefix attention required")
    with torch.no_grad():
        normalized = block.input_layernorm(state[None])
        attention = block.self_attn
        q = attention.q_proj(normalized[:, -1:])[0, 0].float().cpu().numpy()
        k = attention.k_proj(normalized)[0].float().cpu().numpy()
        v = attention.v_proj(normalized)[0].float().cpu().numpy()
        positions = torch.arange(len(state), device=state.device)[None]
        cos, sin = model.model.rotary_emb(state[None], positions)
        cos, sin = (a[0].float().cpu().numpy() for a in (cos, sin))
    dim = attention.head_dim
    k, v = k.reshape(len(state), -1, dim), v.reshape(len(state), -1, dim)
    return {
        "q": rotate_query(q.reshape(-1, dim), cos[-1], sin[-1], attention.scaling),
        "k": k * cos[:, None] + rotate_half(k) * sin[:, None],
        "v": v,
    }


def attention_moments(cache, evidence_mask):
    """Return all-head means, diagonal variances, E mass and entropy; drop no tokens."""
    q, k, v = (np.asarray(cache[key], float) for key in ("q", "k", "v"))
    mask = np.asarray(evidence_mask)
    if (
        q.ndim != 2
        or k.ndim != 3
        or v.shape != k.shape
        or q.shape[1] != k.shape[2]
        or k.shape[1] < 1
        or len(q) % k.shape[1]
        or mask.dtype != bool
        or mask.shape != (len(k),)
        or not mask.any()
    ):
        raise ValueError("Compatible grouped-query arrays and E mask required")
    ratio = len(q) // k.shape[1]
    means, variances, masses, entropies = [], [], [], []
    for h, query in enumerate(q):
        values = v[:, h // ratio]
        mean, p = tilted_readout(k[:, h // ratio], values, query)
        means.append(mean)
        variances.append(p @ ((values - mean) ** 2))
        masses.append(p[mask].sum())
        entropies.append(-np.sum(p * np.log(np.maximum(p, 1e-300))))
    mean = np.concatenate(means)
    return {
        "anchored_attention": mean,
        "anchored_moments": np.concatenate(
            [mean, np.concatenate(variances), masses, entropies]
        ),
        "mass": np.asarray(masses),
        "entropy": np.asarray(entropies),
    }


def author_folds(authors, n_splits=4, salt="GRAIN_CONDITIONAL_BASELINE_V1"):
    """Assign balanced deterministic author folds independently of row order or scores."""
    if len(set(authors)) != len(authors) or not 2 <= n_splits <= len(authors):
        raise ValueError("Unique authors and valid fold count required")
    ordered = sorted(
        authors, key=lambda a: hashlib.sha256(f"{salt}:{a}".encode()).hexdigest()
    )
    lookup = {a: i % n_splits for i, a in enumerate(ordered)}
    return np.array([lookup[a] for a in authors])


def fixed_ridge(train_x, train_y, test_x, alpha=1.0):
    """Fit train-only standardization and trace-normalized linear kernel Ridge."""
    x, y, z = (np.asarray(a, float) for a in (train_x, train_y, test_x))
    if (
        x.ndim != 2
        or z.ndim != 2
        or x.shape[1] != z.shape[1]
        or y.shape != (len(x),)
        or min(x.shape) < 1
        or alpha <= 0
        or not all(np.isfinite(a).all() for a in (x, y, z))
    ):
        raise ValueError(
            "Finite compatible features, targets and positive penalty required"
        )
    center, scale = x.mean(0), x.std(0)
    keep = scale > 1e-8
    if not keep.any():
        return np.full(len(z), y.mean())
    x = (x[:, keep] - center[keep]) / scale[keep]
    z = (z[:, keep] - center[keep]) / scale[keep]
    # Mean training kernel diagonal is one, so the frozen alpha has a fixed scale.
    gram = x @ x.T / x.shape[1]
    coef = np.linalg.solve(gram + alpha * np.eye(len(x)), y - y.mean())
    return y.mean() + (z @ x.T / x.shape[1]) @ coef


def grouped_predictions(features, targets, folds, alpha=1.0):
    """Predict each author outside training; callers aggregate that author's views first."""
    x, y, f = map(np.asarray, (features, targets, folds))
    if y.shape != (len(x),) or f.shape != y.shape or len(np.unique(f)) < 2:
        raise ValueError("One target and fold per author required")
    predictions, constants = np.empty(len(x)), np.empty(len(x))
    for fold in np.unique(f):
        test, train = f == fold, f != fold
        predictions[test] = fixed_ridge(x[train], y[train], x[test], alpha)
        constants[test] = y[train].mean()
    return predictions, constants
