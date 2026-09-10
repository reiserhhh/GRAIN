#!/usr/bin/env python3
"""Source-only conditional pair scoring from aligned encoded text inputs."""

from __future__ import annotations

import numpy as np

from .joint_readout import centered_logits, predict_joint

BLOCKS = ("M", "R", "L")
ARMS = ("fourier_bag", "fourier_position")
PAIRS = ((0, 1), (0, 2))  # Singleton slots correspond to comments 0, 1, 11.


def matrix(x):
    """Validate a finite two-dimensional numerical input."""
    x = np.asarray(x, float)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Finite nonempty matrix required")
    return x


def fit_context(blocks):
    """Freeze block centers/scales on source authors, retaining constant metadata."""
    model, n = {}, None
    for key in BLOCKS:
        x = matrix(blocks[key])
        if n is not None and len(x) != n:
            raise ValueError("Context blocks must share author rows")
        n = len(x)
        center = x.mean(0)
        constant = bool(np.all(np.ptp(x, axis=0) == 0))
        centered = x - center if not constant else np.zeros_like(x)
        scale = float(np.mean(np.sum(centered**2, axis=1)))
        if not constant and scale <= 0:
            raise ValueError("Nonconstant context has invalid energy")
        model[f"{key}_center"] = center
        model[f"{key}_scale"] = np.array(scale)
        model[f"{key}_constant"] = np.array(constant)
        model[f"{key}_source"] = centered / np.sqrt(scale) if scale else centered
    model["n_source"] = np.array(n)
    return model


def context_kernel(blocks, model):
    """Compute evaluation-to-source context similarities without refitting."""
    n = len(matrix(blocks["M"]))
    k = np.zeros((n, int(model["n_source"])))
    active = 0
    for key in BLOCKS:
        x = matrix(blocks[key])
        center = np.asarray(model[f"{key}_center"])
        if len(x) != n or x.shape[1:] != center.shape:
            raise ValueError("Context layout differs from frozen source layout")
        if bool(model[f"{key}_constant"]):
            continue
        v = (x - center) / np.sqrt(float(model[f"{key}_scale"]))
        k += v @ np.asarray(model[f"{key}_source"]).T
        active += 1
    return k / active if active else k


def ridge(k, y, penalty=0.01):
    """Solve mean-loss kernel Ridge with training mean-diagonal scaling."""
    k, y = matrix(k), matrix(y)
    if k.shape != (len(y), len(y)) or penalty <= 0 or not np.isfinite(penalty):
        raise ValueError("Square source kernel and positive penalty required")
    scale = float(np.trace(k) / len(k))
    if scale <= 1e-24:
        return np.zeros_like(y), 0.0
    return np.linalg.solve(k + len(k) * penalty * scale * np.eye(len(k)), y), scale


def singleton_kernel(k_context, phi, source_phi):
    """Product feature kernel; zero contrast has exactly zero increment."""
    k_context, phi, source_phi = map(matrix, (k_context, phi, source_phi))
    if k_context.shape != (len(phi), len(source_phi)):
        raise ValueError("Context/contrast row mismatch")
    return (1 + k_context) * (phi @ source_phi.T)


def fit_complete(blocks, phi, z0, increments, mixed_models, penalty=0.01, values=None):
    """Fit one base and two shared singleton heads on source authors only."""
    z0, increments = centered_logits(matrix(z0)), centered_logits(increments)
    n = len(z0)
    if increments.shape != (n, 3, z0.shape[1]):
        raise ValueError(
            "Three aligned singleton increments per source author required"
        )
    model = fit_context(blocks)
    if int(model["n_source"]) != n:
        raise ValueError("Source endpoint/context count mismatch")
    k = context_kernel(blocks, model)
    model["base_mean"] = z0.mean(0)
    model["base_weights"], scale = ridge(k, z0 - z0.mean(0), penalty)
    model["base_kernel_scale"] = np.array(scale)
    model["penalty"] = np.array(penalty)
    if values is not None:
        values = np.asarray(values, float)
        if values.shape != (z0.shape[1],) or not np.isfinite(values).all():
            raise ValueError("Fixed candidate alphabet must match endpoints")
        model["values"] = values.copy()
    source_index = np.repeat(np.arange(n), 3)
    model["singleton_source_index"] = source_index
    expanded_k = k[np.ix_(source_index, source_index)]
    for arm in ARMS:
        f = np.asarray(phi[arm], float)
        if f.ndim != 3 or f.shape[:2] != (n, 3) or not np.isfinite(f).all():
            raise ValueError("Three contrast vectors per source author required")
        source = f.reshape(n * 3, -1)
        model[f"{arm}_phi_source"] = source.copy()
        sk = singleton_kernel(expanded_k, source, source)
        weights, scale = ridge(sk, increments.reshape(n * 3, -1), penalty)
        model[f"{arm}_singleton_weights"] = weights
        model[f"{arm}_singleton_scale"] = np.array(scale)
        for name in ("left", "right", "weights", "scale"):
            model[f"{arm}_mixed_{name}"] = np.asarray(mixed_models[arm][name]).copy()
    return model


def predict_complete(blocks, phi, model):
    """Produce all complete pair logits from numerical inputs and frozen assets."""
    k = context_kernel(blocks, model)
    n = len(k)
    z0 = centered_logits(model["base_mean"] + k @ model["base_weights"])
    held_index = np.repeat(np.arange(n), 3)
    source_index = model["singleton_source_index"]
    expanded_k = k[np.ix_(held_index, source_index)]
    result = {"base": z0}
    for arm in ARMS:
        f = np.asarray(phi[arm], float)
        if f.ndim != 3 or f.shape[:2] != (n, 3):
            raise ValueError("Three ordered singleton slots required")
        sk = singleton_kernel(
            expanded_k, f.reshape(n * 3, -1), model[f"{arm}_phi_source"]
        )
        inc = centered_logits(
            (sk @ model[f"{arm}_singleton_weights"]).reshape(n, 3, -1)
        )
        additive = np.stack([z0 + inc[:, i] + inc[:, j] for i, j in PAIRS], 1)
        old = {
            key: model[f"{arm}_mixed_{key}"]
            for key in ("left", "right", "weights", "scale")
        }
        mixed = centered_logits(
            np.stack([predict_joint(f[:, i], f[:, j], old) for i, j in PAIRS], 1)
        )
        result.update(
            {
                f"{arm}_singleton": inc,
                f"{arm}_additive": additive,
                f"{arm}_mixed": mixed,
                f"{arm}_full": additive + mixed,
            }
        )
        if "values" in model:
            full = result[f"{arm}_full"]
            p = np.exp(full - full.max(axis=-1, keepdims=True))
            result[f"{arm}_score"] = (p @ model["values"]) / p.sum(-1)
    return result


def error_account(errors):
    """Exact signed per-author squared-error decomposition over named components."""
    arrays = {k: matrix(v) for k, v in errors.items()}
    shapes = {v.shape for v in arrays.values()}
    if len(shapes) != 1:
        raise ValueError("Aligned component errors required")
    keys = list(arrays)
    terms = {f"square_{k}": np.mean(v**2, 1) for k, v in arrays.items()}
    for i, k in enumerate(keys):
        for j in keys[i + 1 :]:
            terms[f"cross_{k}_{j}"] = 2 * np.mean(arrays[k] * arrays[j], 1)
    full = np.mean(sum(arrays.values()) ** 2, 1)
    discrepancy = full - sum(terms.values())
    terms["full_mse"] = full
    terms["identity_error"] = discrepancy
    return terms
