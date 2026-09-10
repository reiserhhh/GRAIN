#!/usr/bin/env python3
"""Source directional secants extend the output range of a conditional reader."""

import numpy as np

from . import context_differential_reader as base
from .context_reading_interaction import replace_last
from .selection_endpoint import endpoint

VARIANTS = (*base.VARIANTS, "source_motion")


def directions(model):
    """Use only this source reference to define centered unit context directions."""
    c = np.asarray(model["c"], float)
    delta = c.mean(axis=0) - c
    norm = np.linalg.norm(delta, axis=1)
    v = np.divide(
        delta, norm[:, None], out=np.zeros_like(delta), where=norm[:, None] > 0
    )
    return v, 0.01 * np.sqrt(float(model["h2"]))


def secant(readers, epsilon):
    """Convert [query,minus/plus,hidden] endpoint observations to a vector slope."""
    r = np.asarray(readers, float)
    if r.ndim != 3 or r.shape[:2] != (2, 2) or not np.isfinite(r).all() or epsilon <= 0:
        raise ValueError("Finite paired readers and positive source step required")
    return ((r[0, 1] - r[1, 1]) - (r[0, 0] - r[1, 0])) / (2 * epsilon)


def read(model, c, variant="source_motion"):
    """Evaluate the fixed three variants without any current native reader."""
    if variant in base.VARIANTS:
        return base.read(model, c, variant == "context_kernel")
    if variant != "source_motion":
        raise ValueError("Unknown reader field")
    p, _ = base.weights(model, c)
    t = np.einsum("ih,ih->i", np.asarray(c) - model["c"], model["v"])
    d = p @ (model["d"] + t[:, None] * model["j"])
    return np.stack((model["m"] + d / 2, model["m"] - d / 2))


def derivative(model, c, dc):
    """Differentiate the motion field, including weight and local-slope terms."""
    p, _ = base.weights(model, c)
    h2 = float(model["h2"])
    dp = p * ((model["c"] - p @ model["c"]) @ dc) / h2 if h2 else np.zeros_like(p)
    t = np.einsum("ih,ih->i", c - model["c"], model["v"])
    dd = (
        dp @ (model["d"] + t[:, None] * model["j"])
        + (p * (model["v"] @ dc)) @ model["j"]
    )
    return np.stack((dd / 2, -dd / 2))


def basis(model, expanded=False):
    """Return the original or secant-extended output range, never recipient-fit."""
    rows = model["d"] - model["d"].mean(axis=0)
    if expanded:
        rows = np.concatenate((rows, model["j"]))
    _, s, vt = np.linalg.svd(rows, full_matrices=False)
    return vt[s > (s[0] * max(rows.shape) * np.finfo(float).eps if s[0] else 0)]


def range_account(model, target, predicted, expanded=False):
    """Orthogonal capacity account in the candidate's declared output range."""
    u = basis(model, expanded)
    z = target - model["d"].mean(axis=0)
    orth = z - (z @ u.T) @ u
    lower, error = float(orth @ orth), float(np.sum((predicted - target) ** 2))
    if lower > error + 1e-9:
        raise ValueError("Candidate outside its declared affine output range")
    return {
        "rank": len(u),
        "native_norm2": float(np.sum(target**2)),
        "vector_error2": error,
        "orthogonal_lower_bound2": lower,
        "in_span_error2": max(0.0, error - lower),
    }


def support_account(model, c):
    """Record extrapolation and source-output range change without tuning a gate."""
    p, _ = base.weights(model, c)
    t = np.einsum("ih,ih->i", c - model["c"], model["v"])
    epsilon = float(model["epsilon"])
    u = basis(model)
    j = model["j"]
    outside = j - (j @ u.T) @ u
    energy = float(np.sum(j * j))
    result = base.diagnostics(model, c)
    result.update(
        projected_distance_over_step=float(p @ abs(t) / epsilon) if epsilon else 0.0,
        projected_distance_over_h=float(p @ abs(t) / np.sqrt(float(model["h2"])))
        if epsilon
        else 0.0,
        original_rank=len(u),
        expanded_rank=len(basis(model, True)),
        source_motion_outside_fraction=float(np.sum(outside**2) / energy)
        if energy
        else 0.0,
    )
    return result


def native_reader(model, state, kwargs, ids, contrast, start=17, read_block=22):
    """Differentiate suffix17..end at post-attention22, not at the H16 input."""
    import torch

    if not start <= read_block < len(model.model.layers):
        raise ValueError("Reader hook must be inside the actual suffix")
    captured = {}

    def hook(_module, args):
        captured["post"] = args[0]

    handle = model.model.layers[
        read_block
    ].post_attention_layernorm.register_forward_pre_hook(hook)
    try:
        with torch.enable_grad():
            x = state.detach().requires_grad_(True)
            score = endpoint(model, x, kwargs, ids, contrast, block=start)
            (r,) = torch.autograd.grad(score, (captured["post"],))
        return float(score.detach().cpu()), r[0, -1].detach().cpu().numpy().astype(
            float
        )
    finally:
        handle.remove()


def shifted(data, v, epsilon):
    """Replace only last row, retaining all exact BODY/layout values."""
    state = data["state"]
    changed = replace_last(
        state, state[-1].astype(float) + epsilon * v, data["positions"]
    )
    return dict(data, state=changed)
