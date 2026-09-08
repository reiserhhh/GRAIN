#!/usr/bin/env python3
"""Query-region gradients, finite path integration and frozen matrix readouts."""

from __future__ import annotations

import numpy as np
import torch


def region_gradient(observer, context, region, mask, kwargs, alphabet, block=16):
    """Differentiate only a query-region leaf through the frozen suffix, no prefix."""
    model = observer.model
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError("Freeze all model weights before measuring region gradients")
    select = torch.as_tensor(mask, dtype=torch.bool, device=context.device)
    if select.shape != (context.shape[1],) or bool(select[-1]):
        raise ValueError("Region must match tokens and exclude final query")
    indices = torch.nonzero(select, as_tuple=True)[0]
    if region.shape != (context.shape[0], len(indices), context.shape[2]):
        raise ValueError("Region shape differs from declared token mask")
    observer.calls += 1
    with torch.enable_grad():
        leaf = region.detach().clone().requires_grad_(True)
        h = torch.index_copy(context.detach(), 1, indices, leaf)
        for layer in model.model.layers[block + 1 :]:
            h = layer(h, **kwargs)
            h = h[0] if isinstance(h, tuple) else h
        logits = model.lm_head(model.model.norm(h[:, -1:]))[0, -1].float()
        ids = torch.as_tensor(alphabet["token_ids"], device=logits.device)
        values = torch.as_tensor(
            alphabet["values"], device=logits.device, dtype=logits.dtype
        )
        score = torch.softmax(logits[ids], dim=0) @ values
        (gradient,) = torch.autograd.grad(score, leaf)
    value = float(score.detach().cpu())
    result = gradient[0].detach().float().cpu().numpy().copy()
    if not np.isfinite(value) or not np.isfinite(result).all():
        raise ValueError("Nonfinite region score or gradient")
    return value, result


def integrated_matrix(gradients):
    """Average the two E paths with three-node Simpson weights; retain token axes."""
    g = np.asarray(gradients, dtype=np.float64)
    if g.ndim != 4 or g.shape[:2] != (2, 3) or not np.isfinite(g).all():
        raise ValueError("Expected finite E(2) x node(3) x token x hidden gradients")
    return np.einsum("etij,t->ij", g, np.array([1.0, 4.0, 1.0]) / 12)


def matrix_response(matrix, displacement):
    """Frobenius contraction in original score units, without norm calibration."""
    g, d = np.asarray(matrix, float), np.asarray(displacement, float)
    if g.ndim != 2 or g.shape != d.shape or not np.isfinite([g, d]).all():
        raise ValueError("Matched finite token-by-hidden matrices required")
    return float(np.sum(g * d))


def frozen_reference_matrix(matrices, roles):
    """Estimate the readout from reference authors only; eval matrices are ignored."""
    roles = np.asarray(roles)
    g = np.asarray(matrices, dtype=np.float64)
    if g.ndim != 3 or roles.shape != (len(g),):
        raise ValueError("One role per author matrix required")
    if set(roles) != {"reference", "evaluation"}:
        raise ValueError("Explicit nonempty reference and evaluation groups required")
    return g[roles == "reference"].mean(axis=0)


def matrix_variants(matrix):
    """Predeclared mean-token and rank1/rank4 controls, fitted to the reference only."""
    g = np.asarray(matrix, float)
    u, s, vt = np.linalg.svd(g, full_matrices=False)
    energy = float(s @ s)
    variants = {"frozen_full": g, "token_mean": np.broadcast_to(g.mean(0), g.shape)}
    info = {}
    for rank in (1, 4):
        r = min(rank, len(s))
        variants[f"rank{rank}"] = (u[:, :r] * s[:r]) @ vt[:r]
        info[f"rank{rank}_energy"] = float(s[:r] @ s[:r] / energy) if energy else 0.0
    info["rank95"] = int(np.searchsorted(np.cumsum(s**2), 0.95 * energy) + 1)
    return variants, info


def response_quality(actual, predicted):
    """Finite-panel errors and signs; no Pearson for the two evaluation authors."""
    y, p = np.asarray(actual, float), np.asarray(predicted, float)
    if y.shape != p.shape or y.ndim != 1 or not np.isfinite([y, p]).all():
        raise ValueError("Finite paired one-dimensional responses required")
    energy = float(y @ y)
    return {
        "n": len(y),
        "rmse": float(np.sqrt(np.mean((p - y) ** 2))),
        "mae": float(np.mean(np.abs(p - y))),
        "error_energy_over_zero": float((p - y) @ (p - y) / energy) if energy else None,
        "same_sign": int(np.sum(p * y > 0)),
        "actual_rms": float(np.sqrt(np.mean(y**2))),
    }


def hidden_subspace_overlap(first, second, rank=4):
    """Compare right singular subspaces in shared hidden coordinates, not token IDs."""
    a, b = np.asarray(first, float), np.asarray(second, float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("Matrices must share model hidden coordinates")
    va = np.linalg.svd(a, full_matrices=False)[2][:rank]
    vb = np.linalg.svd(b, full_matrices=False)[2][:rank]
    return float(np.sum((va @ vb.T) ** 2) / min(len(va), len(vb)))
