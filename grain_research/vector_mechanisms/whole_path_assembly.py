#!/usr/bin/env python3
"""Whole-function composition, paired metrics and error accounting for NEXT55."""

from __future__ import annotations

import numpy as np

from .compiled_query_transfer import QUERIES, fidelity
from .selection_endpoint import endpoint

CANDIDATES = ("identity", "normalized_value", "joint")


def reverse_response(model, state, directions, kwargs, ids, contrast, block=22):
    """One actual suffix reverse supplies all matched directional derivatives."""
    import torch

    if (
        state.ndim != 3
        or state.shape[0] != 1
        or directions.shape[1:] != state.shape[1:]
    ):
        raise ValueError(
            "One full state and aligned direction-by-token-by-hidden required"
        )
    with torch.enable_grad():
        x = state.detach().requires_grad_(True)
        score = endpoint(model, x, kwargs, ids, contrast, block=block)
        (gradient,) = torch.autograd.grad(score, (x,))
    # Match native cache contraction units after consistent FP32 input conversion.
    g = gradient.detach().cpu().numpy()[0]
    d = directions.detach().cpu().numpy()
    tau = np.einsum("th,fth->f", g.astype(float), d.astype(float))
    return float(score.detach().cpu()), g, tau


def paired_queries(frame, keys, values):
    """Subtract aligned raw query values before any nonlinear metric calculation."""
    if (
        set(frame.query_name) != set(QUERIES)
        or frame.duplicated(keys + ["query_name"]).any()
    ):
        raise ValueError("Exactly one row per identity and each frozen query required")
    a, b = [
        frame[frame.query_name == q].set_index(keys)[values].sort_index()
        for q in QUERIES
    ]
    if not a.index.equals(b.index):
        raise ValueError("Same query pair membership required")
    result = a - b
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("Finite paired values required")
    return result.reset_index()


def score_metrics(actual, predicted, constant):
    """Keep score origin, between-author scale and source-constant errors separate."""
    y, p, b = (np.asarray(v, float) for v in (actual, predicted, constant))
    r = fidelity(y, p)
    if b.shape != y.shape or not np.isfinite(b).all():
        raise ValueError("One finite source-only baseline per score required")
    yc, pc = y - y.mean(), p - p.mean()
    variance = float(np.mean(yc * yc))
    denominator = float(np.mean((y - b) ** 2))
    error = p - y
    return {
        "score_r": r["action_r"],
        "score_rmse": r["action_rmse"],
        "score_mse": float(np.mean(error * error)),
        "source_constant_mse": denominator,
        "score_mse_over_source_constant": float(np.mean(error * error) / denominator)
        if denominator
        else float("nan"),
        "actual_mean": float(y.mean()),
        "predicted_mean": float(p.mean()),
        "mean_error": float(error.mean()),
        "error_variance": float(error.var()),
        "actual_sd": float(y.std()),
        "predicted_sd": float(p.std()),
        "descriptive_slope": float(np.mean(yc * pc) / variance)
        if variance
        else float("nan"),
        "authors": len(y),
    }


def error_account(actual, baseline, candidate):
    """Account a frozen candidate's error change without recalibrating its outputs."""
    y, a, b = (np.asarray(v, float) for v in (actual, baseline, candidate))
    if (
        y.ndim != 1
        or y.shape != a.shape
        or y.shape != b.shape
        or not len(y)
        or not np.isfinite([y, a, b]).all()
    ):
        raise ValueError("Aligned finite outputs required")
    e, d = a - y, b - a
    before, after = float(np.mean(e * e)), float(np.mean((b - y) ** 2))
    linear, energy = float(2 * np.mean(e * d)), float(np.mean(d * d))
    return {
        "baseline_mse": before,
        "candidate_mse": after,
        "mse_change": after - before,
        "linear_error_term": linear,
        "correction_energy": energy,
        "error_correction_covariance": float(np.mean((e - e.mean()) * (d - d.mean()))),
        "baseline_mean_error": float(e.mean()),
        "correction_mean": float(d.mean()),
        "relative_error_reduction": 1 - after / before if before else float("nan"),
        "identity_error": abs(after - before - linear - energy),
        "observations": len(y),
    }
