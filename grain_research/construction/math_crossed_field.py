#!/usr/bin/env python3
"""HM30 finite crossed-field projections with explicit physical action units."""

from __future__ import annotations

import numpy as np

from .math_position_support import unit_vector
from .math_source_field import moment_map

CONTEXTS = (
    "proofreading a public notice",
    "preparing a museum audio guide",
    "setting up a sports timetable",
    "assembling a hobby display",
    "sorting a school equipment list",
    "reviewing a cafeteria menu",
    "arranging a recycling schedule",
    "planning a campus tour",
)
WORDINGS = (
    "deeply worried and tense",
    "very nervous and worried",
    "quite anxious and tense",
    "really worried about failure",
)
ARMS = ("unseen_source", "known_mean")
METHODS = ("global_source", "wording_source", "local_derivative")


def material(context, wording):
    """Render every declared crossing without outcome-dependent phrase changes."""
    if not 0 <= context < len(CONTEXTS) or not 0 <= wording < len(WORDINGS):
        raise ValueError("declared context and wording required")
    return (
        f"During {CONTEXTS[context]}, I felt {WORDINGS[wording]} "
        "while I checked a printed list. I judged AI systems generally useful "
        "while I recorded a routine note."
    )


def directions(b):
    """Return the two fixed HM29 physical directions, without its random arm."""
    b = unit_vector(b)
    return {
        name: np.outer(np.linalg.solve(moment_map(4), target), b)
        for name, target in (
            ("unseen_source", [0.0, -1.0, 0.0, 1.0]),
            ("known_mean", [1.0, 0.0, 0.0, 0.0]),
        )
    }


def grid(values):
    """Require a complete finite balanced grid, not missing-cell interpolation."""
    x = np.asarray(values, float)
    if x.ndim < 2 or min(x.shape[:2]) < 2 or not np.isfinite(x).all():
        raise ValueError("finite complete context-by-wording grid required")
    return x


def energy(values, dual_metric=None):
    """Use the declared covector metric; frame changes must transport it too."""
    x = np.asarray(values, float)
    if dual_metric is None:
        return float(np.sum(x * x))
    g = np.asarray(dual_metric, float)
    if (
        x.ndim < 2
        or g.shape != (x.shape[-2], x.shape[-2])
        or not np.isfinite(g).all()
        or not np.allclose(g, g.T)
        or np.linalg.eigvalsh(g).min() <= 0
    ):
        raise ValueError("positive definite matched covector metric required")
    return float(np.sum(np.einsum("...id,ij,...jd->...", x, g, x)))


def decompose(values, dual_metric=None):
    """Orthogonally decompose an equally weighted finite product design."""
    x = grid(values)
    mu = x.mean(axis=(0, 1))
    c = x.mean(axis=1) - mu
    p = x.mean(axis=0) - mu
    interaction = x - mu - c[:, None] - p[None, :]
    centered = x - mu
    n = x.shape[0] * x.shape[1]
    parts = {
        "context": np.broadcast_to(c[:, None], x.shape),
        "wording": np.broadcast_to(p[None, :], x.shape),
        "interaction": interaction,
    }
    ss = {name: energy(v, dual_metric) / n for name, v in parts.items()}
    total = energy(centered, dual_metric) / n
    mean_energy = energy(mu, dual_metric)
    all_energy = energy(x, dual_metric) / n
    summary = {
        **{name + "_energy": v for name, v in ss.items()},
        "total_centered_energy": total,
        "mean_energy": mean_energy,
        "mean_square_energy": all_energy,
        "cancellation_ratio": mean_energy / all_energy if all_energy else None,
        "reconstruction_error": float(
            np.max(abs(x - mu - c[:, None] - p[None, :] - interaction))
        ),
        "energy_identity_error": abs(total - sum(ss.values())),
        **{name + "_share": v / total if total else None for name, v in ss.items()},
    }
    return {"mean": mu, "context": c, "wording": p, "interaction": interaction}, summary


def covariance_parts(left, right):
    """Keep signed paired-target covariance terms on exactly the same cells."""
    x, y = grid(left), grid(right)
    if x.ndim != 2 or x.shape != y.shape:
        raise ValueError("matched scalar crossed grids required")
    a, _ = decompose(x)
    b, _ = decompose(y)
    parts = {
        name: float(np.mean(a[name] * b[name]))
        for name in ("context", "wording", "interaction")
    }
    native = float(np.mean((x - x.mean()) * (y - y.mean())))
    return {
        "native": native,
        **parts,
        "identity_error": abs(native - sum(parts.values())),
    }


def summarize(rows):
    """Report raw and realized unit-displacement errors, never only signs."""
    if not rows:
        raise ValueError("addressed actual actions required")
    y = np.array([r["odd_action"] for r in rows])
    d = np.array([r["physical_delta_norm"] for r in rows])
    if not np.isfinite(d).all() or np.any(d <= 0):
        raise ValueError("positive actual physical displacement required")
    rms = lambda a: float(np.sqrt(np.mean(np.asarray(a, float) ** 2)))
    result = {
        "cells": len(rows),
        "contexts": len({r["context"] for r in rows}),
        "wordings": len({r["wording"] for r in rows}),
        "native_action_rms": rms(y),
        "native_unit_action_rms": rms(y / d),
        "even_change_rms": rms([r["even_change"] for r in rows]),
    }
    for method in METHODS:
        p = np.array([r[method] for r in rows])
        result[method + "_rmse"] = rms(p - y)
        result[method + "_unit_rmse"] = rms((p - y) / d)
        result[method + "_prediction_rms"] = rms(p)
        result[method + "_signs"] = int(sum(p * y > 0))
    return result
