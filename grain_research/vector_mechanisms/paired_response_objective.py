#!/usr/bin/env python3
"""Teach a source-normalized context response in the existing compatible scalar."""

from __future__ import annotations

import numpy as np

from . import full_context_geometry as base
from .context_modulated_potential import view_weights

ARM = "paired_response_objective"


def paired_indices(author_ids, members):
    """Each frozen source author contributes exactly one two-view pair."""
    ids = np.asarray(author_ids)
    if set(ids) != set(members):
        raise ValueError("Source membership differs")
    pairs = [np.flatnonzero(ids == u) for u in members]
    if any(len(p) != 2 for p in pairs):
        raise ValueError("Exactly two views per source author required")
    return pairs


def delta_energy(deltas, original_scale):
    """Equal-author mean-square difference, with no target-dependent floor."""
    if not deltas or not np.isfinite(original_scale) or original_scale <= 0:
        raise ValueError("Positive source scale and paired differences required")
    if any(not np.isfinite(x).all() or not x.size for x in deltas):
        raise ValueError("Finite nonempty source differences required")
    energy = float(np.mean([np.mean(x**2) for x in deltas]))
    if energy <= 1e-12 * original_scale**2:
        raise ValueError("Unsupported source delta energy")
    return energy


def normals(bodies, lasts, gradients, scores, author_ids, model):
    """Reconstruct the frozen base objective and add one PSD paired design block."""
    n = len(bodies)
    if not n or any(len(v) != n for v in (lasts, gradients, scores, author_ids)):
        raise ValueError("Aligned source views required")
    pairs = paired_indices(author_ids, model["members"])
    width = len(model["coef"])
    rank = model["basis"].shape[1]
    prefix = 1 + 2 * rank + len(model["landmarks"]) * (rank + 1)
    sl, sk = float(model["score_scale"]), float(model["density_scale"])
    targets, coordinates = [], []
    gram, rhs = np.zeros((width, width)), np.zeros(width)
    for h, last, g, y, w in zip(
        bodies, lasts, gradients, scores, view_weights(author_ids), strict=True
    ):
        if g.shape != h.shape or not all(
            np.isfinite(x).all() for x in (h, last, g, np.asarray(y))
        ):
            raise ValueError("Finite aligned BODY/last/teacher required")
        z, p, phi = base.design(h, last, model)
        coordinates.append((z, p))
        k = len(h) * (g @ model["basis"]) * model["scale"]
        targets.append(k)
        gram += w * np.outer(phi, phi) / sl**2
        rhs += w * phi * y / sl**2
        norm = np.sqrt(w / k.size) / sk
        for rows, d in base.density_blocks(z, p, last, model):
            a, b = d.reshape(-1, width) * norm, k[rows].ravel() * norm
            gram += a.T @ a
            rhs += a.T @ b
    error = float(
        max(np.max(abs(gram - model["gram"])), np.max(abs(rhs - model["rhs"])))
    )
    if error > 1e-9:
        raise ValueError("Frozen base normal equations differ")
    deltas = []
    for i, j in pairs:
        if not np.array_equal(bodies[i], bodies[j]):
            raise ValueError("Paired BODY must be identical")
        deltas.append((targets[j] - targets[i]) / 2)
    energy = delta_energy(deltas, sk)
    dg, dr = np.zeros_like(gram), np.zeros_like(rhs)
    prefix_error = 0.0
    for (i, j), target in zip(pairs, deltas, strict=True):
        z, p = coordinates[i]
        norm = 1 / np.sqrt(len(pairs) * target.size * energy)
        blocks = zip(
            base.density_blocks(z, p, lasts[i], model),
            base.density_blocks(z, p, lasts[j], model),
            strict=True,
        )
        for (rows, d0), (rows1, d1) in blocks:
            if rows != rows1:
                raise ValueError("Pair block alignment differs")
            d = (d1 - d0) / 2
            prefix_error = max(prefix_error, float(np.max(abs(d[:, :, :prefix]))))
            a, b = d.reshape(-1, width) * norm, target[rows].ravel() * norm
            dg += a.T @ a
            dr += a.T @ b
    if prefix_error != 0:
        raise ValueError("Additive paired derivative prefix did not cancel")
    return (
        dg,
        dr,
        energy,
        {
            "base_normal_error": error,
            "additive_delta_prefix_error": prefix_error,
            "source_delta_energy": energy,
            "extra_half_difference_weight": sk**2 / energy,
            "source_authors": len(pairs),
            "source_views": n,
        },
    )


def objective(bodies, lasts, gradients, scores, author_ids, model, energy):
    """Independently compute every loss from predictions, not normal equations."""
    pairs = paired_indices(author_ids, model["members"])
    value_errors, field_errors, deltas, estimates = [], [], [], []
    for h, last, g, y in zip(bodies, lasts, gradients, scores, strict=True):
        value, density = base.predict(h, last, model)
        target = len(h) * (g @ model["basis"]) * model["scale"]
        value_errors.append((value - y) ** 2 / float(model["score_scale"]) ** 2)
        field_errors.append(
            np.mean((density - target) ** 2) / float(model["density_scale"]) ** 2
        )
        deltas.append(target)
        estimates.append(density)
    weights = view_weights(author_ids)
    delta_loss = (
        np.mean(
            [
                np.mean(
                    ((estimates[j] - estimates[i] - deltas[j] + deltas[i]) / 2) ** 2
                )
                for i, j in pairs
            ]
        )
        / energy
    )
    return np.array(
        [
            weights @ value_errors,
            weights @ field_errors,
            delta_loss,
            float(model["alpha"]) * np.sum(model["coef"][1:] ** 2),
        ]
    )


def fit(bodies, lasts, gradients, scores, author_ids, frozen):
    """One same-dictionary solve, adding only the frozen-plan paired response loss."""
    dg, dr, energy, diag = normals(bodies, lasts, gradients, scores, author_ids, frozen)
    gram, rhs = frozen["gram"] + dg, frozen["rhs"] + dr
    penalty = np.eye(len(rhs)) * float(frozen["alpha"])
    penalty[0, 0] = 0
    coef = np.linalg.solve(gram + penalty, rhs)
    residual = float(np.max(abs((gram + penalty) @ coef - rhs)))
    model = dict(
        frozen,
        coef=coef,
        gram=gram,
        rhs=rhs,
        delta_gram=dg,
        delta_rhs=dr,
        source_delta_energy=np.array(energy),
        arm=np.array(ARM),
    )
    old = objective(bodies, lasts, gradients, scores, author_ids, frozen, energy)
    new = objective(bodies, lasts, gradients, scores, author_ids, model, energy)
    difference = coef - frozen["coef"]
    analytic_change = 2 * difference @ ((gram + penalty) @ frozen["coef"] - rhs)
    analytic_change += difference @ (gram + penalty) @ difference
    objective_error = float(abs((new.sum() - old.sum()) - analytic_change))
    if (
        not np.isfinite(coef).all()
        or residual > 1e-8
        or new.sum() > old.sum() + 1e-8
        or objective_error > 1e-8
    ):
        raise ValueError("Paired scalar objective failed independent reconstruction")
    return model, diag | {
        "candidate": ARM,
        "parameters": len(coef),
        "source_score_loss": float(new[0]),
        "source_density_loss": float(new[1]),
        "source_delta_loss": float(new[2]),
        "source_penalty": float(new[3]),
        "source_objective": float(new.sum()),
        "source_original_objective": float(new[[0, 1, 3]].sum()),
        "frozen_original_objective": float(old[[0, 1, 3]].sum()),
        "frozen_augmented_objective": float(old.sum()),
        "frozen_delta_loss": float(old[2]),
        "normal_equation_error": residual,
        "objective_reconstruction_error": objective_error,
    }
