#!/usr/bin/env python3
"""Finite-group evidence coordinates and event-local vector scoring.

The formal group acts on declared material codes, not assumed latent LLM
states. Scalar contributions outside qualification are zero bookkeeping units,
not observations of a neutral author. No psychological truth is inferred here.
"""

from __future__ import annotations

from itertools import product

import numpy as np
from sklearn.metrics import roc_auc_score

from .math_response import ridge_weights

FACTOR_NAMES = ("relevance", "polarity", "ownership", "actuality")


def characters(z: np.ndarray, joint: bool = True) -> np.ndarray:
    """Evaluate the nonconstant characters of a four-generator sign group."""
    z = np.asarray(z, float)
    if z.ndim != 2 or z.shape[1] != 4:
        raise ValueError("four evidence coordinates are required")
    if not joint:
        return z
    return np.column_stack(
        [
            np.prod(z[:, [j for j in range(4) if m & (1 << j)]], axis=1)
            for m in range(1, 16)
        ]
    )


def ideal_contribution(bits: np.ndarray) -> np.ndarray:
    """Return signed qualified action units, distinct from an author score."""
    bits = np.asarray(bits)
    q = np.all(bits[:, [0, 2, 3]] == 1, axis=1)
    return (q * bits[:, 1]).astype(float)[:, None]


def qualify(z: np.ndarray) -> np.ndarray:
    """Fixed midpoint routing on learned coordinates; not a tuned probability."""
    return np.all(np.asarray(z)[:, [0, 2, 3]] > 0, axis=1)


def affine_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> dict:
    """Fit a source-only ridge map with a centered intercept."""
    w, mu, bias = ridge_weights(x, y, alpha)
    return {"w": w, "mu": mu, "bias": bias}


def affine_score(model: dict, x: np.ndarray) -> np.ndarray:
    """Cold vector readout; no material identifiers or evaluation codes."""
    return (x - model["mu"]) @ model["w"] + model["bias"]


def rbf_kernel(x: np.ndarray, y: np.ndarray, gamma: float) -> np.ndarray:
    """Gaussian kernel on the same frozen, normalized embeddings."""
    squared = np.maximum(
        np.sum(x * x, axis=1)[:, None] + np.sum(y * y, axis=1)[None] - 2 * x @ y.T, 0
    )
    return np.exp(-gamma * squared)


def fit_programs(
    x: np.ndarray,
    bits: np.ndarray,
    family: np.ndarray,
    role: np.ndarray,
    half_design: bool = False,
) -> tuple[dict, dict]:
    """Fit on source; use only validation for finite alpha/gamma selection.

    Joint/additive calibrations see identical LOFO factor supervision. Direct
    Ridge/RBF receive only contribution labels; the supervision cost is reported.
    Half-design is an explicit source ablation, never a new test-set search.
    """
    src, val = role == "source", role == "validation"
    if half_design:
        src = src & (np.prod(bits[:, :4], axis=1) == 1)
    if set(family[src]) & set(family[val]):
        raise ValueError("source and validation families overlap")
    sx, sy = x[src], bits[src, :4].astype(float)
    vx, vy = x[val], bits[val, :4].astype(float)
    candidates = []
    for alpha in (0.01, 0.1, 1.0):
        model = affine_fit(sx, sy, alpha)
        loss = float(np.mean((affine_score(model, vx) - vy) ** 2))
        candidates.append((loss, alpha, model))
    loss, alpha, factors = min(candidates, key=lambda r: r[0])
    oof = np.zeros_like(sy)
    sf = family[src]
    for f in np.unique(sf):
        m = sf == f
        folded = affine_fit(sx[~m], sy[~m], alpha)
        oof[m] = affine_score(folded, sx[m])
    truth = ideal_contribution(bits[src])
    programs = {"factors": factors}
    for name, joint in (("additive", False), ("joint", True)):
        programs[name] = affine_fit(characters(oof, joint), truth, 0.1)
    direct_candidates = []
    for da in (0.01, 0.1, 1.0):
        dm = affine_fit(sx, truth, da)
        dl = float(np.mean((affine_score(dm, vx) - ideal_contribution(bits[val])) ** 2))
        direct_candidates.append((dl, da, dm))
    direct_loss, direct_alpha, programs["direct_ridge"] = min(
        direct_candidates, key=lambda r: r[0]
    )
    center = truth.mean(axis=0)
    kernel_candidates = []
    for gamma in (0.5, 2.0, 8.0):
        kernel = rbf_kernel(sx, sx, gamma)
        dual = np.linalg.solve(kernel + 0.1 * np.eye(len(sx)), truth - center)
        predicted = rbf_kernel(vx, sx, gamma) @ dual + center
        kloss = float(np.mean((predicted - ideal_contribution(bits[val])) ** 2))
        kernel_candidates.append((kloss, gamma, dual))
    kernel_loss, gamma, dual = min(kernel_candidates, key=lambda r: r[0])
    programs["rbf"] = {
        "reference": sx,
        "dual": dual,
        "gamma": np.array(gamma),
        "center": center,
    }
    shuffled = truth[np.random.default_rng(108).permutation(len(truth))]
    programs["permutation"] = affine_fit(sx, shuffled, direct_alpha)
    return programs, {
        "factor_alpha": alpha,
        "validation_factor_mse": loss,
        "factor_oof_mse": np.mean((oof - sy) ** 2, axis=0).tolist(),
        "direct_alpha": direct_alpha,
        "validation_direct_mse": direct_loss,
        "kernel_gamma": gamma,
        "validation_kernel_mse": kernel_loss,
        "source_rows": int(src.sum()),
        "validation_rows": int(val.sum()),
        "source_families": np.unique(sf).tolist(),
        "validation_families": np.unique(family[val]).tolist(),
        "calibration": "source LOFO factors; fixed ridge 0.1",
        "qualification": "three coordinate midpoints >0, fixed before results",
        "half_design": half_design,
    }


def score_programs(programs: dict, x: np.ndarray) -> tuple[dict, np.ndarray]:
    """Evaluate frozen programs on new vectors, without labels or source roles."""
    z = affine_score(programs["factors"], x)
    scores = {}
    for arm in ("additive", "joint"):
        scores[arm] = affine_score(programs[arm], characters(z, arm == "joint"))[:, 0]
    for arm in ("direct_ridge", "permutation"):
        scores[arm] = affine_score(programs[arm], x)[:, 0]
    km = programs["rbf"]
    scores["rbf"] = (
        rbf_kernel(x, km["reference"], float(km["gamma"])) @ km["dual"] + km["center"]
    )[:, 0]
    gate = np.prod(np.clip((z[:, [0, 2, 3]] + 1) / 2, 0, 1), axis=1)
    scores["declared_product"] = gate * z[:, 1]
    return scores, z


def metrics(scores: np.ndarray, z: np.ndarray, bits: np.ndarray) -> dict:
    """Measure contribution error and positive/negative qualification counts."""
    expected = ideal_contribution(bits)[:, 0]
    q = np.all(bits[:, [0, 2, 3]] == 1, axis=1)
    pred = qualify(z)
    out = {
        "unit_rmse": float(np.sqrt(np.mean((scores - expected) ** 2))),
        "qualified_rmse": float(np.sqrt(np.mean((scores[q] - expected[q]) ** 2))),
        "unqualified_rms": float(np.sqrt(np.mean(scores[~q] ** 2))),
        "qualification_tp": int(np.sum(pred & q)),
        "qualification_fp": int(np.sum(pred & ~q)),
        "qualification_fn": int(np.sum(~pred & q)),
        "qualification_tn": int(np.sum(~pred & ~q)),
    }
    for j, name in enumerate(FACTOR_NAMES):
        out[name + "_auc"] = float(roc_auc_score(bits[:, j] == 1, z[:, j]))
        out[name + "_rmse"] = float(np.sqrt(np.mean((z[:, j] - bits[:, j]) ** 2)))
    for name, m in (
        ("relevant", bits[:, 0] == 1),
        ("irrelevant", bits[:, 0] == -1),
        ("other", bits[:, 2] == -1),
        ("hypothetical", bits[:, 3] == -1),
    ):
        m = m & (bits[:, 2] == 1 if name not in ("other",) else True)
        m = m & (bits[:, 3] == 1 if name not in ("hypothetical",) else True)
        out[name + "_polarity_gain"] = float(2 * np.mean(scores[m] * bits[m, 1]))
    return out


def local_profile(
    events: list[dict], scores: np.ndarray, qualified: np.ndarray
) -> dict:
    """Aggregate distinct declared events, preserving absence as null.

    IDs identify supplied event relations, not automatically inferred semantic
    sameness. Identical-ID conflicting values are errors rather than free merges.
    """
    unique = {}
    for event, score, q in zip(events, scores, qualified, strict=True):
        identity = event["event_id"]
        state = (float(score), bool(q))
        if identity in unique and unique[identity] != state:
            raise ValueError("conflicting duplicate event requires explicit revision")
        unique[identity] = state
    observed = [s for s, q in unique.values() if q]
    return {
        "score": float(np.mean(observed)) if observed else None,
        "observed_events": len(observed),
        "declared_events": len(unique),
    }


def combinatorial_probe() -> dict:
    """Exhibit high-order aliasing despite balanced lower-order controls."""
    bits = np.array(list(product((-1, 1), repeat=4)))
    design = np.prod(bits, axis=1) == 1
    y = ideal_contribution(bits)[:, 0]
    # Two candidate response functions agree on all chosen controls.
    alternative = y + 0.25 * (1 - np.prod(bits, axis=1))
    phi = np.column_stack([np.ones(len(bits)), characters(bits)])
    coefficients = phi.T @ y / len(bits)
    return {
        "full_cells": 16,
        "half_cells": int(design.sum()),
        "half_rank": int(np.linalg.matrix_rank(phi[design])),
        "full_rank": int(np.linalg.matrix_rank(phi)),
        "first_order_imbalance": float(np.max(np.abs(bits[design].mean(axis=0)))),
        "second_order_imbalance": float(
            max(
                abs(np.mean(bits[design, i] * bits[design, j]))
                for i in range(4)
                for j in range(i)
            )
        ),
        "observed_function_difference": float(
            np.max(abs(y[design] - alternative[design]))
        ),
        "unobserved_function_difference": float(
            np.max(abs(y[~design] - alternative[~design]))
        ),
        "target_characters": {
            str(i): float(v) for i, v in enumerate(coefficients) if abs(v) > 1e-12
        },
        "scope": "unrestricted response class; exact target prior would remove ambiguity",
    }


def group_diagnostics(
    z: np.ndarray, bits: np.ndarray, family: np.ndarray, role: np.ndarray
) -> list[dict]:
    """Fit action maps on source and test closure on new material families."""
    lookup = {
        (int(f), tuple(b)): i for i, (f, b) in enumerate(zip(family, bits, strict=True))
    }
    maps, rows = [], []
    source, held = role == "source", role == "test"
    for j in range(4):
        changed = bits.copy()
        changed[:, j] *= -1
        paired = np.array(
            [lookup[(int(f), tuple(b))] for f, b in zip(family, changed, strict=True)]
        )
        model = affine_fit(z[source], z[paired[source]], 1e-3)
        maps.append(model)
        direct = affine_score(model, z[held])
        ideal = z[held].copy()
        ideal[:, j] *= -1
        rows.append(
            {
                "operation": FACTOR_NAMES[j],
                "held_affine_action_rmse": float(
                    np.sqrt(np.mean((direct - z[paired[held]]) ** 2))
                ),
                "held_signed_action_rmse": float(
                    np.sqrt(np.mean((ideal - z[paired[held]]) ** 2))
                ),
                "involution_defect": float(
                    np.sqrt(np.mean((affine_score(model, direct) - z[held]) ** 2))
                ),
            }
        )
    for i in range(4):
        for j in range(i):
            ij = affine_score(maps[i], affine_score(maps[j], z[held]))
            ji = affine_score(maps[j], affine_score(maps[i], z[held]))
            rows.append(
                {
                    "operation": FACTOR_NAMES[i] + "/" + FACTOR_NAMES[j],
                    "commutation_defect": float(np.sqrt(np.mean((ij - ji) ** 2))),
                }
            )
    return rows
