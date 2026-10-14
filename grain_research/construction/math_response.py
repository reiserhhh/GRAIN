#!/usr/bin/env python3
"""Finite paired observation models and source-calibrated target readouts.

The constrained inverse is standard GLS/LCMV, not a new mathematical theorem.
Generator truth is kept outside the fit interfaces. Text units are declared
material contrasts, not human psychological scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.linalg import null_space

from . import hidden_evidence_map as historical

SEEDS = (71, 173, 379)
WORLDS = ("shared", "repairable_interaction", "conditional", "collision")
ROLES = {"source": 300, "validation": 100, "application": 200}


@dataclass(frozen=True)
class Observations:
    """Permitted estimator input: observed vectors and assigned control codes."""

    vectors: np.ndarray
    controls: np.ndarray
    group: np.ndarray
    covariance_samples: np.ndarray


def effect_design(bits: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Return the complete balanced Walsh design, without an intercept."""
    names = []
    columns = []
    for mask in range(1, 1 << bits.shape[1]):
        axes = [i for i in range(bits.shape[1]) if mask & (1 << i)]
        names.append("*".join(str(i) for i in axes))
        columns.append(np.prod(bits[:, axes], axis=1))
    return np.column_stack(columns), names


def centered_by_group(x: np.ndarray, group: np.ndarray) -> np.ndarray:
    """Remove paired-material intercepts only for estimating operation columns."""
    result = np.array(x, dtype=float, copy=True)
    for g in np.unique(group):
        m = group == g
        result[m] -= result[m].mean(axis=0)
    return result


def fit_loadings(
    x: np.ndarray, q: np.ndarray, group: np.ndarray
) -> tuple[np.ndarray, float]:
    """Estimate forward columns from source observations, never hidden maps."""
    xc = centered_by_group(x, group)
    qc = centered_by_group(q, group)
    coef = np.linalg.lstsq(qc, xc, rcond=None)[0].T
    residual = xc - qc @ coef.T
    df = max(len(x) - len(np.unique(group)) - q.shape[1], 1)
    variance = float(np.sum(residual**2) / (df * x.shape[1]))
    return coef, variance


def target_dual(
    m: np.ndarray,
    ell: np.ndarray,
    variance: float = 1.0,
    resolution: float = 1e-10,
    constraint_tolerance: float = 1e-7,
) -> dict:
    """Solve M.T w=ell in an explicitly resolved action space.

    Isotropic residual noise is a source shrinkage model. The constraint check
    distinguishes infeasibility at that resolution from an exact oracle proof.
    """
    u, s, vt = np.linalg.svd(m, full_matrices=False)
    keep = s > resolution
    represented = vt[keep].T @ (vt[keep] @ ell)
    defect = float(np.linalg.norm(ell - represented))
    if defect > constraint_tolerance:
        return {
            "status": "unresolved",
            "defect": defect,
            "singular_values": s,
            "rank": int(keep.sum()),
            "w": None,
        }
    w = u[:, keep] @ ((vt[keep] @ ell) / s[keep])
    return {
        "status": "identified",
        "w": w,
        "defect": defect,
        "rank": int(keep.sum()),
        "singular_values": s,
        "noise_variance": float(variance * (w @ w)),
    }


def gls_target(m: np.ndarray, ell: np.ndarray) -> np.ndarray:
    """Independent GLS formula for a full-column-rank, isotropic model."""
    return m @ np.linalg.solve(m.T @ m, ell)


def infeasibility_witness(m: np.ndarray, ell: np.ndarray) -> dict:
    """Produce a kernel certificate: M v=0 but ell.T v is nonzero."""
    kernel = null_space(m)
    if not kernel.size:
        return {"exists": False}
    j = int(np.argmax(np.abs(ell @ kernel)))
    v = kernel[:, j]
    return {
        "exists": bool(abs(ell @ v) > 1e-8),
        "vector": v.tolist(),
        "map_residual": float(np.linalg.norm(m @ v)),
        "target_contrast": float(ell @ v),
    }


def finite_world(seed: int, rho: float, world: str) -> tuple[dict, dict]:
    """Generate separated paired roles using historical C1's 64-D mixing basis."""
    maps = historical.world_maps(seed)
    v = maps["u_e"]
    s = maps["u_s"] + 0.6 * v
    rng = np.random.default_rng([seed, 107, 0])
    j = rng.normal(size=historical.D)
    j /= np.linalg.norm(j)
    cells = np.array(list(product((-1.0, 1.0), repeat=3)))
    t, r, c = cells.T
    q = np.column_stack([t, r, t * r, r * c])
    m = np.column_stack([v, s, np.zeros_like(v), np.zeros_like(v)])
    if world == "repairable_interaction":
        m[:, 2] = 0.9 * j + 0.65 * v
    elif world == "conditional":
        m[:, 3] = 0.8 * v
    elif world == "collision":
        m[:, 1] = v
    elif world != "shared":
        raise ValueError(world)
    views, truth = {}, {"M": m, "target": v, "cells": cells}
    for k, (role, n) in enumerate(ROLES.items(), 1):
        gen = np.random.default_rng([seed, 107, k])
        fingerprints = gen.normal(size=(n, historical.KF)) @ maps["U_f"].T
        noise = gen.normal(scale=0.15, size=(n, 8, historical.D))
        x = fingerprints[:, None] + (q @ m.T)[None] + noise
        a = gen.normal(size=n)
        b = rho * a + np.sqrt(1 - rho**2) * gen.normal(size=n)
        # Recorded covariance observations: repeat noise permits a fixed rank-2
        # covariance reference. These do not reveal b to the estimator.
        cov_samples = a[:, None] * v + b[:, None] * m[:, 1]
        cov_samples += gen.normal(scale=0.03, size=cov_samples.shape)
        views[role] = Observations(
            x.reshape(-1, historical.D),
            np.tile(q, (n, 1)),
            np.repeat(np.arange(n) + k * 10000, 8),
            cov_samples,
        )
        truth[role] = {"target": np.tile(t, n), "vectors": x, "theta": a, "eta": b}
    return views, truth


def covariance_readout(samples: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Historical single-constraint rank-2 covariance reference (not full C1 replay)."""
    cov = np.cov(samples, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    u, lam = eigenvectors[:, -2:], np.maximum(eigenvalues[-2:], 1e-10)
    a = u.T @ v
    raw = u @ (a / lam)
    return raw / (v @ raw)


def bootstrap_interval(
    values: np.ndarray, seed: int = 17, draws: int = 1000
) -> list[float]:
    """Percentile interval of a mean, resampling independent material families."""
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (draws, len(values)))].mean(axis=1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def finite_experiments() -> tuple[list[dict], dict, dict]:
    """Run all 24 finite configurations with source-only estimated columns."""
    rows, witnesses, arrays = [], {}, {}
    for world, rho, seed in product(WORLDS, (0.0, 0.6), SEEDS):
        views, truth = finite_world(seed, rho, world)
        src, app = views["source"], views["application"]
        m, variance = fit_loadings(src.vectors, src.controls, src.group)
        # Uncertainty of a source column, estimated across paired authors.
        per_author = src.vectors.reshape(300, 8, -1)
        per_columns = np.einsum("nkd,kj->ndj", per_author, src.controls[:8]) / 8
        resolution = 3 * float(
            np.sqrt(np.sum(np.var(per_columns, axis=0, ddof=1)) / 300)
        )
        key = f"{world}_rho{rho}_seed{seed}"
        arrays[key] = m
        target = truth["target"]
        # Primary omitted-interaction model and prespecified repairs.
        specifications = [("paired_base", [0, 1]), ("paired_extended", [0, 1, 2])]
        if world == "conditional":
            specifications += [("paired_global", [0, 1, 3])]
        plans = [
            (
                "covariance_single",
                covariance_readout(src.covariance_samples, m[:, 0]),
                "identified",
                None,
            )
        ]
        for name, cols in specifications:
            ell = np.zeros(len(cols))
            ell[0] = 1
            constraint_tolerance = 2 * resolution / np.linalg.norm(m[:, cols], 2)
            solved = target_dual(
                m[:, cols], ell, variance, resolution, constraint_tolerance
            )
            w = solved["w"]
            equivalent = (
                float(np.max(np.abs(w - gls_target(m[:, cols], ell))))
                if w is not None and len(cols) == solved["rank"]
                else None
            )
            plans.append((name, w, solved["status"], equivalent))
        if world == "conditional":
            for condition in (-1, 1):
                cols = np.column_stack([m[:, 0], m[:, 1] + condition * m[:, 3]])
                solved = target_dual(cols, np.array([1.0, 0.0]), variance, resolution)
                plans.append(
                    (f"conditional_{condition}", solved["w"], solved["status"], None)
                )
        x = centered_by_group(app.vectors, app.group)
        for name, w, status, equivalent in plans:
            row = {
                "world": world,
                "rho": rho,
                "seed": seed,
                "arm": name,
                "status": status,
                "loading_error": float(np.linalg.norm(m - truth["M"])),
                "source_resolution": resolution,
                "gls_dual_max_error": equivalent,
            }
            if w is not None:
                mask = np.ones(len(x), bool)
                if name.startswith("conditional_"):
                    mask = np.tile(truth["cells"][:, 2], 200) == int(
                        name.split("_")[-1]
                    )
                score = x @ w
                errors = score[mask] - truth["application"]["target"][mask]
                nuisance = truth["M"][:, 1:]
                if name.startswith("conditional_"):
                    nuisance = (
                        truth["M"][:, 1] + int(name.split("_")[-1]) * truth["M"][:, 3]
                    )[:, None]
                # Coverage includes source loading uncertainty, estimated by
                # leave-source-author contributions, not fixed-M noise alone.
                influence = np.einsum("d,ndj->nj", w, per_columns - m[None])
                calibration_var = float(np.var(influence, axis=0, ddof=1).sum() / 300)
                predictive_sd = np.sqrt(variance * (w @ w) * 7 / 8 + calibration_var)
                cell_mask = mask.reshape(200, 8)[0]
                family_mse = np.mean(
                    (score - truth["application"]["target"]).reshape(200, 8)[
                        :, cell_mask
                    ]
                    ** 2,
                    axis=1,
                )
                row.update(
                    unit_gain=float(w @ target),
                    style_leak=float(np.linalg.norm(w @ nuisance)),
                    mse=float(np.mean(errors**2)),
                    mse_ci=bootstrap_interval(family_mse, seed),
                    coverage=float(np.mean(np.abs(errors) <= 1.96 * predictive_sd)),
                    noise_variance=float(variance * (w @ w)),
                )
            rows.append(row)
        active = [0, 1, 3] if world == "conditional" else [0, 1]
        ell = np.zeros(len(active))
        ell[0] = 1
        witnesses[key] = infeasibility_witness(truth["M"][:, active], ell)
    return rows, witnesses, arrays


def ridge_weights(
    x: np.ndarray, y: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Source-only dual ridge, avoiding a 1024x1024 inverse."""
    mu, ymu = x.mean(axis=0), y.mean(axis=0)
    xc = x - mu
    w = xc.T @ np.linalg.solve(xc @ xc.T + alpha * np.eye(len(x)), y - ymu)
    return w, mu, ymu


def family_effects(
    scores: np.ndarray, bits: np.ndarray, group: np.ndarray, target: int
) -> list[dict]:
    """Compute matched target, style and ownership operations per family."""
    rows = []
    for family in np.unique(group):
        mask = group == family
        z, s = bits[mask], scores[mask, target]
        own = z[:, 2] == 1
        # Averages cancel all other balanced assigned conditions.
        target_gain = float(np.mean(s[own] * z[own, 0]) * 2)
        if target == 1:
            target_gain = float(np.mean(s[own] * z[own, 0] * z[own, 3]) * 2)
        paired_style = []
        for p, o, c in product((-1, 1), repeat=3):
            a = (z[:, 0] == p) & (z[:, 2] == o) & (z[:, 3] == c)
            paired_style.append(
                float(s[a & (z[:, 1] == 1)][0] - s[a & (z[:, 1] == -1)][0])
            )
        other = z[:, 2] == -1
        other_gain = float(np.mean(s[other] * z[other, 0]) * 2)
        if target == 1:
            other_gain = float(np.mean(s[other] * z[other, 0] * z[other, 3]) * 2)
        interaction = float(np.mean(s * z[:, 0] * z[:, 2]) * 4)
        style_joint = float(np.mean(s * z[:, 0] * z[:, 1]) * 4)
        rows.append(
            {
                "family": int(family),
                "target_gain": target_gain,
                "style_rms": float(np.sqrt(np.mean(np.square(paired_style)))),
                "other_target_gain": other_gain,
                "joint_pole_owner": interaction,
                "joint_pole_style": style_joint,
            }
        )
    return rows


def text_targets(bits: np.ndarray) -> np.ndarray:
    """Declared material units, with ineligible evidence having zero contribution."""
    y = bits[:, 0] * (bits[:, 2] + 1) / 2
    return np.column_stack([y, y * bits[:, 3]])


def fit_text_models(
    x: np.ndarray, bits: np.ndarray, group: np.ndarray, roles: np.ndarray
) -> tuple[dict, dict]:
    """Fit on source, select ridge on validation; never access held targets."""
    q, names = effect_design(bits)
    source, valid = roles == "source", roles == "validation"
    y = text_targets(bits[source])
    m, variance = fit_loadings(x[source], q[source], group[source])
    ell = np.zeros((15, 2))
    ell[names.index("0"), 0] = ell[names.index("0*2"), 0] = 0.5
    ell[names.index("0*3"), 1] = ell[names.index("0*2*3"), 1] = 0.5
    solved = [target_dual(m, ell[:, i], variance) for i in range(2)]
    if any(s["w"] is None for s in solved):
        raise ValueError("source action columns cannot identify both declared targets")
    w = np.column_stack([s["w"] for s in solved])
    mu = x[source].mean(axis=0)
    gls = np.column_stack([gls_target(m, ell[:, i]) for i in range(2)])
    alphas, candidates = (0.001, 0.01, 0.1, 1.0, 10.0), []
    for alpha in alphas:
        wr, mur, ymur = ridge_weights(x[source], y, alpha)
        pred = (x[valid] - mur) @ wr + ymur
        loss = float(np.mean((pred - text_targets(bits[valid])) ** 2))
        candidates.append((loss, alpha, wr, mur, ymur))
    best = min(candidates, key=lambda v: v[0])
    d = (x[source] - mu).T @ y / (y[:, 0] @ y[:, 0])
    # Single-constraint covariance reference within the resolved source space.
    u, _, _ = np.linalg.svd(m, full_matrices=False)
    z = (x[source] - mu) @ u
    cov = np.cov(z, rowvar=False) + variance * np.eye(15)
    cw = []
    for i in range(2):
        a = u.T @ d[:, i]
        raw = u @ np.linalg.solve(cov, a)
        cw.append(raw / (d[:, i] @ raw))
    # Additive action-space ablation, not an intentionally under-informed fit.
    additive_cols = [names.index(str(i)) for i in range(4)]
    additive_u, _, _ = np.linalg.svd(m[:, additive_cols], full_matrices=False)
    additive_w = additive_u @ (additive_u.T @ w)
    rng = np.random.default_rng(107)
    permuted_w, permuted_mu, permuted_intercept = ridge_weights(
        x[source], y[rng.permutation(len(y))], best[1]
    )
    blind_y = np.column_stack([bits[source, 0], bits[source, 0] * bits[source, 3]])
    blind_w, blind_mu, blind_intercept = ridge_weights(x[source], blind_y, best[1])
    prototype = d / np.linalg.norm(d, axis=0)
    models = {
        "paired_dual": {"w": w, "mu": mu, "intercept": np.zeros(2)},
        "paired_gls": {"w": gls, "mu": mu, "intercept": np.zeros(2)},
        "ridge": {"w": best[2], "mu": best[3], "intercept": best[4]},
        "covariance_single": {
            "w": np.column_stack(cw),
            "mu": mu,
            "intercept": np.zeros(2),
        },
        "additive_space": {"w": additive_w, "mu": mu, "intercept": np.zeros(2)},
        "label_permutation": {
            "w": permuted_w,
            "mu": permuted_mu,
            "intercept": permuted_intercept,
        },
        "ownership_blind": {"w": blind_w, "mu": blind_mu, "intercept": blind_intercept},
        "uncalibrated_prototype": {"w": prototype, "mu": mu, "intercept": np.zeros(2)},
    }
    info = {
        "columns": names,
        "source_variance": variance,
        "action_rank": int(np.linalg.matrix_rank(m)),
        "singular_values": np.linalg.svd(m, compute_uv=False).tolist(),
        "gls_dual_max_error": float(np.max(np.abs(w - gls))),
        "selected_ridge_alpha": best[1],
        "validation_grid": [{"mse": b[0], "alpha": b[1]} for b in candidates],
    }
    return models, info


def score_bundle(model: dict, x: np.ndarray) -> np.ndarray:
    """Cold score: accepts only vectors and frozen coefficients, not labels/IDs."""
    if "factor_w" in model:
        factors = (x - model["factor_mu"]) @ model["factor_w"] + model[
            "factor_intercept"
        ]
        features = binding_features(factors, bool(model["joint"]))
        return (features - model["mu"]) @ model["w"] + model["intercept"]
    return (x - model["mu"]) @ model["w"] + model["intercept"]


def binding_features(factors: np.ndarray, joint: bool) -> np.ndarray:
    """Construct multilinear action coordinates, not a product likelihood."""
    p, o, c = factors.T
    if not joint:
        return factors
    return np.column_stack([p, o, c, p * o, p * c, o * c, p * o * c])


def fit_binding_model(
    x: np.ndarray, bits: np.ndarray, groups: np.ndarray, roles: np.ndarray, joint: bool
) -> tuple[dict, dict]:
    """Cross-fit strong single actions, then jointly calibrate material units.

    Calibration sees source out-of-family predictions. Factors are continuous
    readout features, not independent probabilities. Selection uses validation
    families only; confirmation/test rows are never used by this fit.
    """
    source, valid = roles == "source", roles == "validation"
    source_targets = bits[source][:, [0, 2, 3]].astype(float)
    valid_targets = bits[valid][:, [0, 2, 3]].astype(float)
    candidates = []
    for alpha in (0.001, 0.01, 0.1, 1.0, 10.0):
        w, mu, intercept = ridge_weights(x[source], source_targets, alpha)
        loss = float(np.mean(((x[valid] - mu) @ w + intercept - valid_targets) ** 2))
        candidates.append((loss, alpha, w, mu, intercept))
    best = min(candidates, key=lambda t: t[0])
    oof = np.zeros((source.sum(), 3))
    source_groups = groups[source]
    for family in np.unique(source_groups):
        held = source_groups == family
        w, mu, intercept = ridge_weights(
            x[source][~held], source_targets[~held], best[1]
        )
        oof[held] = (x[source][held] - mu) @ w + intercept
    features = binding_features(oof, joint)
    w, mu, intercept = ridge_weights(features, text_targets(bits[source]), 0.1)
    model = {
        "factor_w": best[2],
        "factor_mu": best[3],
        "factor_intercept": best[4],
        "joint": np.array(joint),
        "w": w,
        "mu": mu,
        "intercept": intercept,
    }
    return model, {
        "factor_alpha": best[1],
        "calibration_alpha": 0.1,
        "calibration_mode": "source leave-family-out predicted features",
        "validation_factor_mse": best[0],
        "source_oof_factor_mse": np.mean((oof - source_targets) ** 2, axis=0).tolist(),
    }


def dynamics_probe() -> tuple[list[dict], dict]:
    """Fixed-metric, matched-spectrum coupled dynamics with a rotation counterexample."""
    matrices = {
        "normal": np.eye(2) * 0.8,
        "nonnormal": np.array([[0.8, 4.0], [0, 0.8]]),
        "normal_rotation": 0.8 * np.array([[0, -1.0], [1.0, 0]]),
    }
    b, c, s = (
        np.array([0.0, 1.0]),
        np.array([1.0, 0.0]),
        np.array([[2.0, 0.7], [0, 0.4]]),
    )
    rows, fits = [], {}
    for name, a in matrices.items():
        rng = np.random.default_rng(107)
        x = np.zeros((60, 33, 2))
        drive = rng.normal(size=(60, 32))
        for t in range(32):
            x[:, t + 1] = (
                x[:, t] @ a.T
                + drive[:, t, None] * b
                + rng.normal(scale=0.03, size=(60, 2))
            )
        # Source trajectories only; unknown A/B estimated from measured state/input.
        design = np.column_stack([x[:40, :-1].reshape(-1, 2), drive[:40].ravel()])
        estimated = np.linalg.lstsq(design, x[:40, 1:].reshape(-1, 2), rcond=None)[0].T
        ah, bh = estimated[:, :2], estimated[:, 2]
        fits[name] = {
            "estimated_A": ah.tolist(),
            "estimated_B": bh.tolist(),
            "A_error": float(np.linalg.norm(ah - a)),
            "held_one_step_mse": float(
                np.mean(
                    (x[40:, 1:] - x[40:, :-1] @ ah.T - drive[40:, :, None] * bh) ** 2
                )
            ),
        }
        for k in range(13):
            ak = np.linalg.matrix_power(a, k)
            transformed = np.linalg.matrix_power(s @ a @ np.linalg.inv(s), k)
            response = float(c @ ak @ b)
            covariant = float((c @ np.linalg.inv(s)) @ transformed @ (s @ b))
            rows.append(
                {
                    "system": name,
                    "lag": k,
                    "gain": float(np.linalg.norm(ak, 2)),
                    "response": response,
                    "estimated_response": float(c @ np.linalg.matrix_power(ah, k) @ bh),
                    "gauge_error": abs(response - covariant),
                    "commutator_norm": float(np.linalg.norm(a.T @ a - a @ a.T)),
                }
            )
    return rows, fits
