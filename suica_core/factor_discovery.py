#!/usr/bin/env python3
"""Opportunity-conditioned factor discovery utilities for SUICA V6.

The module keeps three empirical objects separate:

``static``
    Stable author deviations after conditioning on the shared opportunity
    mean, plus the author's opportunity-choice profile.
``dynamic``
    Within-condition path descriptors computed after removing the shared
    opportunity mean and the author's local level.
``hybrid``
    Regularized author-specific response slopes on opportunity coordinates.

No personality labels enter construction, factor-number selection, rotation,
or confirmation.  Big Five/MBTI may only be joined downstream as external
anchors.
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def stable_user_split(user_id: str, salt: str = "suica-v6-factor-v1") -> str:
    """Return a deterministic, approximately balanced user-level split."""
    digest = hashlib.sha256(f"{salt}::{user_id}".encode()).digest()
    return "discovery" if int.from_bytes(digest[:8], "big") % 2 == 0 else "confirmation"


def safe_corr(x: Iterable[float], y: Iterable[float], min_n: int = 20) -> float:
    """Pearson correlation with finite-value and variance guards."""
    xa, ya = np.asarray(list(x), float), np.asarray(list(y), float)
    mask = np.isfinite(xa) & np.isfinite(ya)
    if mask.sum() < min_n or np.std(xa[mask]) < 1e-12 or np.std(ya[mask]) < 1e-12:
        return float("nan")
    return float(np.corrcoef(xa[mask], ya[mask])[0, 1])


@dataclass
class OpportunityModel:
    """Shared condition means and low-dimensional opportunity coordinates."""

    feature_cols: list[str]
    global_mean: np.ndarray
    condition_mean: dict[str, np.ndarray]
    condition_z: dict[str, np.ndarray]
    supported_conditions: set[str]
    axis_loadings: np.ndarray
    axis_explained_variance: np.ndarray


@dataclass
class StableCrossViewResult:
    """A discovery-only stable subspace estimated from early/late views."""

    n_factors: int
    directions: np.ndarray
    rotated_directions: np.ndarray
    stable_eigenvalues: np.ndarray
    permutation_thresholds: np.ndarray
    stable_share: np.ndarray
    center: np.ndarray
    scale: np.ndarray


def fit_opportunity_model(
    frame: pd.DataFrame,
    feature_cols: list[str],
    discovery_users: set[str],
    *,
    n_axes: int = 3,
    min_condition_users: int = 20,
) -> OpportunityModel:
    """Fit condition means and opportunity PCA using discovery users only.

    Condition means are first averaged within user-condition cells and then
    across users, preventing prolific authors from defining the opportunity.
    Unsupported conditions receive the discovery global mean and zero axes.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    d = frame.loc[frame["user_id"].astype(str).isin(discovery_users)].copy()
    cell = d.groupby(["user_id", "condition"], observed=True)[feature_cols].mean()
    support = cell.reset_index().groupby("condition")["user_id"].nunique()
    supported = set(support.loc[support >= min_condition_users].index.astype(str))
    if len(supported) < n_axes + 2:
        raise ValueError(
            f"Only {len(supported)} supported conditions; need at least {n_axes + 2}."
        )
    global_mean = cell[feature_cols].mean().to_numpy(float)
    cond_df = cell.groupby(level="condition")[feature_cols].mean()
    cond_supported = cond_df.loc[cond_df.index.astype(str).isin(supported)].copy()
    scaler = StandardScaler().fit(cond_supported.to_numpy(float))
    scaled = scaler.transform(cond_supported.to_numpy(float))
    k = min(n_axes, scaled.shape[0] - 1, scaled.shape[1])
    pca = PCA(n_components=k, random_state=0).fit(scaled)
    z = pca.transform(scaled)
    condition_mean = {str(c): cond_df.loc[c].to_numpy(float) for c in cond_df.index}
    condition_z = {str(c): z[i] for i, c in enumerate(cond_supported.index)}
    return OpportunityModel(
        feature_cols=feature_cols,
        global_mean=global_mean,
        condition_mean=condition_mean,
        condition_z=condition_z,
        supported_conditions=supported,
        axis_loadings=pca.components_.T,
        axis_explained_variance=pca.explained_variance_ratio_,
    )


def _condition_mean(model: OpportunityModel, condition: str) -> np.ndarray:
    if condition in model.supported_conditions:
        return model.condition_mean[condition]
    return model.global_mean


def _condition_z(model: OpportunityModel, condition: str) -> np.ndarray:
    return model.condition_z.get(condition, np.zeros(model.axis_loadings.shape[1]))


def _pooled_path_descriptors(sequences: list[np.ndarray]) -> dict[str, np.ndarray]:
    """Compute per-coordinate path descriptors over independent cells.

    Cell boundaries are respected: lag, roughness, trend, change-point and
    multiscale statistics never bridge two subreddits/conditions.
    """
    valid = [np.asarray(s, float) for s in sequences if len(s) >= 2]
    if not valid:
        return {}
    p = valid[0].shape[1]
    all_rows = np.vstack(valid)
    volatility = np.std(all_rows, axis=0, ddof=1) if len(all_rows) > 1 else np.full(p, np.nan)

    lag_num = np.zeros(p)
    lag_den = np.zeros(p)
    rough_sum = np.zeros(p)
    rough_n = 0
    slopes: list[np.ndarray] = []
    cps: list[np.ndarray] = []
    coarse_vars: list[np.ndarray] = []
    for seq in valid:
        centered = seq - np.mean(seq, axis=0, keepdims=True)
        lag_num += np.sum(centered[:-1] * centered[1:], axis=0)
        lag_den += np.sum(centered[:-1] ** 2, axis=0)
        diff = np.diff(seq, axis=0)
        rough_sum += np.sum(diff ** 2, axis=0)
        rough_n += len(diff)
        pos = np.linspace(-1.0, 1.0, len(seq))
        denom = float(np.sum(pos ** 2))
        slopes.append((pos[:, None] * centered).sum(axis=0) / max(denom, 1e-12))
        if len(seq) >= 4:
            cs = np.cumsum(centered, axis=0)[:-1]
            scale = np.sqrt(np.arange(1, len(seq))[:, None] *
                            (len(seq) - np.arange(1, len(seq)))[:, None] / len(seq))
            cps.append(np.max(np.abs(cs) / np.maximum(scale, 1e-12), axis=0))
            pairs = (seq[: len(seq) // 2 * 2:2] + seq[1: len(seq) // 2 * 2:2]) / 2.0
            if len(pairs) >= 2:
                coarse_vars.append(np.var(pairs, axis=0, ddof=1))

    shrink = rough_n / (rough_n + 10.0)
    inertia = shrink * np.divide(lag_num, lag_den, out=np.zeros(p), where=lag_den > 1e-12)
    roughness = np.sqrt(rough_sum / max(rough_n, 1))
    trend = np.median(np.vstack(slopes), axis=0)
    changepoint = np.median(np.vstack(cps), axis=0) if cps else np.full(p, np.nan)
    coarse = np.median(np.vstack(coarse_vars), axis=0) if coarse_vars else np.full(p, np.nan)
    multiscale = np.divide(coarse, volatility ** 2, out=np.full(p, np.nan),
                           where=volatility > 1e-12)
    return {
        "volatility": volatility,
        "inertia": inertia,
        "roughness": roughness,
        "trend": trend,
        "changepoint": changepoint,
        "multiscale": multiscale,
    }


def build_family_features(
    frame: pd.DataFrame,
    model: OpportunityModel,
    *,
    ridge_alpha: float = 2.0,
    min_slices: int = 8,
    min_conditions_hybrid: int = 5,
) -> dict[str, pd.DataFrame]:
    """Build user-half Static, Dynamic and Hybrid candidate matrices."""
    q = model.axis_loadings.shape[1]
    rows: dict[str, list[dict[str, float | str]]] = {
        "static": [], "dynamic": [], "hybrid": []
    }
    ordered = frame.sort_values(["user_id", "half", "condition", "slice_index"])
    for (user, half), group in ordered.groupby(["user_id", "half"], sort=False, observed=True):
        if len(group) < min_slices:
            continue
        cells = []
        sequences = []
        z_rows = []
        for cond, cell in group.groupby("condition", sort=False, observed=True):
            cond = str(cond)
            x = cell[model.feature_cols].to_numpy(float)
            mu = _condition_mean(model, cond)
            residual = x - mu
            cells.append(residual.mean(axis=0))
            sequences.append(residual)
            z_rows.append(_condition_z(model, cond))
        cell_mat = np.vstack(cells)
        a_u = np.mean(cell_mat, axis=0)
        centered_sequences = [s - a_u for s in sequences]

        static_row: dict[str, float | str] = {"user_id": str(user), "half": str(half)}
        for j, name in enumerate(model.feature_cols):
            static_row[f"level::{name}"] = float(a_u[j])
        z_mat = np.vstack(z_rows)
        for k in range(q):
            static_row[f"choice::opportunity_axis_{k + 1}"] = float(np.mean(z_mat[:, k]))
        rows["static"].append(static_row)

        desc = _pooled_path_descriptors(centered_sequences)
        dynamic_row: dict[str, float | str] = {"user_id": str(user), "half": str(half)}
        for kind, values in desc.items():
            for j, name in enumerate(model.feature_cols):
                dynamic_row[f"{kind}::{name}"] = float(values[j])
        rows["dynamic"].append(dynamic_row)

        hybrid_row: dict[str, float | str] = {"user_id": str(user), "half": str(half)}
        supported_mask = np.array([np.any(np.abs(z) > 0) for z in z_rows])
        if supported_mask.sum() >= min_conditions_hybrid:
            z_fit = z_mat[supported_mask]
            y_fit = cell_mat[supported_mask] - a_u
            design = np.column_stack([np.ones(len(z_fit)), z_fit])
            penalty = np.eye(q + 1) * ridge_alpha
            penalty[0, 0] = 0.0
            coef = np.linalg.solve(design.T @ design + penalty, design.T @ y_fit)[1:]
            for k in range(q):
                for j, name in enumerate(model.feature_cols):
                    hybrid_row[f"response_z{k + 1}::{name}"] = float(coef[k, j])
        rows["hybrid"].append(hybrid_row)
    return {key: pd.DataFrame(value) for key, value in rows.items()}


def varimax(loadings: np.ndarray, gamma: float = 1.0, max_iter: int = 100,
            tol: float = 1e-7) -> tuple[np.ndarray, np.ndarray]:
    """Orthogonal varimax rotation returning rotated loadings and rotation."""
    p, k = loadings.shape
    rotation = np.eye(k)
    previous = 0.0
    for _ in range(max_iter):
        lam = loadings @ rotation
        u, s, vh = np.linalg.svd(
            loadings.T @ (lam ** 3 - (gamma / p) * lam @ np.diag(np.sum(lam ** 2, axis=0)))
        )
        rotation = u @ vh
        objective = float(np.sum(s))
        if previous and objective - previous < tol * previous:
            break
        previous = objective
    return loadings @ rotation, rotation


def parallel_analysis(
    matrix: np.ndarray,
    *,
    n_iter: int = 100,
    quantile: float = 0.95,
    max_factors: int = 12,
    seed: int = 42,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Horn parallel analysis using independent column permutations."""
    x = np.asarray(matrix, float)
    observed = np.linalg.eigvalsh(np.corrcoef(x, rowvar=False))[::-1]
    rng = np.random.default_rng(seed)
    null = np.empty((n_iter, x.shape[1]))
    for b in range(n_iter):
        perm = np.column_stack([rng.permutation(x[:, j]) for j in range(x.shape[1])])
        null[b] = np.linalg.eigvalsh(np.corrcoef(perm, rowvar=False))[::-1]
    threshold = np.quantile(null, quantile, axis=0)
    k = int(min(max_factors, np.sum(observed > threshold)))
    return k, observed, threshold


def align_loading_matrices(reference: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Align candidate factors to reference by maximum absolute congruence."""
    ref = np.asarray(reference, float)
    cand = np.asarray(candidate, float)
    numer = ref.T @ cand
    denom = np.linalg.norm(ref, axis=0)[:, None] * np.linalg.norm(cand, axis=0)[None, :]
    phi = np.divide(numer, denom, out=np.zeros_like(numer), where=denom > 0)
    r_idx, c_idx = linear_sum_assignment(-np.abs(phi))
    aligned = np.zeros_like(ref)
    congruence = np.full(ref.shape[1], np.nan)
    for r, c in zip(r_idx, c_idx):
        sign = 1.0 if phi[r, c] >= 0 else -1.0
        aligned[:, r] = cand[:, c] * sign
        congruence[r] = abs(phi[r, c])
    return aligned, congruence


def _inverse_sqrt(matrix: np.ndarray, ridge: float = 0.10) -> np.ndarray:
    """Symmetric inverse square root with a relative eigenvalue floor."""
    vals, vecs = np.linalg.eigh((matrix + matrix.T) / 2.0)
    positive = vals[vals > 1e-12]
    reference = float(np.median(positive)) if len(positive) else 1.0
    floor = max(1e-8, ridge * reference)
    return (vecs * (1.0 / np.sqrt(np.maximum(vals, floor)))) @ vecs.T


def _permute_within_strata(rng: np.random.Generator, strata: np.ndarray) -> np.ndarray:
    idx = np.arange(len(strata))
    out = idx.copy()
    for value in np.unique(strata):
        members = idx[strata == value]
        out[members] = rng.permutation(members)
    return out


def fit_stable_crossview(
    early: np.ndarray,
    late: np.ndarray,
    *,
    n_permutations: int = 499,
    max_factors: int = 12,
    min_stable_share: float = 0.05,
    noise_ridge: float = 0.10,
    strata: np.ndarray | None = None,
    seed: int = 42,
    forced_factors: int | None = None,
) -> StableCrossViewResult:
    """Find dimensions that reproduce across disjoint occasions.

    The operator is the symmetric early/late cross-covariance whitened by
    half-difference noise. Factor number is selected against late-view author
    permutations; no external labels or pooled author averages are used.
    """
    e, l = np.asarray(early, float), np.asarray(late, float)
    if e.shape != l.shape or e.ndim != 2:
        raise ValueError("early and late must be same-shape 2D matrices")
    center = np.nanmean(np.vstack([e, l]), axis=0)
    scale = np.nanstd(np.vstack([e, l]), axis=0, ddof=1)
    scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
    med = np.nanmedian(np.vstack([e, l]), axis=0)
    e = (np.where(np.isfinite(e), e, med) - center) / scale
    l = (np.where(np.isfinite(l), l, med) - center) / scale
    e -= e.mean(axis=0)
    l -= l.mean(axis=0)
    n = len(e)
    noise = np.cov((e - l).T, ddof=1) / 2.0
    whitener = _inverse_sqrt(noise, ridge=noise_ridge)

    def operator_for(late_view: np.ndarray) -> np.ndarray:
        cross = (e.T @ late_view + late_view.T @ e) / (2.0 * max(n - 1, 1))
        return (whitener @ cross @ whitener + whitener @ cross.T @ whitener) / 2.0

    def spectrum(late_view: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        vals, vecs = np.linalg.eigh(operator_for(late_view))
        order = np.argsort(vals)[::-1]
        return vals[order], whitener @ vecs[:, order]

    observed, directions = spectrum(l)
    rng = np.random.default_rng(seed)
    if strata is None:
        strata = np.zeros(n, dtype=int)
    strata = np.asarray(strata)
    if n_permutations:
        null = np.empty((n_permutations, len(observed)))
        for b in range(n_permutations):
            perm = _permute_within_strata(rng, strata)
            null[b] = np.linalg.eigvalsh(operator_for(l[perm]))[::-1]
        threshold = np.quantile(null, 0.95, axis=0)
        # The strict legacy ">" gate must not be decided by last-bit
        # differences between eigensolvers (e.g. a stratum of one person).
        eps = (
            np.finfo(float).eps
            * 128
            * len(observed)
            * max(1.0, np.abs(null).max(), np.abs(observed).max())
        )
        share_check = np.maximum(observed, 0) / max(
            np.maximum(observed, 0).sum(), 1e-12
        )
        if np.any(
            (observed > 0)
            & (share_check >= min_stable_share)
            & (abs(observed - threshold) <= eps)
        ):
            rng = np.random.default_rng(seed)
            for b in range(n_permutations):
                null[b] = spectrum(l[_permute_within_strata(rng, strata)])[0]
            threshold = np.quantile(null, 0.95, axis=0)
    else:
        threshold = np.full_like(observed, np.nan)
    positive_total = np.maximum(observed, 0).sum()
    share = np.maximum(observed, 0) / max(positive_total, 1e-12)
    eligible = (observed > threshold) & (observed > 0) & (share >= min_stable_share)
    k = (min(max_factors, int(forced_factors)) if forced_factors is not None
         else min(max_factors, int(np.sum(eligible))))
    raw = directions[:, :k]
    if k:
        raw = raw / np.maximum(np.linalg.norm(raw, axis=0, keepdims=True), 1e-12)
        rotated, _ = varimax(raw)
        rotated = rotated / np.maximum(np.linalg.norm(rotated, axis=0, keepdims=True), 1e-12)
    else:
        rotated = raw
    return StableCrossViewResult(
        n_factors=k,
        directions=raw,
        rotated_directions=rotated,
        stable_eigenvalues=observed,
        permutation_thresholds=threshold,
        stable_share=share,
        center=center,
        scale=scale,
    )


def transform_stable_crossview(matrix: np.ndarray, result: StableCrossViewResult) -> np.ndarray:
    """Score rows using the frozen discovery scaling and rotated subspace."""
    x = np.asarray(matrix, float)
    med = result.center
    x = np.where(np.isfinite(x), x, med)
    return ((x - result.center) / result.scale) @ result.rotated_directions


def subspace_similarity(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    """Return projector similarity and largest principal angle in degrees."""
    if reference.shape[1] == 0 or candidate.shape[1] == 0:
        return float("nan"), float("nan")
    k = min(reference.shape[1], candidate.shape[1])
    q1, _ = np.linalg.qr(reference[:, :k])
    q2, _ = np.linalg.qr(candidate[:, :k])
    singular = np.linalg.svd(q1.T @ q2, compute_uv=False)
    similarity = float(np.sum(singular ** 2) / k)
    angle = float(np.degrees(np.arccos(np.clip(np.min(singular), -1.0, 1.0))))
    return similarity, angle
