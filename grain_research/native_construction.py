#!/usr/bin/env python3
"""Source-frozen readers comparing native queries with frozen text representations."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ARMS = (
    "native_raw",
    "native_calibrated",
    "semantic",
    "semantic_native",
    "additive",
    "additive_native",
)
ALPHAS = (1.0, 10.0, 100.0, 1000.0)


def fixed_splits(folds):
    """Partition one author per row with official fold 0 held and fold 1 valid."""
    folds = np.asarray(folds)
    if folds.ndim != 1 or not np.isin(folds, range(5)).all():
        raise ValueError("Expected official Big5 folds 0..4")
    result = {
        "train": np.flatnonzero(folds >= 2),
        "valid": np.flatnonzero(folds == 1),
        "held": np.flatnonzero(folds == 0),
        "source": np.flatnonzero(folds != 0),
    }
    if min(len(v) for v in result.values()) < 4:
        raise ValueError("Insufficient source/validation/held support")
    return result


def mean_r(y, p):
    """Mean all criterion correlations, rejecting silent removal of a trait."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    if y.shape != p.shape or y.ndim != 2 or not np.isfinite([y, p]).all():
        raise ValueError("Finite paired criterion matrices required")
    yc, pc = y - y.mean(0), p - p.mean(0)
    denominator = np.linalg.norm(yc, axis=0) * np.linalg.norm(pc, axis=0)
    if np.any(denominator < 1e-14):
        raise ValueError("Undefined criterion correlation")
    return float(np.mean(np.sum(yc * pc, axis=0) / denominator))


class NativeConstructionFeatures:
    """Fit source-only PCA/scaling and expose the equivalent raw-block map."""

    def __init__(self, arm):
        if arm not in ARMS[1:]:
            raise ValueError("Unknown learned arm")
        self.arm = arm

    def fit(self, blocks, indices):
        """Learn feature transforms exclusively on the supplied source rows."""
        self.fit_indices_ = np.asarray(indices, int).copy()
        if len(np.unique(self.fit_indices_)) != len(self.fit_indices_):
            raise ValueError("Repeated source row")
        self.keys_ = []
        if self.arm != "native_calibrated":
            self.keys_.append("semantic")
            x = np.asarray(blocks["semantic"])[indices]
            self.pca_ = PCA(
                n_components=min(64, len(x) - 1, x.shape[1]),
                svd_solver="randomized",
                random_state=924,
            ).fit(x)
        if self.arm.startswith("additive"):
            self.keys_.extend(["grain", "affinity"])
        if "native" in self.arm:
            self.keys_.append("native")
        raw = self._raw(blocks, indices)
        if not np.isfinite(raw).all():
            raise ValueError("Nonfinite features")
        self.scaler_ = StandardScaler().fit(raw)
        return self

    def _raw(self, blocks, indices):
        """Concatenate only the predefined blocks in their frozen order."""
        return np.column_stack(
            [
                self.pca_.transform(blocks[k][indices])
                if k == "semantic"
                else blocks[k][indices]
                for k in self.keys_
            ]
        )

    def transform(self, blocks, indices):
        """Apply fixed coordinates to either one author or an arbitrary batch."""
        return self.scaler_.transform(self._raw(blocks, indices))

    def affine(self, head, blocks):
        """Export exact prediction intercept and weights in original block space."""
        coefficient = head.coef_.T / self.scaler_.scale_[:, None]
        intercept = head.intercept_ - self.scaler_.mean_ @ coefficient
        weights, start = {}, 0
        for key in self.keys_:
            width = (
                self.pca_.n_components_ if key == "semantic" else blocks[key].shape[1]
            )
            w = coefficient[start : start + width]
            start += width
            if key == "semantic":
                w = self.pca_.components_.T @ w
                intercept = intercept - self.pca_.mean_ @ w
            weights[key] = w
        if start != len(coefficient):
            raise ValueError("Incomplete affine map")
        return intercept, weights


def fit_selected(blocks, y, split, arm):
    """Select alpha on source validation, then refit on the source union only."""
    train, valid, source = (split[k] for k in ("train", "valid", "source"))
    feature = NativeConstructionFeatures(arm).fit(blocks, train)
    x, xv = (feature.transform(blocks, ix) for ix in (train, valid))
    trials = []
    for alpha in ALPHAS:
        head = Ridge(alpha=alpha, solver="lsqr").fit(x, y[train])
        trials.append(
            {
                "arm": arm,
                "alpha": alpha,
                "validation_mean_r": mean_r(y[valid], head.predict(xv)),
            }
        )
    best = max(trials, key=lambda t: t["validation_mean_r"])
    feature = NativeConstructionFeatures(arm).fit(blocks, source)
    head = Ridge(alpha=best["alpha"], solver="lsqr").fit(
        feature.transform(blocks, source), y[source]
    )
    return {"features": feature, "head": head, "chosen": best, "trials": trials}


def paired_interval(y, a, b, draws=1000, seed=930):
    """Conditional paired-author delta-r interval with explicit invalid draws."""
    if draws < 20:
        raise ValueError("At least 20 bootstrap draws required")
    rng, values, invalid = np.random.default_rng(seed), [], 0
    for _ in range(draws):
        ix = rng.integers(0, len(y), len(y))
        try:
            values.append(mean_r(y[ix], a[ix]) - mean_r(y[ix], b[ix]))
        except ValueError:
            invalid += 1
    if len(values) < 0.95 * draws:
        raise ValueError("Insufficient finite bootstrap draws")
    lower, upper = np.quantile(values, [0.025, 0.975])
    return {
        "delta_mean_r": mean_r(y, a) - mean_r(y, b),
        "lower": float(lower),
        "upper": float(upper),
        "draws": draws,
        "invalid_draws": invalid,
        "seed": seed,
        "scope": "paired held-author percentile CI conditional on fixed heads; no refit",
    }
