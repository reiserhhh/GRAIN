#!/usr/bin/env python3
"""Finite teacher-only linear/nonlinear comparisons with training-only geometry."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

METHODS = ("ridge", "stumps", "trees", "ridge_stumps", "ridge_trees")
ALPHAS = (1.0, 10.0, 100.0)
TREES = (100, 300)


def grouped_folds(groups, count=5, salt="s4_outer"):
    """Allocate authors by the historical SHA256 ordering; views stay together."""
    groups = np.asarray(groups, str)
    names = sorted(
        set(groups),
        key=lambda value: hashlib.sha256(f"{salt}:{value}".encode()).hexdigest(),
    )
    if count < 2 or len(names) < count:
        raise ValueError("Need at least one author per fold and two folds")
    lookup = {name: i % count for i, name in enumerate(names)}
    return np.asarray([lookup[name] for name in groups])


def finite_matrix(x):
    """Reject missing or malformed vector data rather than changing membership."""
    x = np.asarray(x, float)
    if x.ndim != 2 or min(x.shape) < 1 or not np.isfinite(x).all():
        raise ValueError("Require a finite nonempty feature matrix")
    return x


@dataclass
class Geometry:
    """Training-standardized PCA coordinates shared by all reader classes."""

    scaler: StandardScaler
    pca: PCA

    @classmethod
    def fit(cls, x, rank=64):
        """Fit neither scales nor PCA on held-out authors."""
        x = finite_matrix(x)
        if len(x) < 3 or rank < 1:
            raise ValueError("Need three training authors and positive rank")
        scaler = StandardScaler().fit(x)
        # Match the existing S4 floor, including nearly constant coordinates.
        scaler.scale_ = np.maximum(np.std(x, axis=0), 1e-6)
        pca = PCA(
            n_components=min(rank, len(x) - 1, x.shape[1]),
            svd_solver="randomized",
            random_state=731,
        ).fit(scaler.transform(x))
        return cls(scaler, pca)

    def transform(self, x):
        """Transform without adapting to the size or composition of a new batch."""
        return self.pca.transform(self.scaler.transform(finite_matrix(x)))


def tree_head(method, count, n_train, seed=731):
    """Build a bounded deterministic CPU tree model; import only when requested."""
    from lightgbm import LGBMRegressor

    depth = 1 if "stumps" in method else 3
    return LGBMRegressor(
        objective="regression",
        n_estimators=int(count),
        learning_rate=0.04,
        max_depth=depth,
        num_leaves=2 if depth == 1 else 7,
        min_child_samples=max(5, int(np.ceil(0.02 * n_train))),
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=2,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )


@dataclass
class Reader:
    """A fixed geometry and separately inspectable linear/boosted readout."""

    geometry: Geometry
    method: str
    linear: Ridge | None
    nonlinear: object | None

    def parts(self, x):
        """Return additive predicted parts, not causal or orthogonal components."""
        z = self.geometry.transform(x)
        a = self.linear.predict(z) if self.linear is not None else np.zeros(len(z))
        b = (
            self.nonlinear.booster_.predict(z, num_threads=2)
            if self.nonlinear is not None
            else np.zeros(len(z))
        )
        return np.asarray(a), np.asarray(b)

    def predict(self, x):
        """Apply the complete fixed reader to new authors."""
        a, b = self.parts(x)
        return a + b


def fit_heads(z, target, method, spec):
    """Fit all parts using only supplied training coordinates and teacher values."""
    if method not in METHODS:
        raise ValueError("Unknown readout class")
    z = finite_matrix(z)
    y = np.asarray(target, float)
    if y.shape != (len(z),) or not np.isfinite(y).all():
        raise ValueError("Require finite paired teacher values")
    linear, nonlinear = None, None
    residual = y
    if method.startswith("ridge"):
        linear = Ridge(alpha=float(spec["alpha"])).fit(z, y)
        residual = y - linear.predict(z)
    if method != "ridge":
        nonlinear = tree_head(method, spec["trees"], len(z)).fit(z, residual)
    return linear, nonlinear


def head_prediction(z, heads):
    """Read a fitted head without re-fitting or consulting validation targets."""
    linear, nonlinear = heads
    result = linear.predict(z) if linear is not None else np.zeros(len(z))
    if nonlinear is not None:
        result = result + nonlinear.booster_.predict(z, num_threads=2)
    return result


def select_readers(x, teacher, names, salt, rank=64, alphas=ALPHAS, trees=TREES):
    """Select reader hyperparameters by inner author-CV teacher MSE only."""
    x = finite_matrix(x)
    teacher = np.asarray(teacher, float)
    names = np.asarray(names, str)
    if teacher.shape != (len(x),) or names.shape != (len(x),):
        raise ValueError("Unpaired authors/teacher/features")
    if not np.isfinite(teacher).all() or not alphas or not trees:
        raise ValueError("Finite teacher and nonempty candidate grids required")
    folds = grouped_folds(names, 3, salt)
    cache = []
    ridge_losses = {a: [] for a in alphas}
    for fold in range(3):
        train, valid = folds != fold, folds == fold
        geometry = Geometry.fit(x[train], rank)
        z, zv = geometry.transform(x[train]), geometry.transform(x[valid])
        cache.append((z, zv, teacher[train], teacher[valid]))
        for alpha in alphas:
            h = fit_heads(z, teacher[train], "ridge", {"alpha": alpha})
            ridge_losses[alpha].extend((head_prediction(zv, h) - teacher[valid]) ** 2)
    alpha = min(alphas, key=lambda a: (np.mean(ridge_losses[a]), a))
    specs = {
        "ridge": {"alpha": alpha, "inner_mse": float(np.mean(ridge_losses[alpha]))}
    }
    for method in METHODS[1:]:
        errors = {n: [] for n in trees}
        for z, zv, y, yv in cache:
            for count in trees:
                heads = fit_heads(z, y, method, {"alpha": alpha, "trees": count})
                errors[count].extend((head_prediction(zv, heads) - yv) ** 2)
        count = min(trees, key=lambda n: (np.mean(errors[n]), n))
        specs[method] = {
            "alpha": alpha,
            "trees": count,
            "inner_mse": float(np.mean(errors[count])),
        }
    return specs


def fit_readers(x, teacher, specs, rank=64):
    """Refit each selected class on one shared outer-training geometry."""
    geometry = Geometry.fit(x, rank)
    z = geometry.transform(x)
    return {
        name: Reader(geometry, name, *fit_heads(z, teacher, name, spec))
        for name, spec in specs.items()
    }


def mixed_difference(predict, base, first, second):
    """Measure finite coordinate interaction, with no derivative assumptions."""
    base, first, second = map(finite_matrix, (base, first, second))
    if not (base.shape == first.shape == second.shape):
        raise ValueError("Mixed differences require equal vector shapes")
    return (
        predict(base + first + second)
        - predict(base + first)
        - predict(base + second)
        + predict(base)
    )


def paired_mse_interval(teacher, prediction, baseline, draws=1000, seed=731):
    """Bootstrap paired authors; positive delta means lower error than baseline."""
    y, p, b = [np.asarray(v, float) for v in (teacher, prediction, baseline)]
    if y.ndim != 1 or y.shape != p.shape or y.shape != b.shape or len(y) < 3:
        raise ValueError("Need at least three paired author predictions")
    if not all(np.isfinite(v).all() for v in (y, p, b)) or draws < 1:
        raise ValueError("Require finite scores and positive bootstrap count")
    delta = (y - b) ** 2 - (y - p) ** 2
    rng = np.random.default_rng(seed)
    boot = np.array(
        [delta[rng.integers(len(y), size=len(y))].mean() for _ in range(draws)]
    )
    return {
        "teacher_mse_gain": float(delta.mean()),
        "teacher_mse_gain_lower": float(np.quantile(boot, 0.025)),
        "teacher_mse_gain_upper": float(np.quantile(boot, 0.975)),
    }
