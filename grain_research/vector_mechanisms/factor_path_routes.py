#!/usr/bin/env python3
"""Finite, author-held comparisons of independent GRAIN factors and query paths."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data_views import stable_key
from .factor_catalog import association

ARMS = ("grain", "path", "joint", "interaction")
ALPHAS = (1.0, 10.0, 100.0, 1000.0)
SEED = 20260910


def historical_overlap(raw_reference_ids, pseudonyms, *, seed):
    """Compare author identities in the source cache's exact pseudonym namespace."""
    encoded = {stable_key(str(value), seed)[:24] for value in raw_reference_ids}
    return encoded.intersection(str(value) for value in pseudonyms)


def fold_diagnostics(y, prediction, folds, *, binary=False):
    """Expose pooled-CV intercept effects without changing fitted predictions."""
    y, p, folds = np.asarray(y, float), np.asarray(prediction, float), np.asarray(folds)
    if y.ndim != 1 or y.shape != p.shape or folds.shape != y.shape:
        raise ValueError("Aligned out-of-fold vectors required")
    if len(set(folds)) < 2 or not np.isfinite([y, p]).all():
        raise ValueError("Finite values and multiple folds required")
    baseline, yc, pc, aucs = np.empty_like(y), y.copy(), p.copy(), []
    for fold in np.unique(folds):
        test = folds == fold
        baseline[test] = y[~test].mean()
        yc[test] -= y[test].mean()
        pc[test] -= p[test].mean()
        if binary and np.unique(y[test]).size == 2:
            aucs.append(roc_auc_score(y[test], p[test]))
    return {
        "within_fold_centered_r": association(pc, yc)["pearson_r"],
        "training_mean_pooled_r": association(baseline, y)["pearson_r"],
        "training_mean_mse": float(np.mean((baseline - y) ** 2)),
        "mse_gain_vs_training_mean": float(np.mean((baseline - y) ** 2 - (p - y) ** 2)),
        "mean_within_fold_auc": float(np.mean(aucs)) if aucs else None,
        "auc_valid_folds": len(aucs),
        "minority_count": int(min(np.sum(y == 0), np.sum(y == 1))) if binary else None,
    }


def design_matrix(grain, path, arm):
    """Construct fixed columns; no data-fitted preprocessing occurs here."""
    g, p = np.asarray(grain, float), np.asarray(path, float)
    if g.ndim != 2 or p.ndim != 2 or len(g) != len(p):
        raise ValueError("Aligned author matrices required")
    if not np.isfinite(g).all() or not np.isfinite(p).all():
        raise ValueError("Nonfinite author features")
    if arm == "grain":
        return g
    if arm == "path":
        return p
    if arm == "joint":
        return np.column_stack((g, p))
    if arm == "interaction":
        return np.column_stack(
            (g, p, (g[:, :, None] * p[:, None, :]).reshape(len(g), -1))
        )
    raise ValueError(f"Unknown arm: {arm}")


def nested_predictions(x, y, authors, *, seed=SEED):
    """Five-fold author-held Ridge with train-only inner selection and scaling."""
    x, y, authors = np.asarray(x, float), np.asarray(y, float), np.asarray(authors)
    if len(set(authors)) != len(authors):
        raise ValueError("Aggregate repeated views first; one row per author required")
    if x.ndim != 2 or y.shape != (len(x),) or len(authors) != len(x) or len(x) < 15:
        raise ValueError("At least 15 aligned authors required")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Finite inputs required")
    pred, folds, chosen = np.zeros(len(y)), np.zeros(len(y), int), []
    for fold, (train, test) in enumerate(
        KFold(5, shuffle=True, random_state=seed).split(x)
    ):
        losses = []
        inner = list(KFold(3, shuffle=True, random_state=seed + fold + 1).split(train))
        for alpha in ALPHAS:
            errors = []
            for a, b in inner:
                model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
                model.fit(x[train[a]], y[train[a]])
                errors.extend((model.predict(x[train[b]]) - y[train[b]]) ** 2)
            losses.append(float(np.mean(errors)))
        alpha = ALPHAS[int(np.argmin(losses))]
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(
            x[train], y[train]
        )
        pred[test], folds[test] = model.predict(x[test]), fold
        chosen.append(
            {"fold": fold, "alpha": alpha, "train_n": len(train), "test_n": len(test)}
        )
    return pred, folds, chosen


def metrics(y, prediction, *, binary=False):
    """Return association and error metrics without calibration or clipping."""
    y, p = np.asarray(y, float), np.asarray(prediction, float)
    return association(p, y, binary=binary) | {
        "mae": float(np.mean(np.abs(p - y))),
        "rmse": float(np.sqrt(np.mean((p - y) ** 2))),
    }


def paired_delta(y, a, b, *, binary=False, draws=1000, seed=SEED):
    """Bootstrap paired author predictions, conditional on the fitted CV outputs."""
    y, a, b = (np.asarray(v, float) for v in (y, a, b))
    if y.ndim != 1 or a.shape != y.shape or b.shape != y.shape:
        raise ValueError("Aligned paired predictions required")
    if not np.isfinite([y, a, b]).all():
        raise ValueError("Finite paired predictions required")

    def statistic(yy, aa, bb):
        if min(np.ptp(yy), np.ptp(aa), np.ptp(bb)) < 1e-14:
            return None
        out = [np.corrcoef(yy, aa)[0, 1] - np.corrcoef(yy, bb)[0, 1]]
        if binary:
            out.append(roc_auc_score(yy, aa) - roc_auc_score(yy, bb))
        return out

    rng, values = np.random.default_rng(seed), []
    observed = statistic(y, a, b)
    for _ in range(draws):
        ix = rng.integers(0, len(y), len(y))
        row = statistic(y[ix], a[ix], b[ix])
        if row is not None:
            values.append(row)
    result = {"bootstrap_valid": len(values)}
    for k, name in enumerate(("delta_r", "delta_auc") if binary else ("delta_r",)):
        result[name] = observed[k] if observed else None
        result[name + "_lower"] = (
            float(np.quantile(values, 0.025, axis=0)[k]) if values else None
        )
        result[name + "_upper"] = (
            float(np.quantile(values, 0.975, axis=0)[k]) if values else None
        )
    return result


def aligned_directions(grain, states, *, alpha=100.0, seed=SEED):
    """Map standardized source coordinates to states; norm-match a shuffled null."""
    g, h = np.asarray(grain, float), np.asarray(states, float)
    if g.ndim != 2 or h.ndim != 2 or len(g) != len(h) or len(g) < 10:
        raise ValueError("Aligned source matrices required")
    if not np.isfinite(g).all() or not np.isfinite(h).all() or alpha <= 0:
        raise ValueError("Finite data and positive ridge required")
    center, scale = g.mean(0), g.std(0)
    if np.any(scale < 1e-12):
        raise ValueError("Constant GRAIN factor")
    x, y = (g - center) / scale, h - h.mean(0)
    gram = x.T @ x + alpha * np.eye(x.shape[1])
    real = np.linalg.solve(gram, x.T @ y)
    perm = np.random.default_rng(seed).permutation(len(x))
    null = np.linalg.solve(gram, x[perm].T @ y)
    rn, nn = np.linalg.norm(real, axis=1), np.linalg.norm(null, axis=1)
    if np.any(nn < 1e-14) or np.any(rn < 1e-14):
        raise ValueError("Degenerate alignment direction")
    null *= (rn / nn)[:, None]
    return real, null, center, scale


def split_direction(direction, p_shape, q_shape):
    """Restore a flattened query-token matrix and disjoint final-token vector."""
    d = np.asarray(direction)
    n = int(np.prod(p_shape))
    if d.ndim != 1 or len(d) != n + int(np.prod(q_shape)):
        raise ValueError("Direction and P/Q dimensions disagree")
    return d[:n].reshape(p_shape), d[n:].reshape(q_shape)
