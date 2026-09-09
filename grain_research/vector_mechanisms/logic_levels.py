#!/usr/bin/env python3
"""Level-specific associations and coordinate-covariant function accounting.

These are descriptive, finite-sample identities, not fitted psychological laws.
Repeated views always remain grouped by author in resampling.
"""

from __future__ import annotations

import numpy as np


def _panel(x, y):
    """Validate a complete, balanced author/view/feature panel."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim != 3 or y.shape != x.shape[:2] or min(x.shape[:2]) < 2:
        raise ValueError("Balanced author x view x feature and author x view required")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Missing or nonfinite observations are not silently dropped")
    return x, y


def correlations(x, y):
    """Return column-wise Pearson r, retaining undefined constants as NaN."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim != 2 or y.shape != (len(x),):
        raise ValueError("Aligned matrix and vector required")
    xc, yc = x - x.mean(0), y - y.mean()
    denominator = np.sqrt(np.sum(xc**2, axis=0) * np.sum(yc**2))
    return np.divide(
        xc.T @ yc,
        denominator,
        out=np.full(x.shape[1], np.nan),
        where=denominator > 1e-14,
    )


def level_associations(x, y, criterion):
    """Keep pooled, between, within, transition, and criterion links distinct."""
    x, y = _panel(x, y)
    criterion = np.asarray(criterion, float)
    if criterion.shape != (len(x),) or not np.isfinite(criterion).all():
        raise ValueError("One finite criterion per author required")
    k = x.shape[2]
    xb, yb = x.mean(1), y.mean(1)
    xw, yw = x - xb[:, None], y - yb[:, None]
    xd, yd = np.diff(x, axis=1), np.diff(y, axis=1)
    # Recompute common-transition offsets in every resample.
    xd, yd = xd - xd.mean(0), yd - yd.mean(0)
    return {
        "pooled": correlations(x.reshape(-1, k), y.ravel()),
        "between": correlations(xb, yb),
        "within": correlations(xw.reshape(-1, k), yw.ravel()),
        "transition": correlations(xd.reshape(-1, k), yd.ravel()),
        "criterion": correlations(xb, criterion),
    }


def covariance_accounting(x, y):
    """Exact total = between + within covariance under equal empirical weights.

    All moments use ddof=0. Dividing the two covariance terms by the TOTAL
    standard-deviation product produces additive correlation contributions,
    not the separately standardized between/within correlations.
    """
    x, y = _panel(x, y)
    xb, yb = x.mean(1), y.mean(1)
    xc, yc = x - x.mean((0, 1)), y - y.mean()
    xw, yw = x - xb[:, None], y - yb[:, None]
    total = np.mean(xc * yc[:, :, None], axis=(0, 1))
    between = np.mean((xb - xb.mean(0)) * (yb - yb.mean())[:, None], axis=0)
    within = np.mean(xw * yw[:, :, None], axis=(0, 1))
    vx, vy = np.mean(xc**2, axis=(0, 1)), np.mean(yc**2)
    denominator = np.sqrt(vx * vy)
    terms = {}
    for name, value in (("total", total), ("between", between), ("within", within)):
        terms[name + "_covariance"] = value
        terms[name + "_r_contribution"] = np.divide(
            value, denominator, out=np.full_like(value, np.nan), where=denominator > 0
        )
    terms["closure_error"] = total - between - within
    terms["between_x_variance_fraction"] = np.divide(
        xb.var(0), vx, out=np.full_like(vx, np.nan), where=vx > 0
    )
    return terms


def infer_levels(x, y, criterion, *, bootstrap=1999, permutations=3999, seed=20260910):
    """Bootstrap and permute complete authors, retaining paired trajectories."""
    x, y = _panel(x, y)
    criterion = np.asarray(criterion, float)
    if bootstrap < 1 or permutations < 1:
        raise ValueError("Positive resample counts required")
    observed = level_associations(x, y, criterion)
    rng = np.random.default_rng(seed)
    boots = {name: [] for name in observed}
    exceed = {name: np.zeros(x.shape[2], int) for name in observed}
    valid = {name: np.zeros(x.shape[2], int) for name in observed}
    for _ in range(bootstrap):
        ix = rng.integers(0, len(x), len(x))
        sample = level_associations(x[ix], y[ix], criterion[ix])
        for name in observed:
            boots[name].append(sample[name])
    for _ in range(permutations):
        ix = rng.permutation(len(x))
        sample = level_associations(x, y[ix], criterion[ix])
        for name in observed:
            finite = np.isfinite(sample[name])
            valid[name] += finite
            exceed[name] += finite & (np.abs(sample[name]) >= np.abs(observed[name]))
    out = {}
    for name, r in observed.items():
        samples = np.asarray(boots[name])
        lo, hi = np.full_like(r, np.nan), np.full_like(r, np.nan)
        for j in range(len(r)):
            finite = samples[:, j][np.isfinite(samples[:, j])]
            if len(finite):
                lo[j], hi[j] = np.quantile(finite, (0.025, 0.975))
        p = (1 + exceed[name]) / (1 + valid[name])
        p[~np.isfinite(r)] = np.nan
        out[name] = {
            "r": r,
            "lower": lo,
            "upper": hi,
            "p": p,
            "valid_bootstraps": np.isfinite(samples).sum(0),
            "valid_permutations": valid[name],
        }
    return out


def coordinate_accounting(x, y, transform, *, alpha=100.0):
    """Compare a fixed function, subspace and penalty under X_new = X A.

    Coordinate re-expression is exact bookkeeping. Re-fitting an isotropic
    ridge in a nonorthogonal chart is a DIFFERENT statistical model unless
    its penalty is transported to A.T @ A. No held-out performance is claimed.
    """
    x, y, a = np.asarray(x, float), np.asarray(y, float), np.asarray(transform, float)
    if x.ndim != 2 or y.shape != (len(x),) or a.shape != (x.shape[1],) * 2:
        raise ValueError("Aligned full-rank chart required")
    if not all(np.isfinite(v).all() for v in (x, y, a)) or alpha <= 0:
        raise ValueError("Finite inputs and positive ridge alpha required")
    if np.linalg.matrix_rank(a) != len(a):
        raise ValueError("Chart transform must be invertible")
    x, y = x - x.mean(0), y - y.mean()
    x = x / np.where(x.std(0) > 1e-12, x.std(0), 1.0)
    xa = x @ a
    if np.linalg.matrix_rank(x) != x.shape[1] or np.dot(y, y) <= 1e-14:
        raise ValueError("Nondegenerate full-column-rank source panel required")
    beta = np.linalg.solve(x.T @ x + alpha * np.eye(x.shape[1]), x.T @ y)
    transported = np.linalg.solve(a, beta)
    covariant = np.linalg.solve(xa.T @ xa + alpha * a.T @ a, xa.T @ y)
    naive = np.linalg.solve(xa.T @ xa + alpha * np.eye(len(a)), xa.T @ y)
    fitted = x @ np.linalg.lstsq(x, y, rcond=None)[0]
    fitted_a = xa @ np.linalg.lstsq(xa, y, rcond=None)[0]
    r2 = lambda f: float(1 - np.sum((y - f) ** 2) / np.sum(y**2))
    return {
        "rank": int(np.linalg.matrix_rank(x)),
        "in_sample_projection_r2": r2(fitted),
        "transformed_projection_r2": r2(fitted_a),
        "projection_max_error": float(np.max(np.abs(fitted_a - fitted))),
        "transported_function_max_error": float(
            np.max(np.abs(xa @ transported - x @ beta))
        ),
        "covariant_refit_max_error": float(np.max(np.abs(xa @ covariant - x @ beta))),
        "naive_refit_rms_difference": float(
            np.sqrt(np.mean((xa @ naive - x @ beta) ** 2))
        ),
        "max_coordinate_abs_r_before": float(np.max(np.abs(correlations(x, y)))),
        "max_coordinate_abs_r_after": float(np.max(np.abs(correlations(xa, y)))),
    }


def endpoint_accounting(minus, baseline, plus):
    """Separate finite odd/even response in probability and exact binary log-odds.

    Even response in probability can be a sigmoid-coordinate effect, not an
    internal nonlinear mechanism. Every input must be an unclipped probability.
    """
    arrays = [np.asarray(v, float) for v in (minus, baseline, plus)]
    if (
        arrays[0].ndim != 1
        or len(arrays[0]) < 2
        or any(v.shape != arrays[0].shape for v in arrays)
    ):
        raise ValueError("Aligned author endpoint vectors required")
    if any(not np.isfinite(v).all() or np.any((v <= 0) | (v >= 1)) for v in arrays):
        raise ValueError("Strictly interior native probabilities required, no clipping")
    rows = []
    for coordinate in ("probability", "log_odds"):
        m, b, p = (
            arrays
            if coordinate == "probability"
            else [np.log(v) - np.log1p(-v) for v in arrays]
        )
        odd, even = (p - m) / 2, (p + m) / 2 - b
        energy = np.mean(odd**2)
        rows.append(
            {
                "coordinate": coordinate,
                "authors": len(m),
                "mean_odd": float(odd.mean()),
                "mean_abs_odd": float(np.abs(odd).mean()),
                "sd_odd": float(odd.std()),
                "negative_odd_count": int(np.sum(odd < 0)),
                "common_odd_energy": float(odd.mean() ** 2),
                "heterogeneous_odd_energy": float(odd.var()),
                "common_odd_energy_fraction": float(odd.mean() ** 2 / energy)
                if energy
                else None,
                "mean_even": float(even.mean()),
                "mean_abs_even": float(np.abs(even).mean()),
                "max_abs_even": float(np.max(np.abs(even))),
                "odd_even_endpoint_replay_error": float(
                    max(
                        np.max(np.abs(b + even + odd - p)),
                        np.max(np.abs(b + even - odd - m)),
                    )
                ),
            }
        )
    return rows


def residual_pullback_example(*, points=64):
    """Counterexample to dropping the query-readable residual from a pullback.

    G(t)=t, H(t)=(t,t^2), D=(1,0), F(H)=H1*H2. A fixed fitted D
    captures only one differential component; quadrature verifies both terms.
    """
    z, weights = np.polynomial.legendre.leggauss(points)
    t, weights = (z + 1) / 2, weights / 2
    represented = np.dot(weights, t**2)
    residual = np.dot(weights, 2 * t**2)
    return {
        "endpoint_change": 1.0,
        "represented_work": float(represented),
        "residual_work": float(residual),
        "sum_error": float(abs(represented + residual - 1)),
        "error_if_residual_deleted": float(abs(represented - 1)),
    }
