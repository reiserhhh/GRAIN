"""Validated covariance propagation and finite-mixture uncertainty accounting."""

import numpy as np


def validate_covariance(covariance):
    """Reject invalid covariance instead of silently clipping impossible changes."""
    c = np.asarray(covariance, dtype=float)
    if c.ndim != 2 or c.shape[0] != c.shape[1] or not np.isfinite(c).all():
        raise ValueError("finite square covariance required")
    scale = max(1.0, float(np.max(np.abs(c))))
    if not np.allclose(c, c.T, atol=1e-12 * scale, rtol=1e-10):
        raise ValueError("covariance must be symmetric")
    c = (c + c.T) / 2
    if np.linalg.eigvalsh(c).min() < -1e-10 * scale:
        raise ValueError("covariance is not positive semidefinite")
    return c


def change_interval(mean_pair, covariance, z=1.96):
    """Return paired change/MDC under the declared joint model, not clinical MCID."""
    mean = np.asarray(mean_pair, dtype=float)
    c = validate_covariance(covariance)
    if (
        mean.shape != (2,)
        or c.shape != (2, 2)
        or not np.isfinite(mean).all()
        or not np.isfinite(z)
        or z <= 0
    ):
        raise ValueError("two finite means,2x2 covariance and positive z required")
    direction = np.array([-1.0, 1.0])
    variance = max(0.0, float(direction @ c @ direction))
    delta = float(direction @ mean)
    half = float(z * np.sqrt(variance))
    return {
        "change": delta,
        "variance": variance,
        "lower": delta - half,
        "upper": delta + half,
        "mdc_model": half,
    }


def mixture_components(means, covariances):
    """Exact equal-weight finite mixture: E[V|fit]+Var[E|fit], ddof zero."""
    m = np.asarray(means, dtype=float)
    v = np.asarray(covariances, dtype=float)
    if (
        m.ndim != 2
        or len(m) == 0
        or not np.isfinite(m).all()
        or v.shape != (len(m), m.shape[1], m.shape[1])
    ):
        raise ValueError(
            "aligned finite source-fit means and covariance matrices required"
        )
    for matrix in v:
        validate_covariance(matrix)
    center = m.mean(axis=0)
    between = (m - center).T @ (m - center) / len(m)
    within = v.mean(axis=0)
    return {
        "mean": center,
        "within": within,
        "between": between,
        "total": within + between,
    }
