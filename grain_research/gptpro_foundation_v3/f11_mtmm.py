"""Crossed trait/reader/occasion covariance, moment fitting and Gaussian REML."""

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

CELLS = [(k, r, t) for k in range(3) for r in range(3) for t in range(2)]


def designs(conditional=False):
    """Return trait, reader and occasion designs for the declared 18 cells."""
    f, m, c = np.zeros((18, 3)), np.zeros((18, 3)), np.zeros((18, 2))
    for i, (k, r, t) in enumerate(CELLS):
        f[i, k] = 1 + 0.3 * (r - 1) * (2 * t - 1) if conditional else 1
        m[i, r], c[i, t] = 1, 1
    return f, m, c


def mtmm_covariance(
    sigma, method=0.25, event=0.0, error=0.25, actor=0.0, conditional=False
):
    """Construct the actual covariance of f, author-reader, event and error terms."""
    sigma = np.asarray(sigma, dtype=float)
    if (
        sigma.shape != (3, 3)
        or not np.allclose(sigma, sigma.T)
        or not np.isfinite(sigma).all()
    ):
        raise ValueError("finite symmetric three-trait covariance required")
    if np.linalg.eigvalsh(sigma).min() < -1e-10 or min(method, event, error, actor) < 0:
        raise ValueError("negative variance")
    f, m, c = designs(conditional)
    return (
        f @ sigma @ f.T
        + method * (m @ m.T)
        + event * (c @ c.T)
        + error * np.eye(18)
        + actor * np.ones((18, 18))
    )


def moment_fit(source, conditional=False, include_event=True):
    """Least-squares covariance components; retain raw estimates before PSD repair."""
    f, m, c = designs(conditional)
    basis, pairs = [], []
    for i in range(3):
        for j in range(i + 1):
            s = np.zeros((3, 3))
            s[i, j] = s[j, i] = 1
            basis.append(f @ s @ f.T)
            pairs.append((i, j))
    basis += [m @ m.T, np.eye(18)]
    if include_event:
        basis.append(c @ c.T)
    empirical = np.cov(np.asarray(source), rowvar=False)
    design = np.column_stack([b.ravel() for b in basis])
    coeff = np.linalg.lstsq(design, empirical.ravel(), rcond=None)[0]
    sigma = np.zeros((3, 3))
    for value, (i, j) in zip(coeff[:6], pairs, strict=True):
        sigma[i, j] = sigma[j, i] = value
    values, vectors = np.linalg.eigh(sigma)
    repaired = (vectors * np.maximum(values, 1e-8)) @ vectors.T
    return {
        "sigma": repaired,
        "raw_sigma": sigma,
        "method": max(float(coeff[6]), 1e-8),
        "error": max(float(coeff[7]), 1e-8),
        "event": max(float(coeff[8]), 1e-8) if include_event else 0.0,
        "raw_components": coeff,
        "design_rank": int(np.linalg.matrix_rank(design)),
        "mean": np.mean(source, axis=0),
    }


def unpack(theta, include_event=True):
    """Cholesky trait covariance plus log-positive nuisance variances."""
    l = np.zeros((3, 3))
    indices = np.tril_indices(3)
    l[indices] = theta[:6]
    l[np.diag_indices(3)] = np.exp(l.diagonal())
    return (
        l @ l.T,
        float(np.exp(theta[6])),
        float(np.exp(theta[7])),
        float(np.exp(theta[8])) if include_event else 0.0,
    )


def fit_mtmm(source, conditional=False, include_event=True):
    """Fit source covariance by REML with an unknown fixed 18-cell mean vector."""
    y = np.asarray(source, dtype=float)
    if y.ndim != 2 or y.shape[1] != 18 or len(y) < 20 or not np.isfinite(y).all():
        raise ValueError("finite source authors x18 matrix, n>=20 required")
    initial = moment_fit(y, conditional, include_event)
    l = np.linalg.cholesky(initial["sigma"])
    l[np.diag_indices(3)] = np.log(l.diagonal())
    theta = np.r_[
        l[np.tril_indices(3)],
        np.log(
            [initial["method"], initial["error"]]
            + ([initial["event"]] if include_event else [])
        ),
    ]
    centered = y - y.mean(axis=0)
    scatter = centered.T @ centered

    def objective(parameters):
        sigma, method, error, event = unpack(parameters, include_event)
        v = mtmm_covariance(sigma, method, event, error, conditional=conditional)
        chol = cho_factor(v, lower=True)
        return 0.5 * (
            (len(y) - 1) * 2 * np.log(np.diag(chol[0])).sum()
            + np.trace(cho_solve(chol, scatter))
        )

    bounds = [(-12, 8) if i in (0, 2, 5) else (-20, 20) for i in range(6)] + [
        (-18, 8)
    ] * (len(theta) - 6)
    result = minimize(
        objective,
        theta,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-6},
    )
    sigma, method, error, event = unpack(result.x, include_event)
    return {
        "sigma": sigma,
        "method": method,
        "error": error,
        "event": event,
        "mean": y.mean(axis=0),
        "success": bool(result.success),
        "message": str(result.message),
        "objective": float(result.fun),
        "gradient_max": float(np.max(np.abs(result.jac))),
        "iterations": int(result.nit),
        "theta": result.x,
    }


def predict_traits(model, observations, conditional=False, cells=None):
    """BLUP for new authors using source-only parameters and available cells."""
    selected = np.arange(18) if cells is None else np.asarray(cells, dtype=int)
    f, _, _ = designs(conditional)
    v = mtmm_covariance(
        model["sigma"],
        model["method"],
        model["event"],
        model["error"],
        conditional=conditional,
    )
    cross = model["sigma"] @ f[selected].T
    gain = np.linalg.solve(v[np.ix_(selected, selected)], cross.T).T
    scores = (np.asarray(observations)[:, selected] - model["mean"][selected]) @ gain.T
    variance = model["sigma"] - gain @ cross.T
    return scores, variance
