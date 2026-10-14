#!/usr/bin/env python3
"""Same-text multichannel calibration and conditional response moments.

Public factorial anchors fix observable event units. Joint covariance handles
shared text, and a resolution matrix exposes unidentifiable combinations.
"""

from __future__ import annotations

import numpy as np


def factorial(poles):
    """Complete four-corner basis of two binary event observations."""
    p = np.asarray(poles, float)
    if p.ndim != 2 or p.shape[1] != 2 or not np.isin(p, [-1, 1]).all():
        raise ValueError("Two declared binary event poles required")
    return np.c_[np.ones(len(p)), p, p[:, 0] * p[:, 1]]


def calibrate(observations, condition, poles, shrinkage=0.1):
    """Fit endpoint factorial means and within-corner paraphrase covariance."""
    y, z = np.asarray(observations, float), np.asarray(condition, float)
    x = factorial(poles)
    if y.ndim != 2 or len(y) != len(z) or len(y) != len(x) or not np.isfinite(y).all():
        raise ValueError("Finite, aligned calibration observations required")
    if set(z) != {-1, 1} or not 0 < shrinkage <= 1:
        raise ValueError("Endpoint calibration and fixed positive shrinkage required")
    cells = []
    for c in [-1, 1]:
        xx, yy = x[z == c], y[z == c]
        if np.linalg.matrix_rank(xx) != 4 or len(xx) <= 4:
            raise ValueError("All four polarity corners with residual degrees required")
        coef = np.linalg.lstsq(xx, yy, rcond=None)[0]
        residual = yy - xx @ coef
        empirical = residual.T @ residual / (len(xx) - 4)
        # Fixed scale-relative variance floor, not a held-out tuning parameter.
        signal_scale = np.maximum(yy.var(axis=0), 1e-20)
        diagonal = np.maximum(np.diag(empirical), signal_scale * 1e-8)
        cov = (1 - shrinkage) * empirical + shrinkage * np.diag(diagonal)
        cells.append(
            {
                "condition": c,
                "offset": coef[0],
                "loading": coef[1:].T,
                "covariance": cov,
                "empirical_covariance": empirical,
                "n": len(xx),
                "residual_df": len(xx) - 4,
            }
        )
    return {"cells": cells, "shrinkage": shrinkage, "channels": y.shape[1]}


def at_condition(model, condition):
    """Interpolate the SOURCE meter; expose rather than learn from the held cell."""
    z = float(condition)
    if not np.isfinite(z) or abs(z) > 1:
        raise ValueError("Declared interpolation support [-1,1] required")
    weights = [(1 - z) / 2, (1 + z) / 2]
    return {
        field: sum(
            w * np.asarray(cell[field]) for w, cell in zip(weights, model["cells"])
        )
        for field in ["offset", "loading", "covariance"]
    }


def decoder(offset, loading, covariance, *, tolerance=1e-8):
    """GLS via whitening, allowing EXACT duplicated channels without fake precision."""
    b, l, r = (
        np.asarray(offset, float),
        np.asarray(loading, float),
        np.asarray(covariance, float),
    )
    if l.shape != (len(b), 3) or r.shape != (len(b), len(b)):
        raise ValueError("Aligned three-moment meter required")
    if not all(np.isfinite(a).all() for a in (b, l, r)):
        raise ValueError("Finite calibration required")
    scale = np.sqrt(np.diag(r))
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Each retained channel needs positive observation variance")
    correlation = (r / scale[:, None]) / scale[None, :]
    e, q = np.linalg.eigh((correlation + correlation.T) / 2)
    if e.min() < -1e-9 * max(e.max(), 1e-20):
        raise ValueError("Observation covariance must be positive semidefinite")
    keep = e > max(e.max() * 1e-12, 1e-30)
    if not keep.any():
        raise ValueError("No stochastic observation support")
    whiten = (q[:, keep] / np.sqrt(e[keep])).T / scale[None, :]
    ll = whiten @ l
    u, s, vt = np.linalg.svd(ll, full_matrices=False)
    active = s > max(s[0] * tolerance, 1e-20)
    w = (vt[active].T / s[active]) @ u[:, active].T @ whiten
    resolution = w @ l
    return {
        "weights": w,
        "offset": -w @ b,
        "resolution": resolution,
        "rank": int(active.sum()),
        "singular_values": s,
        "working_covariance": w @ r @ w.T,
        "information": ll.T @ ll,
        "observation_rank": int(keep.sum()),
    }


def build_meter(model, condition, channels, *, diagonal=False, tolerance=1e-8):
    """Choose the stated observer family, not its best held-out subset."""
    m = at_condition(model, condition)
    index = np.asarray(channels, int)
    cov = m["covariance"][np.ix_(index, index)]
    if diagonal:
        cov = np.diag(np.diag(cov))
    d = decoder(m["offset"][index], m["loading"][index], cov, tolerance=tolerance)
    return dict(
        **d,
        channels=index,
        condition=float(condition),
        diagonal=diagonal,
        raw_offset=m["offset"][index],
        loading=m["loading"][index],
        covariance=cov,
    )


def read_events(values, meter):
    """Return calibrated event-moment projections without labels or author fitting."""
    y = np.asarray(values, float)
    if not np.isfinite(y).all():
        raise ValueError("Finite current observations required")
    return (
        y[..., np.asarray(meter["channels"], int)] @ np.asarray(meter["weights"]).T
        + meter["offset"]
    )


def conditional_function(endpoint_events, *, mode="joint"):
    """One person's level and response, without replacing them by a population prior.

    Inputs: author x endpoint(-1,+1) x event x moment. Paired cross-condition event
    sampling is not assumed; report material covariance conditionally on the meter.
    """
    y = np.asarray(endpoint_events, float)
    if (
        y.ndim != 4
        or y.shape[1] != 2
        or y.shape[-1] != 3
        or y.shape[2] < 2
        or not np.isfinite(y).all()
    ):
        raise ValueError("Two endpoint event sets with three moment readings required")
    means = y.mean(axis=2)
    a = (means[:, 0] + means[:, 1]) / 2
    b = (means[:, 1] - means[:, 0]) / 2
    if mode == "level":
        b = np.zeros_like(b)
    elif mode == "slope":
        a = np.zeros_like(a)
    elif mode != "joint":
        raise ValueError("Unknown function restriction")
    errors = y - means[:, :, None, :]
    cov = np.einsum("azni,aznj->azij", errors, errors) / (y.shape[2] * (y.shape[2] - 1))
    return {
        "level": a,
        "slope": b,
        "material_mean_covariance": cov,
        "level_covariance": (cov[:, 0] + cov[:, 1]) / 4,
        "slope_covariance": (cov[:, 0] + cov[:, 1]) / 4,
        "level_slope_covariance": (cov[:, 1] - cov[:, 0]) / 4,
    }


def predict_function(fitted, condition):
    """Evaluate the frozen person function at a supported condition."""
    z = float(condition)
    if not np.isfinite(z) or abs(z) > 1:
        raise ValueError("Condition outside declared function support")
    return np.asarray(fitted["level"]) + z * np.asarray(fitted["slope"])


def joint_probabilities(moments):
    """Convert three signed moments to a valid four-event joint distribution."""
    m = np.asarray(moments, float)
    if m.shape[-1] != 3 or not np.isfinite(m).all():
        raise ValueError("Three finite joint moments required")
    poles = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    p = (1 + m @ factorial(poles)[:, 1:].T) / 4
    if np.min(p) < -1e-12:
        raise ValueError("Invalid joint moments; no clipping allowed")
    return p
