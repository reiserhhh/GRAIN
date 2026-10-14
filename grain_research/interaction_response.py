#!/usr/bin/env python3
"""Conditional actor/partner responses, control functions and observation boundaries."""

from __future__ import annotations

import numpy as np


def designs(condition, partner, assignment=None):
    """Keep authored condition, observed partner input and assignment distinct."""
    z, u = (np.asarray(v, float) for v in (condition, partner))
    if (
        z.ndim != 1
        or z.shape != u.shape
        or not np.isfinite(z).all()
        or not np.isfinite(u).all()
    ):
        raise ValueError("Aligned finite condition and partner observations required")
    x = np.c_[np.ones(len(z)), z, u, z * u]
    if assignment is None:
        return x, None
    r = np.asarray(assignment, float)
    if r.shape != z.shape or not np.isfinite(r).all():
        raise ValueError(
            "Assignment must be separately observed, not invented from text"
        )
    return x, np.c_[np.ones(len(z)), z, r, z * r]


def hac_meat(moment, lags=4):
    """Bartlett covariance of moment sums for within-session dependent observations."""
    g = np.asarray(moment, float)
    if g.ndim != 2 or not np.isfinite(g).all() or lags < 0:
        raise ValueError("Finite moment matrix and nonnegative lag required")
    result = g.T @ g
    for lag in range(1, min(lags, len(g) - 1) + 1):
        cross = g[lag:].T @ g[:-lag]
        result += (1 - lag / (lags + 1)) * (cross + cross.T)
    return (result + result.T) / 2


def estimate_response(y, condition, partner, assignment=None, *, lags=4):
    """Estimate observational and IV/control-function response from past opportunities.

    Constant partner or absent assignment is a declared identification boundary.
    The data cannot verify exclusion of a direct assignment effect. Estimates are
    not interpreted causally unless the construction/design supplies that premise.
    """
    y = np.asarray(y, float)
    x, w = designs(condition, partner, assignment)
    if y.shape != (len(x),) or len(x) < 12 or not np.isfinite(y).all():
        raise ValueError("At least twelve complete opportunities required")
    if np.linalg.matrix_rank(x) < x.shape[1]:
        return {
            "status": "conditional_sum_only",
            "reason": "Partner variation cannot separate level/response coefficients",
            "rank": int(np.linalg.matrix_rank(x)),
            "conditional_mean": np.linalg.lstsq(x[:, :2], y, rcond=None)[0],
        }
    ols = np.linalg.lstsq(x, y, rcond=None)[0]
    oi = np.linalg.inv(x.T @ x)
    residual = y - x @ ols
    result = {
        "status": "observational_only",
        "ols": ols,
        "ols_cov": oi @ hac_meat(x * residual[:, None], lags) @ oi,
        "condition_number": float(np.linalg.cond(x)),
        "assignment_observed": assignment is not None,
    }
    if w is None:
        return result
    if np.linalg.matrix_rank(w.T @ x) < 4:
        result["instrument_status"] = "sample_rank_failure"
        return result
    bread = np.linalg.inv(w.T @ x)
    iv = bread @ w.T @ y
    iv_error = y - x @ iv
    first = np.linalg.lstsq(w, x[:, 2:], rcond=None)[0]
    remainder = x[:, 2:] - w @ first
    augmented = np.c_[x, remainder]
    cf = np.linalg.lstsq(augmented, y, rcond=None)[0]
    reduced = np.linalg.lstsq(w[:, :2], x[:, 2:], rcond=None)[0]
    sse0 = np.sum((x[:, 2:] - w[:, :2] @ reduced) ** 2, axis=0)
    sse1 = np.sum(remainder**2, axis=0)
    strength = ((sse0 - sse1) / 2) / (sse1 / (len(w) - 4))
    result.update(
        {
            "status": "identified_under_assignment_assumptions",
            "iv": iv,
            "iv_cov": bread @ hac_meat(w * iv_error[:, None], lags) @ bread.T,
            "cf": cf,
            "first_stage": first,
            "first_stage_f": strength,
            "iv_condition_number": float(np.linalg.cond(w.T @ x)),
            "iv_cf_max_difference": float(np.max(abs(cf[:4] - iv))),
        }
    )
    return result


def apply_response(fit, condition, partner, assignment=None, *, arm="ols"):
    """New-opportunity expression, actor profile and common-input evidence, no refit."""
    x, w = designs(condition, partner, assignment)
    if arm == "ols":
        beta, common = fit["ols"], np.zeros(len(x))
    elif arm in ("iv", "control_function"):
        beta = fit["iv"]
        if w is None:
            raise ValueError("New assignment required for this observation design")
        common = (x[:, 2:] - w @ fit["first_stage"]) @ fit["cf"][4:]
        if arm == "iv":
            common = np.zeros(len(x))
    else:
        raise ValueError("Unknown response arm")
    profile = x @ beta
    return {"profile": profile, "common_input": common, "response": profile + common}


def opportunity_profile(scores, accepted, query_weights=None):
    """Aggregate supplied dialogue evidence with coverage, not a trait probability.

    A vector of act-reader outputs is an observation of text behavior. Missing
    evidence remains missing; absent acts do not certify absent personality traits.
    """
    y, mask = np.asarray(scores, float), np.asarray(accepted, bool)
    if y.ndim != 2 or y.shape != mask.shape or not np.isfinite(y).all():
        raise ValueError(
            "Aligned finite event/act scores and observation mask required"
        )
    weights = (
        np.ones(len(y)) if query_weights is None else np.asarray(query_weights, float)
    )
    if (
        weights.shape != (len(y),)
        or np.any(weights < 0)
        or not np.isfinite(weights).all()
    ):
        raise ValueError("Nonnegative opportunity weights required")
    count = np.sum(mask * weights[:, None], axis=0)
    score = np.divide(
        np.sum(y * mask * weights[:, None], axis=0),
        count,
        out=np.full(y.shape[1], np.nan),
        where=count > 0,
    )
    return {"score": score, "observed_weight": count, "opportunities": len(y)}
