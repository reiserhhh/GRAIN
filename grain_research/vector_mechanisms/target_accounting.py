#!/usr/bin/env python3
"""Descriptive target/proxy projection and observer-aware criterion accounts."""

from __future__ import annotations

import numpy as np

from .criterion_accounting import criterion_account
from .reading_levels import affine_error_account


def projection_account(target, proxy, criterion):
    """Keep amplitude error distinct from criterion-linked non-affine error.

    This is an in-sample description of frozen outputs, not calibration for use
    on new authors. Only target and proxy determine the alignment coefficient.
    """
    result = affine_error_account(target, proxy, criterion)
    t, p, y = (np.asarray(a, float) for a in (target, proxy, criterion))
    target_r = float(np.corrcoef(t, y)[0, 1])
    proxy_r = float(np.corrcoef(p, y)[0, 1])
    retained = result["teacher_fidelity_r"] * proxy_r
    result.update(
        target_r=target_r,
        proxy_r=proxy_r,
        delta_r=proxy_r - target_r,
        aligned_contribution=retained,
        projection_identity_error=abs(
            target_r - retained - result["non_affine_contribution"]
        ),
    )
    return result


def projection_draws(target, proxy, criterion, indices):
    """Recompute the descriptive projection on each paired author resample."""
    projection_account(target, proxy, criterion)
    indices = np.asarray(indices)
    n = len(target)
    if (
        indices.ndim != 2
        or indices.shape[1] != n
        or not np.issubdtype(indices.dtype, np.integer)
        or indices.size == 0
        or indices.min() < 0
        or indices.max() >= n
    ):
        raise ValueError("Matched integer author bootstrap indices required")
    t, p, y = (np.asarray(a, float)[indices] for a in (target, proxy, criterion))
    t, p, y = (a - a.mean(1, keepdims=True) for a in (t, p, y))
    vt, vp, vy = ((a * a).mean(1) for a in (t, p, y))
    ctp, cty, cpy = ((a * b).mean(1) for a, b in ((t, p), (t, y), (p, y)))
    valid = (vt > 1e-24) & (vp > 1e-24) & (vy > 1e-24)
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = ctp / vp
        denominator = np.sqrt(vt * vy)
        target_r = cty / denominator
        proxy_r = cpy / np.sqrt(vp * vy)
        aligned = slope * cpy / denominator
        non_affine = (cty - slope * cpy) / denominator
        values = {
            "target_r": target_r,
            "proxy_r": proxy_r,
            "delta_r": proxy_r - target_r,
            "teacher_fidelity_r": ctp / np.sqrt(vt * vp),
            "alignment_slope": slope,
            "aligned_contribution": aligned,
            "non_affine_contribution": non_affine,
            "amplitude_contribution": (slope - 1) * cpy / denominator,
            "total_error_criterion_contribution": (cty - cpy) / denominator,
        }
    return {k: np.where(valid, v, np.nan) for k, v in values.items()}


def native_chain(
    native, fixed, configuration, residual, extracted, criterion, same_observer=None
):
    """Allocate signed covariance under one native-score denominator.

    The optional same-observer score splits runtime conditions from formula
    approximation; the resulting chain still reconstructs the original native.
    """
    arrays = [
        np.asarray(a, float)
        for a in (native, fixed, configuration, residual, extracted, criterion)
    ]
    if any(a.ndim != 1 or a.shape != arrays[0].shape for a in arrays):
        raise ValueError("Exactly matched author arrays required")
    native, fixed, configuration, residual, extracted, criterion = arrays
    if same_observer is None:
        parts = {"observer_and_formula_gap": native - fixed}
    else:
        observed = np.asarray(same_observer, float)
        if observed.shape != native.shape:
            raise ValueError("Same-observer author alignment required")
        parts = {
            "observer_condition_gap": native - observed,
            "same_observer_formula_gap": observed - fixed,
        }
    parts.update(
        other_fixed_components=fixed - configuration,
        fold_grain_reference=configuration - residual,
        extracted_residual=extracted,
        extraction_error=residual - extracted,
    )
    values = np.column_stack(list(parts.values()))
    account = criterion_account(native, values, criterion)
    return {**account, "names": list(parts), "values": values}
