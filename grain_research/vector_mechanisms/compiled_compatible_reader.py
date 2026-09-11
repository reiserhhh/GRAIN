#!/usr/bin/env python3
"""Matched source teaching of native versus compiled-compatible coordinates."""

import numpy as np

from . import evidence_coordinate_reader as coord

base = coord.base
VARIANTS = ("original4_native_coordinates", "original4_compatible_coordinates")
ORIGINAL = np.arange(0, 8, 2)


def teaching(current, native, readers, common, origin, units):
    """Bind five paired observables to one source-only effective reader target.

    Inputs are [author, query, five observables, hidden] and native paired
    readers. The four action observations are values, not scalar derivatives.
    This routine is also usable after prediction freeze for evaluation targets.
    """
    current, native, readers, common, origin, units = [
        np.asarray(a, float) for a in (current, native, readers, common, origin, units)
    ]
    if (
        current.ndim != 4
        or current.shape[1:3] != (2, 5)
        or native.shape != current.shape
        or readers.shape != (len(current), 2, current.shape[-1])
        or common.shape != (current.shape[-1],)
        or origin.shape != common.shape
        or units.shape != (5,)
        or np.any(units <= 0)
        or not all(
            np.isfinite(a).all()
            for a in (current, native, readers, common, origin, units)
        )
    ):
        raise ValueError("Aligned finite five-observable paired material required")
    average, delta = current.mean(axis=1), current[:, 0] - current[:, 1]
    frame = average / units[None, :, None]
    offset = delta @ common + average @ origin
    pair = np.einsum("nqkh,nqh->nqk", native, readers)
    target = pair[:, 0] - pair[:, 1]
    difference = readers[:, 0] - readers[:, 1]
    native_coordinates = np.einsum("nkh,nh->nk", frame, difference - origin)
    compatible = (target - offset) / units
    other = offset + units * native_coordinates - target
    np.testing.assert_allclose(
        compatible, native_coordinates - other / units, atol=1e-10
    )
    return {
        "frame": frame,
        "offset": offset,
        "contrast": target,
        "native_coordinates": native_coordinates,
        "compatible_coordinates": compatible,
        "other_error": other,
    }


def fit(old_model, target):
    """One original-four solve, retaining exact old units, weights and lambda."""
    target = np.asarray(target, float)
    if target.shape != (4, 5) or not np.isfinite(target).all():
        raise ValueError("Exactly four original source values by five coordinates")
    if len(old_model["x"]) != 28 or np.any(old_model["kinds"][ORIGINAL]):
        raise ValueError("Original four positions must be values in the old28 contract")
    x, v, kinds, w = [old_model[k][ORIGINAL] for k in ("x", "v", "kinds", "weights")]
    lam = float(old_model["regularizer"])
    if not np.isfinite(w).all() or np.any(w <= 0) or not np.isfinite(lam) or lam <= 0:
        raise ValueError("Positive frozen source weights and regularizer required")
    kernel = base.gram(x, v, kinds, x, v, kinds)
    a = kernel + np.diag(lam / w**2)
    beta = np.linalg.solve(a, target)
    normal_error = np.linalg.norm(a @ beta - target) / max(
        np.linalg.norm(target), np.finfo(float).tiny
    )
    if not np.isfinite(beta).all() or normal_error > 1e-9:
        raise ValueError("Matched source coordinate solve failed")
    return {
        "x": x,
        "v": v,
        "kinds": kinds,
        "weights": w,
        "regularizer": np.asarray(lam),
        "beta": beta,
        "d0": np.zeros(5),
        "target": target,
        "fitted": kernel @ beta,
        "original_indices": ORIGINAL,
        "normal_residual_relative": np.asarray(normal_error),
        "regularized_condition": np.asarray(np.linalg.cond(a)),
        "coefficients": np.asarray(beta.size),
        "moving": np.asarray(True),
        "fixed_frame": old_model["fixed_frame"],
        "units": old_model["units"],
    }
