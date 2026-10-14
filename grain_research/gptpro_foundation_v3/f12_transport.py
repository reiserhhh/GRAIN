"""Pilot-only inverse calibration with explicit condition and range support."""

import numpy as np


def fit_transport(old, new, conditions, mode):
    """Fit paired pilot calibration; never infer a missing condition anchor."""
    r, s, z = [np.asarray(x, dtype=float) for x in (old, new, conditions)]
    if r.ndim != 1 or not (r.shape == s.shape == z.shape) or len(r) < 4:
        raise ValueError("at least four aligned pilot pairs required")
    if not all(np.isfinite(x).all() for x in (r, s, z)):
        raise ValueError("finite pilot observations required")
    if mode == "global":
        x, target = np.column_stack([np.ones(len(r)), r]), s
    elif mode == "conditional":
        x, target = np.column_stack([np.ones(len(r)), z, r, z * r]), s
    elif mode == "cubic":
        x, target = np.column_stack([np.ones(len(s)), s, s**3]), r
    else:
        raise ValueError("unknown calibration mode")
    rank = int(np.linalg.matrix_rank(x))
    if rank < x.shape[1]:
        return {
            "status": "MISSING_ANCHOR",
            "mode": mode,
            "rank": rank,
            "required_rank": x.shape[1],
            "required_anchor": "paired material varying both level and condition",
        }
    coef = np.linalg.lstsq(x, target, rcond=None)[0]
    result = {
        "status": "READY",
        "mode": mode,
        "coefficients": coef,
        "rank": rank,
        "conditions": np.unique(z),
        "new_range": [float(s.min()), float(s.max())],
    }
    if mode == "cubic":
        points = [s.min(), s.max()]
        if s.min() <= 0 <= s.max():
            points.append(0.0)
        minimum = float(np.min(coef[1] + 3 * coef[2] * np.array(points) ** 2))
        result["min_derivative_on_pilot_range"] = minimum
        if minimum <= 0:
            result["status"] = "NONMONOTONE_INVERSE"
    return result


def transport_score(model, new, conditions):
    """Return values and support; undefined/unvalidated values remain NaN."""
    s, z = np.broadcast_arrays(
        np.asarray(new, dtype=float), np.asarray(conditions, dtype=float)
    )
    output = np.full(s.shape, np.nan)
    support = np.zeros(s.shape, dtype=bool)
    if model["status"] != "READY":
        return output, support
    support = np.isfinite(s) & np.isfinite(z) & np.isin(z, model["conditions"])
    coef = model["coefficients"]
    if model["mode"] == "cubic":
        low, high = model["new_range"]
        support &= (s >= low) & (s <= high)
        output[support] = coef[0] + coef[1] * s[support] + coef[2] * s[support] ** 3
    else:
        if model["mode"] == "global":
            offset = np.full(s.shape, coef[0])
            scale = np.full(s.shape, coef[1])
        else:
            offset = coef[0] + coef[1] * z
            scale = coef[2] + coef[3] * z
        support &= np.abs(scale) > 1e-12
        # A negative affine scale is invertible; its sign must be preserved.
        output[support] = (s[support] - offset[support]) / scale[support]
    return output, support
