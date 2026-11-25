#!/usr/bin/env python3
"""Audit the incremental constraint on the last common level at a decision cutoff.

The JSON CLI takes the keyword arguments of assess(). Inputs are aligned,
equally stepped observations up to and including the cutoff, never a future
suffix. Innovation budgets and static bounds are supplied assumptions, not CIs.
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from .riccati_mobius import solve_axis_trajectory_lp


def moment_screen(previous):
    """Necessary covariance check for the unit-loading, uncorrelated-components moment model."""
    a = np.asarray(previous, dtype=float)
    if a.ndim != 2 or a.shape[1] != 2 or len(a) < 3 or not np.isfinite(a).all():
        raise ValueError("expected at least three finite paired previous levels")
    v = np.cov(a.T)
    c = float(v[0, 1])
    parts = [c, float(v[0, 0] - c), float(v[1, 1] - c)]
    return {
        "n": len(a),
        "common_sample_covariance": c,
        "implied_private_variances": parts[1:],
        "status": "unsupported_calibration"
        if min(parts) <= 0
        else "moment_signs_admissible",
    }


def compare(
    view1, view2, half_widths, rates, budgets, c_bound, threshold=0.0, calibration_id=""
):
    y, z, w = (np.asarray(v, dtype=float) for v in (view1, view2, half_widths))
    r, e = np.asarray(rates, dtype=float), np.asarray(budgets, dtype=float)
    if (
        not calibration_id
        or y.ndim != 1
        or len(y) < 2
        or y.shape != z.shape
        or w.shape != y.shape
        or r.shape != (3,)
        or e.shape != (3,)
        or not all(np.isfinite(v).all() for v in (y, z, w, r, e, c_bound, threshold))
        or np.any(w < 0)
        or np.any(e < 0)
        or c_bound <= 0
    ):
        raise ValueError("invalid calibration or unaligned/nonfinite prefix")
    mid = (y + z) / 2
    lower, upper = np.maximum(-c_bound, mid - w), np.minimum(c_bound, mid + w)
    out = {
        "calibration_id": calibration_id,
        "steps": len(y),
        "threshold": float(threshold),
        "view1": float(y[-1]),
        "view2": float(z[-1]),
        "midpoint": float(mid[-1]),
        "gap_1_minus_2": float(y[-1] - z[-1]),
        "static": None,
        "common_only": None,
        "simple_lag": None,
        "full": None,
        "private_width_gain": None,
        "gain_over_simple_lag": None,
        "vs_simple_lag": None,
        "status": "incompatible_static",
    }
    if lower[-1] <= upper[-1]:
        out["static"] = [float(lower[-1]), float(upper[-1])]
    if np.any(lower > upper):
        return out
    lo, hi = lower[0], upper[0]
    for left, right in zip(lower[1:], upper[1:]):
        a, b = sorted((r[0] * lo, r[0] * hi))
        lo, hi = max(left, a - e[0]), min(right, b + e[0])
        if lo > hi + 1e-9:
            return dict(out, status="incompatible_common")
        if lo > hi:
            lo = hi = (lo + hi) / 2
    out["common_only"] = [float(lo), float(hi)]
    # Cheap same-history comparator. Each view provides one noisy lag contrast
    # for c[t]. It discards the dependence of that contrast on the common shock.
    sl, sh = lower[0], upper[0]
    for t in range(len(y) - 1):
        for series, rate, err in zip((y, z), r[1:], e[1:]):
            g = r[0] - rate
            s = series[t + 1] - rate * series[t]
            noise = e[0] + err
            if g == 0:
                if abs(s) > noise + 1e-9:
                    return dict(out, status="incompatible_private")
            else:
                low, high = sorted(((s - noise) / g, (s + noise) / g))
                sl, sh = max(sl, low), min(sh, high)
        if sl > sh + 1e-9:
            return dict(out, status="incompatible_private")
        a, b = sorted((r[0] * sl, r[0] * sh))
        next_lo, next_hi = max(lower[t + 1], a - e[0]), min(upper[t + 1], b + e[0])
        for series, rate, err in zip((y, z), r[1:], e[1:]):
            a, b = sorted((rate * sl, rate * sh))
            s = series[t + 1] - rate * series[t]
            next_lo, next_hi = max(next_lo, a + s - err), min(next_hi, b + s + err)
        sl, sh = next_lo, next_hi
        if sl > sh + 1e-9:
            return dict(out, status="incompatible_private")
    out["simple_lag"] = [float(sl), float(sh)]
    full = solve_axis_trajectory_lp(y, z, *r, *e, w, c_bound=c_bound, endpoint_only=True)
    if full is None:
        return dict(out, status="incompatible_private")
    left, right = (float(v[-1]) for v in full)
    tol = 1e-7 * max(1.0, abs(lo), abs(hi), abs(threshold))
    if left < sl - tol or right > sh + tol or sl < lo - tol or sh > hi + tol:
        raise ArithmeticError("non-nested LP bounds")
    gain = max(0.0, (hi - lo) - (right - left))

    def side(a, b):
        return 1 if a > threshold + tol else -1 if b < threshold - tol else 0

    label = (
        "private_decision_increment"
        if side(left, right) and not side(lo, hi)
        else "private_precision_only"
        if gain > tol
        else "no_private_increment"
    )
    extra = max(0.0, sh - sl - (right - left))
    vs_lag = (
        "changes_threshold_decision"
        if side(left, right) and not side(sl, sh)
        else "narrows_only"
        if extra > tol
        else "simple_lag_sufficient"
    )
    return dict(
        out,
        full=[left, right],
        private_width_gain=gain,
        status=label,
        gain_over_simple_lag=extra,
        vs_simple_lag=vs_lag,
        direction_simple_lag=side(sl, sh),
        direction_static=side(*out["static"]),
        direction_common_only=side(lo, hi),
        direction_full=side(left, right),
    )


def assess(
    view1, view2, *, dates1=None, dates2=None, previous_levels=None, **calibration
):
    """Front door for aligned prefixes; no calibration means direct comparison only.

    previous_levels screens *moment-based* calibration only. Do not apply that
    assumption to independently measured transition laws.
    """
    a, b = (np.asarray(x, dtype=float) for x in (view1, view2))
    if (
        a.ndim != 1
        or not len(a)
        or a.shape != b.shape
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
    ):
        raise ValueError("expected finite, aligned, nonempty views")
    if (dates1 is not None or dates2 is not None) and (
        not isinstance(dates1, list)
        or dates1 != dates2
        or len(dates1) != len(a)
        or not all(isinstance(d, str) and d for d in dates1)
        or any(l >= r for l, r in itertools.pairwise(dates1))
    ):
        raise ValueError("dates must be ordered, unique and matched across views")
    screen = None if previous_levels is None else moment_screen(previous_levels)
    direct = {
        "steps": len(a),
        "view1": float(a[-1]),
        "view2": float(b[-1]),
        "midpoint": float((a[-1] + b[-1]) / 2),
        "gap_1_minus_2": float(a[-1] - b[-1]),
        "cutoff": dates1[-1] if dates1 else None,
        "moment_screen": screen,
    }
    if screen and screen["status"] == "unsupported_calibration":
        return dict(
            direct, status="unsupported_calibration", full=None, vs_simple_lag=None
        )
    if not calibration or len(a) < 2:
        return dict(direct, status="direct_only", full=None, vs_simple_lag=None)
    return direct | compare(a, b, **calibration)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    print(
        json.dumps(
            assess(**json.loads(p.parse_args().input.read_text())),
            allow_nan=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
