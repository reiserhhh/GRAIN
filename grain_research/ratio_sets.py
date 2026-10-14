"""Packet 33: set-valued inference for a ratio of regression coefficients (Fieller test inversion).

For scalar fixed-design OLS readouts a = alpha_hat, b = beta_hat with estimated covariance V = s^2 (X'X)^-1 (residual df),
under normal homoscedastic independent errors the two-sided level-alpha test of H0: alpha - r beta = 0 is inverted:
    (a - r b)^2 <= t^2 (Vaa - 2 r Vab + r^2 Vbb),     t = t_{df, 1 - alpha/2}
i.e.  A r^2 + B r + C <= 0  with  A = b^2 - t^2 Vbb,  B = -2 (a b - t^2 Vab),  C = a^2 - t^2 Vaa.
The target is the ratio of population coefficients, not E(a/b). The set is a union of closed intervals on the extended
real line: bounded, point, ray, two rays, whole line or empty. Infinities are explicit; no grid, no clipping.
Standard mathematics (Fieller 1954; von Luxburg & Franz 2007); not a new theorem.
"""
from __future__ import annotations

import math

import numpy as np

INF = math.inf
REL_TOL = 1e-12


def _norm(intervals):
    iv = sorted((float(lo), float(hi)) for lo, hi in intervals if lo <= hi)
    out = []
    for lo, hi in iv:
        if out and lo <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def fieller(a, b, Vaa, Vab, Vbb, t):
    """Fieller set for r = alpha/beta as a normalized list of (lo, hi) intervals (lo may be -inf, hi +inf)."""
    t2 = t * t
    A = b * b - t2 * Vbb; B = -2.0 * (a * b - t2 * Vab); C = a * a - t2 * Vaa
    scaleA = max(b * b, t2 * abs(Vbb), 1e-300)
    if abs(A) <= REL_TOL * scaleA:                       # linear (boundary) case
        scaleB = max(abs(a * b), t2 * abs(Vab), 1e-300)
        if abs(B) <= REL_TOL * scaleB:
            return [(-INF, INF)] if C <= REL_TOL * max(a * a, t2 * abs(Vaa), 1e-300) else []
        x = -C / B
        return _norm([(-INF, x)] if B > 0 else [(x, INF)])
    disc = B * B - 4 * A * C
    tol_d = REL_TOL * max(B * B, abs(4 * A * C), 1e-300)
    if disc < -tol_d:
        return [] if A > 0 else [(-INF, INF)]
    sq = math.sqrt(max(disc, 0.0))
    q = -0.5 * (B + math.copysign(sq, B if B != 0 else 1.0))
    if q == 0.0:
        r1 = r2 = -B / (2 * A)
    else:
        r1, r2 = sorted((q / A, C / q))
    if A > 0:
        return _norm([(r1, r2)])
    return _norm([(-INF, r1), (r2, INF)]) if r1 < r2 else [(-INF, INF)]


def in_fieller(r, a, b, Vaa, Vab, Vbb, t, tol=1e-9):
    """Direct test-inversion membership (independent of the root algebra)."""
    lhs = (a - r * b) ** 2; rhs = t * t * (Vaa - 2 * r * Vab + r * r * Vbb)
    return lhs <= rhs + tol * max(1.0, abs(lhs), abs(rhs))


def map_affine(U, c, o, tol=1e-15):
    """Map an r-set to theta through the FIXED calibration r = c theta + o. Negative c reverses endpoints (valid);
    c = 0 is a separate case: the readout does not depend on theta, so the theta-set is the whole line if o is in U,
    else empty (flag 'c_zero')."""
    if abs(c) <= tol:
        return ([(-INF, INF)] if contains(U, o) else []), "c_zero"
    out = []
    for lo, hi in U:
        x, y = (lo - o) / c, (hi - o) / c
        out.append((min(x, y), max(x, y)))
    return _norm(out), "c_negative" if c < 0 else "c_positive"


def intersect(U, W):
    out = []
    for lo1, hi1 in U:
        for lo2, hi2 in W:
            lo, hi = max(lo1, lo2), min(hi1, hi2)
            if lo <= hi:
                out.append((lo, hi))
    return _norm(out)


def contains(U, x):
    return any(lo <= x <= hi for lo, hi in U)


def classify(U):
    if not U:
        return "empty"
    if len(U) == 1:
        lo, hi = U[0]
        if lo == -INF and hi == INF:
            return "whole_line"
        if lo == -INF or hi == INF:
            return "ray"
        return "point" if lo == hi else "bounded"
    if len(U) == 2 and U[0][0] == -INF and U[1][1] == INF:
        return "two_rays"
    return "bounded_union" if all(math.isfinite(lo) and math.isfinite(hi) for lo, hi in U) else "mixed_union"


def width(U):
    return sum(hi - lo for lo, hi in U) if U and all(math.isfinite(lo) and math.isfinite(hi) for lo, hi in U) else (INF if U else 0.0)


def to_json(U):
    enc = lambda v: "inf" if v == INF else ("-inf" if v == -INF else v)
    return [[enc(lo), enc(hi)] for lo, hi in U]
