"""R12 packet 28: reliability, context diversity and informative coverage in the DECLARED scalar world.

World (frozen F04 excitation packet): y_ut = a_u + b_u z_t + e_ut, a ~ N(0, 1), b ~ N(.5, .5^2), e ~ N(0, .5^2) iid.
Readings are simulated readings, not measured language tokens.

Per-half OLS slope (with intercept): b_hat = b + sum_t zc_t e_t / Szz, so Var(b_hat | b) = sigma2 / Szz exactly,
Szz = centered sum of squares of that half's z. For independent half errors and random b:
    half reliability      rho_AB = Var(b) / sqrt((Var(b) + v_A)(Var(b) + v_B)),   v_h = sigma2 / Szz_h
    full-slope reliability rho_full = Var(b) / (Var(b) + sigma2 / Szz_full)
When v_A = v_B = v and the halves share the same z mean (so Szz_full = Szz_A + Szz_B = 2 Szz_h), rho_full equals the
Spearman-Brown value 2 rho / (1 + rho). K = 1 gives Szz = 0: the slope is unidentified (rank-1 design) and is refused,
not scored as reliability 0 or by a pseudoinverse. Bayes posterior MSE under the source prior is a separate quantity.
"""
from __future__ import annotations

import numpy as np

VAR_A, MEAN_B, SD_B, SIGMA = 1.0, 0.5, 0.5, 0.5
VAR_B, SIGMA2 = SD_B ** 2, SIGMA ** 2


class Unidentified(ValueError):
    """Slope not identified (Szz = 0)."""


def design_z(n, k):
    """Old F04 design: k conditions evenly over [-1, 1], cycled to n readings (np.resize). K = 1 refused."""
    if k < 2:
        raise Unidentified("K=1: single condition, Szz=0; slope unidentified (refused, not reliability 0)")
    if n % 2 or (n // 2) % k:
        raise ValueError("block halves must each contain every condition equally often")
    return np.resize(np.linspace(-1.0, 1.0, k), n)


def halves(n):
    """Balanced disjoint block halves [0, n/2) and [n/2, n)."""
    return np.arange(0, n // 2), np.arange(n // 2, n)


def szz(z):
    zc = np.asarray(z, float) - np.mean(z)
    return float(zc @ zc)


def information(z, sigma2=SIGMA2):
    """Fisher information of (a, b) for one half: X'X / sigma2 with X = [1, z]; rank and smallest positive eigenvalue."""
    X = np.column_stack([np.ones(len(z)), z])
    w = np.linalg.eigvalsh(X.T @ X / sigma2)
    keep = w > w.max() * 1e-12
    return {"rank": int(keep.sum()), "min_positive_eig": float(w[keep].min()), "max_eig": float(w.max())}


def slope_var(szz_h, sigma2=SIGMA2):
    if szz_h <= 0:
        raise Unidentified("Szz=0")
    return sigma2 / szz_h


def half_reliability(vA, vB, var_b=VAR_B):
    return var_b / np.sqrt((var_b + vA) * (var_b + vB))


def full_reliability(szz_full, var_b=VAR_B, sigma2=SIGMA2):
    return var_b / (var_b + slope_var(szz_full, sigma2))


def spearman_brown(r):
    return 2 * r / (1 + r)


def posterior_var_b(z, var_a=VAR_A, var_b=VAR_B, sigma2=SIGMA2):
    """Bayes posterior variance (= Bayes MSE) of b under the source prior; NOT an OLS repeat reliability."""
    X = np.column_stack([np.ones(len(z)), z])
    return float(np.linalg.inv(np.diag([1 / var_a, 1 / var_b]) + X.T @ X / sigma2)[1, 1])


def equal_szz(z, target):
    """Diagnostic: rescale a balanced design about its mean so EACH half has centered SS = target.
    Changes the coordinate / support of z; it is an algebraic diagnostic, not extra empirical contexts."""
    hA, hB = halves(len(z))
    sA, sB = szz(z[hA]), szz(z[hB])
    if abs(sA - sB) > 1e-12 * max(sA, 1.0):
        raise ValueError("halves not balanced")
    m = z.mean()
    return m + (z - m) * np.sqrt(target / sA)


def ols_slopes(Y, z):
    """Row-wise OLS slope with intercept."""
    zc = z - z.mean()
    return ((Y - Y.mean(1, keepdims=True)) * zc).sum(1) / (zc @ zc)


def slope_weights(z):
    zc = z - z.mean()
    return zc / (zc @ zc)


def corr_rows(A, B):
    """Correlation along the last axis (vectorised over bootstrap draws)."""
    Ac = A - A.mean(-1, keepdims=True); Bc = B - B.mean(-1, keepdims=True)
    return (Ac * Bc).sum(-1) / np.sqrt((Ac ** 2).sum(-1) * (Bc ** 2).sum(-1))


def stratified_indices(n_per_stratum, n_strata, draws, rng):
    """Paired cluster bootstrap: resample whole authors WITHIN each seed stratum; the same indices are applied to every
    design (paired). Returns (draws, n_strata * n_per_stratum) indices into the stacked author axis."""
    parts = [s * n_per_stratum + rng.integers(0, n_per_stratum, (draws, n_per_stratum)) for s in range(n_strata)]
    return np.concatenate(parts, axis=1)


def divided_differences(x, f):
    """First and second divided differences on grid x (e.g. log2 readings or log2 K).
    D_i = (f_{i+1} - f_i) / (x_{i+1} - x_i);  C_i = (D_{i+1} - D_i) / (x_{i+2} - x_i)."""
    x = np.asarray(x, float); f = np.asarray(f, float)
    D = np.diff(f, axis=-1) / np.diff(x)
    C = np.diff(D, axis=-1) / (x[2:] - x[:-2])
    return D, C
