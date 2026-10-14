"""R13 packet 30: freedom-dependent variance components from crossed author x condition x occasion x reader observations.

Design (balanced): y[a, c, o, r], authors a crossed with conditions c (known centered intensities z_c), occasions o nested in
(a, c) (separate texts), readers r crossed with every text. Random-effects components estimated by expected mean squares
(ANOVA) from observations only; the fitter never sees generating values.

Components: e (reader x text residual), O (text / occasion within author x condition), ACR, W (author x condition remainder
after the author x linear-z slope), SLOPE (variance of author slopes on z), AR, CR, A (author level), R (reader).
EMS (nA, nC, nO, nR; Szz = sum z_c^2):
  MS_e     = e
  MS_O     = e + nR O
  MS_ACR   = e + nO ACR
  MS_ACrem = e + nR O + nO ACR + nO nR W
  MS_Alin  = e + nR O + nO ACR + nO nR W + nO nR Szz SLOPE
  MS_AR    = e + nO ACR + nC nO AR
  MS_CR    = e + nO ACR + nA nO CR
  MS_A     = e + nR O + nO ACR + nO nR W + nC nO AR + nC nO nR A
  MS_R     = e + nO ACR + nC nO AR + nA nO CR + nA nC nO R
With nO = 1 the rows MS_e and MS_O have zero df: e, O, ACR and W are then aliased and must be flagged UNIDENTIFIED.
"""
from __future__ import annotations

import numpy as np

COMPONENTS = ("e", "O", "ACR", "W", "SLOPE", "AR", "CR", "A", "R")
LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)
TOL = 1e-9


def mean_squares(y, z):
    """Balanced ANOVA mean squares and df from observations y (nA, nC, nO, nR) and known centered z (nC,). No truth."""
    nA, nC, nO, nR = y.shape
    z = np.asarray(z, float); assert abs(z.sum()) < 1e-12
    Szz = float(z @ z)
    m = y.mean(); mA = y.mean((1, 2, 3)); mC = y.mean((0, 2, 3)); mR = y.mean((0, 1, 2))
    mAC = y.mean((2, 3)); mAR = y.mean((1, 2)); mCR = y.mean((0, 2)); mACR = y.mean(2); mACO = y.mean(3)
    eA = mA - m; eC = mC - m; eR = mR - m
    eAC = mAC - mA[:, None] - mC[None, :] + m
    eAR = mAR - mA[:, None] - mR[None, :] + m
    eCR = mCR - mC[:, None] - mR[None, :] + m
    eACR = mACR - mAC[:, :, None] - mAR[:, None, :] - mCR[None, :, :] + mA[:, None, None] + mC[None, :, None] + mR[None, None, :] - m
    eO = mACO - mAC[:, :, None]
    eRes = y - mACO[..., None] - mACR[:, :, None, :] + mAC[:, :, None, None]
    SS = {"A": nC * nO * nR * (eA ** 2).sum(), "C": nA * nO * nR * (eC ** 2).sum(), "R": nA * nC * nO * (eR ** 2).sum(),
          "AC": nO * nR * (eAC ** 2).sum(), "AR": nC * nO * (eAR ** 2).sum(), "CR": nA * nO * (eCR ** 2).sum(), "ACR": nO * (eACR ** 2).sum(),
          "O": nR * (eO ** 2).sum(), "e": (eRes ** 2).sum()}
    b = (eAC * z[None, :]).sum(1) / Szz                     # author slope deviations on z (condition means over o, r)
    SS["Alin"] = nO * nR * Szz * (b ** 2).sum()
    SS["ACrem"] = SS["AC"] - SS["Alin"]
    df = {"A": nA - 1, "C": nC - 1, "R": nR - 1, "AC": (nA - 1) * (nC - 1), "AR": (nA - 1) * (nR - 1), "CR": (nC - 1) * (nR - 1),
          "ACR": (nA - 1) * (nC - 1) * (nR - 1), "O": nA * nC * (nO - 1), "e": nA * nC * (nO - 1) * (nR - 1), "Alin": nA - 1, "ACrem": (nA - 1) * (nC - 2)}
    MS = {k: SS[k] / df[k] for k in df if df[k] > 0}
    total = ((y - m) ** 2).sum()
    return {"SS": SS, "df": df, "MS": MS, "Szz": Szz, "shape": (nA, nC, nO, nR), "ss_identity_error": abs(sum(SS[k] for k in ("A", "C", "R", "AC", "AR", "CR", "ACR", "O", "e")) - total) / total}


def ems_matrix(shape, Szz):
    nA, nC, nO, nR = shape
    rows = {"e": {"e": 1}, "O": {"e": 1, "O": nR}, "ACR": {"e": 1, "ACR": nO}, "ACrem": {"e": 1, "O": nR, "ACR": nO, "W": nO * nR},
            "Alin": {"e": 1, "O": nR, "ACR": nO, "W": nO * nR, "SLOPE": nO * nR * Szz}, "AR": {"e": 1, "ACR": nO, "AR": nC * nO},
            "CR": {"e": 1, "ACR": nO, "CR": nA * nO}, "A": {"e": 1, "O": nR, "ACR": nO, "W": nO * nR, "AR": nC * nO, "A": nC * nO * nR},
            "R": {"e": 1, "ACR": nO, "AR": nC * nO, "CR": nA * nO, "R": nA * nC * nO}}
    return {k: np.array([v.get(c, 0.0) for c in COMPONENTS], float) for k, v in rows.items()}


def identifiable(K, vec):
    """vec (combination of components) is identifiable iff it lies in the row space of K (available mean squares)."""
    coef, *_ = np.linalg.lstsq(K.T, vec, rcond=None)
    return bool(np.abs(K.T @ coef - vec).max() < TOL * max(1.0, np.abs(vec).max())), coef


def estimate(y, z):
    """Estimate components from observations. Unidentified components are None with flag UNIDENTIFIED (never 0).
    Negative ANOVA estimates are retained (flagged), never truncated silently."""
    ms = mean_squares(y, z)
    E = ems_matrix(ms["shape"], ms["Szz"])
    avail = [k for k in E if k in ms["MS"]]
    K = np.stack([E[k] for k in avail]); v = np.array([ms["MS"][k] for k in avail])
    est, flags = {}, {}
    for j, c in enumerate(COMPONENTS):
        ok, coef = identifiable(K, np.eye(len(COMPONENTS))[j])
        est[c] = float(coef @ v) if ok else None
        flags[c] = "UNIDENTIFIED" if not ok else ("NEGATIVE_ESTIMATE" if est[c] < 0 else "OK")
    return {"components": est, "flags": flags, "ms": ms, "K": K, "avail": avail, "v": v}


def combo(fit, weights):
    """Estimate a linear combination sum_c w_c * component_c; None if not identifiable."""
    vec = np.array([weights.get(c, 0.0) for c in COMPONENTS], float)
    ok, coef = identifiable(fit["K"], vec)
    return float(coef @ fit["v"]) if ok else None


def error_weights_level(shape):
    """Error variance of the author level aggregate x_a (mean over c, o, r); the z-slope averages out since sum z = 0."""
    nA, nC, nO, nR = shape
    return {"W": 1 / nC, "AR": 1 / nR, "ACR": 1 / (nC * nR), "O": 1 / (nC * nO), "e": 1 / (nC * nO * nR)}


def error_weights_slope(shape, Szz):
    nA, nC, nO, nR = shape
    return {"W": 1 / Szz, "ACR": 1 / (nR * Szz), "O": 1 / (nO * Szz), "e": 1 / (nO * nR * Szz)}


def g_coefficients(fit):
    """Repeatability-type (relative G) coefficients estimated from components only. None if not identifiable."""
    shape, Szz = fit["ms"]["shape"], fit["ms"]["Szz"]
    A, S = combo(fit, {"A": 1}), combo(fit, {"SLOPE": 1})
    el, es = combo(fit, error_weights_level(shape)), combo(fit, error_weights_slope(shape, Szz))
    gl = A / (A + el) if None not in (A, el) and (A + el) != 0 else None
    gs = S / (S + es) if None not in (S, es) and (S + es) != 0 else None
    return {"G_level": gl, "G_slope": gs, "level_error_var": el, "slope_error_var": es}


def scores(y, z):
    """Observed author scores: level aggregate and z-slope (no truth)."""
    x = y.mean((1, 2, 3)); mAC = y.mean((2, 3)); b = (mAC - mAC.mean(1, keepdims=True)) @ z / (z @ z)
    return x, b


def aligned(cov_xt, var_t, var_x):
    """Target-aligned linear variance Cov(x,t)^2/Var(t) and fidelity Cov(x,t)^2/(Var(t)Var(x)) for a scalar target t."""
    a = cov_xt ** 2 / var_t
    return a, a / var_x
