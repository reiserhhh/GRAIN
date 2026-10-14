"""R04 packet 32: generating causes (goal G, audience A, language L, proficiency K) versus query-specific sensors.

Declared static subcase (FU3-e's opportunity/history/state feedback held fixed = absent here; already answered there).
Expression (pre-realization):   e_ut = theta_u w_th + g_t w_G + a_t w_A + d_{L_u}
Realization (K on the SPEAKER):  x_ut = K_u M_{L_u} e_ut + sig_x eps_ut           (K = attainable expression gain)
Query sensors on the SAME x:     y_utq = r_q' x_ut + lam_q a_t + sig_q eta_utq     (lam_q = reader-side context response)
Per author, with recorded crossed (g, a): OLS  y_q = alpha_q + bG_q g + bA_q a  gives
    alpha_q = K (theta b_th,q + b_d,q),  bG_q = K b_G,q,  bA_q = K b_A,q + lam_q,   b_.,q = r_q' M_L w_.
so rho_q = alpha_q / bG_q = theta c_q^L + o_q^L with K cancelling (c = b_th/b_G, o = b_d/b_G).
Estimation uses observations and designed/recorded inputs only; author theta/K and loadings are simulator-only.
"""
from __future__ import annotations

import numpy as np

W_TH = np.array([1.0, 0.0, 0.4]); W_G = np.array([0.0, 2.0, 0.5]); W_A = np.array([0.3, 0.2, 1.0])
M = {0: np.eye(3), 1: np.diag([1.3, 0.8, 1.0])}
D = {0: np.zeros(3), 1: np.array([0.3, -0.2, 0.1])}
SENS = {"q1": (np.array([1.0, 0.5, 0.0]), 0.0), "q2": (np.array([0.0, 0.6, 1.0]), 0.4), "q1b": (np.array([1.0, 0.5, 0.0]), 0.0)}
PAIRS = {"distinct": ("q1", "q2"), "replica": ("q1", "q1b")}
SIG_X, SIG_Q, SD_LOGK, MU_L1, T = 0.3, 0.3, 0.25, 0.4, 32


def crossed(U):
    cells = np.array([(0, 0), (1, 0), (0, 1), (1, 1)] * (T // 4), float)
    return np.tile(cells[:, 0], (U, 1)), np.tile(cells[:, 1], (U, 1))


def generate(theta, K, L, g, a, pair, Ex, Eq):
    """SIMULATOR ONLY."""
    U = len(theta); y = np.zeros((U, T, 2))
    for lang in (0, 1):
        s = L == lang
        if not s.any():
            continue
        e = theta[s, None, None] * W_TH + g[s, :, None] * W_G + a[s, :, None] * W_A + D[lang]
        x = K[s, None, None] * (e @ M[lang].T) + SIG_X * Ex[s]
        for j, q in enumerate(PAIRS[pair]):
            r, lam = SENS[q]
            y[s, :, j] = x @ r + lam * a[s] + SIG_Q * Eq[s, :, j]
    return y


def author_ols(y, g, a):
    """Per-author OLS of each query on [1, g, a] (recorded design only). Returns coefficients, residual covariance, (X'X)^-1."""
    X = np.stack([np.ones_like(g), g, a], -1)
    XtXi = np.linalg.inv(np.einsum("uti,utj->uij", X, X))
    B = np.einsum("uij,utj,utq->uiq", XtXi, X, y)
    res = y - np.einsum("uti,uiq->utq", X, B)
    S = np.einsum("utq,utr->uqr", res, res) / (y.shape[1] - 3)
    return {"alpha": B[:, 0], "bG": B[:, 1], "bA": B[:, 2], "S": S, "XtXi": XtXi}


def ratio(st):
    """rho_q = alpha_q/bG_q and its delta-method noise variance per author and query."""
    al, bg, S, Xi = st["alpha"], st["bG"], st["S"], st["XtXi"]
    rho = al / bg
    sqq = np.einsum("uqq->uq", S)
    va, vb, cab = sqq * Xi[:, 0, 0][:, None], sqq * Xi[:, 1, 1][:, None], sqq * Xi[:, 0, 1][:, None]
    return rho, (va + rho ** 2 * vb - 2 * rho * cab) / bg ** 2


def calibrate(src0, src1_mono=None, anchor0=None, anchor1=None):
    """Source-only calibration (no theta/K/loadings). Reference nu: L0 source population standardized (declared).
    L1 either from bilingual anchors (same authors observed in both languages) or, without anchors, by within-L1
    standardization (which implicitly sets the L1 group mean to the L0 reference mean)."""
    r0, v0 = ratio(src0)
    c0 = np.sqrt(np.var(r0, 0, ddof=1) - v0.mean(0)); e0 = r0.mean(0)
    cal = {"off": {0: e0}, "scale": {0: c0}, "method_L1": None}
    if anchor0 is not None:
        a0, av0 = ratio(anchor0); a1, _ = ratio(anchor1)
        s = np.array([np.cov(a1[:, j], a0[:, j])[0, 1] / (np.var(a0[:, j], ddof=1) - av0[:, j].mean()) for j in range(a0.shape[1])])
        e1 = a1.mean(0) - s * a0.mean(0)
        cal["off"][1] = e1 + s * e0; cal["scale"][1] = s * c0; cal["method_L1"] = "bilingual_anchor"
    else:
        r1, v1 = ratio(src1_mono)
        cal["off"][1] = r1.mean(0); cal["scale"][1] = np.sqrt(np.var(r1, 0, ddof=1) - v1.mean(0)); cal["method_L1"] = "within_language_standardization"
    return cal


def reader_side(src0):
    """Reader-side context response lam_q vs generation-side audience effect: across authors bA_q = (b_A/b_G) bG_q + lam_q;
    errors-in-variables corrected slope; identified only if realization gain varies across authors."""
    bG, bA, S, Xi = src0["bG"], src0["bA"], src0["S"], src0["XtXi"]
    vbg = np.einsum("uqq->uq", S) * Xi[:, 1, 1][:, None]
    out = []
    for j in range(bG.shape[1]):
        sl = np.cov(bA[:, j], bG[:, j])[0, 1] / (np.var(bG[:, j], ddof=1) - vbg[:, j].mean())
        out.append({"slope_bA_on_bG": float(sl), "lambda_hat": float(bA[:, j].mean() - sl * bG[:, j].mean()), "var_bG_signal": float(np.var(bG[:, j], ddof=1) - vbg[:, j].mean())})
    return out


def score(st, L, cal):
    """theta-hat = mean_q (rho_q - off_q^L)/scale_q^L; delta-method SE conditional on the frozen calibration."""
    rho, _ = ratio(st)
    off = np.where(L[:, None] == 1, cal["off"][1], cal["off"][0]); sc = np.where(L[:, None] == 1, cal["scale"][1], cal["scale"][0])
    th = ((rho - off) / sc).mean(1)
    Q = rho.shape[1]
    al, bg, S, Xi = st["alpha"], st["bG"], st["S"], st["XtXi"]
    ga = 1 / (Q * sc * bg); gb = -al / (Q * sc * bg ** 2)
    var = np.zeros(len(th))
    for i in range(Q):
        for j in range(Q):
            var += S[:, i, j] * (ga[:, i] * ga[:, j] * Xi[:, 0, 0] + gb[:, i] * gb[:, j] * Xi[:, 1, 1] + (ga[:, i] * gb[:, j] + gb[:, i] * ga[:, j]) * Xi[:, 0, 1])
    return th, np.sqrt(var)


def oracle_cal(pair):
    """REFERENCE ONLY (labelled): known loadings."""
    cal = {"off": {}, "scale": {}, "method_L1": "oracle_known_loadings"}
    for lang in (0, 1):
        bth = np.array([SENS[q][0] @ M[lang] @ W_TH for q in PAIRS[pair]]); bg = np.array([SENS[q][0] @ M[lang] @ W_G for q in PAIRS[pair]])
        bd = np.array([SENS[q][0] @ M[lang] @ D[lang] for q in PAIRS[pair]])
        cal["scale"][lang] = bth / bg; cal["off"][lang] = bd / bg
    return cal
