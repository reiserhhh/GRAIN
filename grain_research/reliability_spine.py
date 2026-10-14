"""R09 reliability spine (docs/GRAIN_R09_RELIABILITY_SPINE_PROTOCOL_V1.md): design, generator, REML, posterior, reliability.

Binding reliability definition: R = 1 - E[Var(theta|Y)] / Var(theta) (total variance), per scalar target.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve

OBS = np.array([(-1., -1.), (-1., 1.), (1., -1.), (1., 1.)])
HELD = np.array([(-.5, 0.), (.5, 0.), (0., -.5), (0., .5)])
CELLS = [(ci, f, r) for ci in range(4) for f in range(3) for r in range(2)]          # 24 per occasion
MU = np.array([0., 0., .5, .2, -.1, .4])
SIG = np.zeros((6, 6))
SIG[0, 0] = SIG[1, 1] = 1.; SIG[0, 1] = SIG[1, 0] = .3
SIG[2, 2], SIG[3, 3], SIG[4, 4], SIG[5, 5] = .16, .04, .04, .16
SIG[2, 5] = SIG[5, 2] = .048
G = np.array([[.2, -.1, -.1], [-.15, .05, .10]])                                   # [k, f], sum_f = 0
BETA_TRUE = np.r_[MU, G[0, 0], G[0, 1], G[1, 0], G[1, 1]]
RHO, SW2, SE2 = .6, .16, .25
SS2 = SW2 / (1 - RHO ** 2)
WORLDS = {"LOW": {"sd2": .05, "curved": False}, "HIGH": {"sd2": .15, "curved": False}, "CURVED": {"sd2": .05, "curved": True}}


def orders(seed: int = 18090) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack([rng.permutation(24) for _ in range(4)])                    # P[p][t] = cell index at time t


P_ORDERS = orders()
ROW = np.array([(o, ci, f, r, k, cell) for o in range(2) for cell, (ci, f, r) in enumerate(CELLS) for k in range(2)])
R_O, R_CI, R_F, R_R, R_K, R_CELL = ROW.T
NROW = len(ROW)                                                                  # 96


def design():
    X = np.zeros((NROW, 10))
    for i, (o, ci, f, r, k, cell) in enumerate(ROW):
        X[i, k] = 1.
        X[i, 2 + 2 * k: 4 + 2 * k] = OBS[ci]
        X[i, 6 + 2 * k: 8 + 2 * k] = [int(f == 0) - int(f == 2), int(f == 1) - int(f == 2)]
    Zd = np.zeros((NROW, 24))
    Zd[np.arange(NROW), (R_CI * 3 + R_F) * 2 + R_K] = 1.
    return X, X[:, :6].copy(), Zd


X, ZT, ZD = design()


def pair_of(u: int) -> tuple[int, int]:
    return (u % 4, (u + 1) % 4)


def times(pair) -> np.ndarray:
    inv = [np.argsort(P_ORDERS[p]) for p in pair]                               # inv[o][cell] = time
    return np.array([inv[o][cell] for o, cell in zip(R_O, R_CELL)])


def state_cov(rho: float, ss2: float, pair) -> np.ndarray:
    t = times(pair)
    same = (R_O[:, None] == R_O[None, :]) & (R_K[:, None] == R_K[None, :])
    return np.where(same, ss2 * rho ** np.abs(t[:, None] - t[None, :]), 0.)


def generate(world: str, seed: int, n_src: int = 300, n_app: int = 120):
    wi = list(WORLDS).index(world)
    rng = np.random.default_rng([18092, wi, seed])
    n = n_src + n_app
    theta = rng.multivariate_normal(MU, SIG, n)
    gamma = rng.normal(.3, .2, (n, 2)) if WORLDS[world]["curved"] else np.zeros((n, 2))
    d = rng.normal(0, np.sqrt(WORLDS[world]["sd2"]), (n, 24))
    Y = np.zeros((n, NROW)); S = np.zeros((n, NROW))
    for u in range(n):
        t = times(pair_of(u))
        path = np.zeros((2, 24, 2))
        for o in range(2):
            for k in range(2):
                s = rng.normal(0, np.sqrt(SS2))
                for tt in range(24):
                    if tt > 0:
                        s = RHO * s + rng.normal(0, np.sqrt(SW2))
                    path[o, tt, k] = s
        S[u] = path[R_O, t, R_K]
        mean = X @ np.r_[theta[u], BETA_TRUE[6:]]                                # kappa = 0 at observed conditions
        Y[u] = mean + d[u, (R_CI * 3 + R_F) * 2 + R_K] + S[u] + rng.normal(0, np.sqrt(SE2), NROW)
    m_held = np.zeros((n, 2, 4))
    for k in range(2):
        m_held[:, k, :] = theta[:, [k]] + theta[:, 2 + 2 * k: 4 + 2 * k] @ HELD.T + .875 * gamma[:, [k]]
    return {"Y": Y, "S": S, "theta": theta, "gamma": gamma, "d": d, "m_held": m_held, "n_src": n_src, "pairs": [pair_of(u) for u in range(n)]}


# ------------------------------------------------------------ covariance parameters and REML
TRI = np.tril_indices(6)


def unpack(psi):
    L = np.zeros((6, 6)); L[TRI] = psi[:21]
    L[np.diag_indices(6)] = np.exp(np.diag(L))
    return {"Sig": L @ L.T, "sd2": float(np.exp(psi[21])), "rho": float(np.tanh(psi[22])), "sw2": float(np.exp(psi[23])), "se2": float(np.exp(psi[24]))}


def pack(Sig, sd2, rho, sw2, se2):
    L = np.linalg.cholesky(Sig); Lc = L.copy(); Lc[np.diag_indices(6)] = np.log(np.diag(L))
    return np.r_[Lc[TRI], np.log(sd2), np.arctanh(rho), np.log(sw2), np.log(se2)]


def V_of(par, pair, rows=None):
    V = ZT @ par["Sig"] @ ZT.T + par["sd2"] * ZD @ ZD.T + state_cov(par["rho"], par["sw2"] / (1 - par["rho"] ** 2), pair) + par["se2"] * np.eye(NROW)
    return V if rows is None else V[np.ix_(rows, rows)]


def reml_objective(psi, groups):
    try:
        par = unpack(psi)
        if not np.isfinite(list(par.values())[1:]).all():
            return 1e12
        XtVX = np.zeros((10, 10)); XtVy = np.zeros(10); ld = 0.; pieces = []
        for pair, Yg in groups:
            cf = cho_factor(V_of(par, pair), lower=True)
            ld += len(Yg) * 2 * np.log(np.diag(cf[0])).sum()
            ViX = cho_solve(cf, X); ViY = cho_solve(cf, Yg.T)
            XtVX += len(Yg) * X.T @ ViX; XtVy += X.T @ ViY.sum(1)
            pieces.append((Yg, ViX, ViY))
        beta = np.linalg.solve(XtVX, XtVy)
        quad = 0.
        for Yg, ViX, ViY in pieces:
            quad += (Yg * ViY.T).sum() - 2 * beta @ (X.T @ ViY.sum(1)) + len(Yg) * beta @ (X.T @ ViX) @ beta
        sign, ldx = np.linalg.slogdet(XtVX)
        return float(ld + quad + ldx)
    except (np.linalg.LinAlgError, ValueError):
        return 1e12


def gls_beta(par, groups):
    XtVX = np.zeros((10, 10)); XtVy = np.zeros(10)
    for pair, Yg in groups:
        cf = cho_factor(V_of(par, pair), lower=True)
        XtVX += len(Yg) * X.T @ cho_solve(cf, X); XtVy += X.T @ cho_solve(cf, Yg.T).sum(1)
    return np.linalg.solve(XtVX, XtVy)


def ols_theta(Yu):
    """Per-person OLS of y on (1, c1, c2) per coordinate over all rows -> theta (a1,a2,B11,B12,B21,B22), naive residual var per k."""
    out = np.zeros(6); s2 = np.zeros(2); XtXi = []
    for k in range(2):
        m = R_K == k
        A = np.c_[np.ones(m.sum()), OBS[R_CI[m]]]
        coef, *_ = np.linalg.lstsq(A, Yu[m], rcond=None)
        out[k] = coef[0]; out[2 + 2 * k: 4 + 2 * k] = coef[1:]
        res = Yu[m] - A @ coef
        s2[k] = res @ res / (m.sum() - 3); XtXi.append(np.linalg.inv(A.T @ A))
    return out, s2, XtXi


# ------------------------------------------------------------ posterior (Gaussian conditioning)
def posterior(par, pair, rows):
    """b = (theta - mu [6], d [24], row states [96]); returns gain (126 x n_rows) and posterior covariance (126 x 126)."""
    P = np.zeros((126, 126))
    P[:6, :6] = par["Sig"]; P[6:30, 6:30] = par["sd2"] * np.eye(24)
    P[30:, 30:] = state_cov(par["rho"], par["sw2"] / (1 - par["rho"] ** 2), pair)
    rows = np.asarray(rows, int)
    if len(rows) == 0:
        return np.zeros((126, 0)), P
    Zb = np.c_[ZT[rows], ZD[rows], np.eye(NROW)[rows]]
    V = Zb @ P @ Zb.T + par["se2"] * np.eye(len(rows))
    cf = cho_factor(V, lower=True)
    Kg = cho_solve(cf, Zb @ P).T
    return Kg, P - Kg @ Zb @ P


def reliability(prior_var: float, post_vars) -> float:
    """R = 1 - E[Var(theta|Y)] / Var(theta)."""
    return float(1 - np.mean(post_vars) / prior_var)


def rows_half(o: int, f: int) -> np.ndarray:
    return np.flatnonzero((R_O == o) & (R_F == f))


def rows_prefix(pair, o_target: int, t_target: int) -> np.ndarray:
    t = times(pair)
    return np.flatnonzero((R_O < o_target) | ((R_O == o_target) & (t <= t_target)))


def row_at(pair, o: int, t_target: int, k: int) -> int:
    t = times(pair)
    return int(np.flatnonzero((R_O == o) & (t == t_target) & (R_K == k))[0])


def held_L(k: int, h: int) -> np.ndarray:
    L = np.zeros(6); L[k] = 1.; L[2 + 2 * k: 4 + 2 * k] = HELD[h]
    return L


def icc_two_way(M):
    """McGraw & Wong single-measure ICC(A,1) (absolute agreement) and ICC(C,1) (consistency) for an n x 2 matrix."""
    n, k = M.shape
    gm = M.mean(); rm = M.mean(1); cm = M.mean(0)
    ssr = k * ((rm - gm) ** 2).sum(); ssc = n * ((cm - gm) ** 2).sum(); sst = ((M - gm) ** 2).sum()
    sse = sst - ssr - ssc
    msr, msc, mse = ssr / (n - 1), ssc / (k - 1), sse / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)), float((msr - mse) / (msr + (k - 1) * mse))
