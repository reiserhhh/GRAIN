"""Packet 34: paper-equation implementation of ALIRT (Varadarajan et al., NAACL 2024) plus the operator-aware comparator.

NOT the authors' code (repo 404; mirt/mirtCAT unavailable). Phases:
 1. per-item ridge mapping of source measure Y on item response features (D_poly), quartile thresholds -> K=4 categories;
 2. Samejima graded response model (one discrimination per item, ordered thresholds) fitted by marginal ML on D_tr;
 3. CAT: maximum Fisher information over unadministered items, MAP update from all administered responses.
Fitters/policies take observations and source data only; generating values are used only by the simulator/auditor.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logsumexp

K = 4
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(41)
GH_LW = np.log(GH_W / GH_W.sum())


# ---------------- Phase 1: polytomization
def ridge_fit(X, y, lam=1.0):
    Xc = np.column_stack([np.ones(len(X)), X]); P = lam * np.eye(Xc.shape[1]); P[0, 0] = 0.0
    return np.linalg.solve(Xc.T @ Xc + P, Xc.T @ y)


def ridge_predict(coef, X):
    return coef[0] + X @ coef[1:]


def polytomize_fit(Xpoly, Ypoly):
    """Xpoly (N, J, p): per-item ridge of Y on item features; thresholds = internal quartiles of in-sample predictions."""
    J = Xpoly.shape[1]; coefs, thr = [], []
    for j in range(J):
        c = ridge_fit(Xpoly[:, j], Ypoly); coefs.append(c)
        thr.append(np.quantile(ridge_predict(c, Xpoly[:, j]), [k / K for k in range(1, K)]))
    return {"coef": coefs, "thr": np.array(thr)}


def predict_items(pm, X):
    return np.stack([ridge_predict(pm["coef"][j], X[:, j]) for j in range(X.shape[1])], 1)


def categorize(pm, Yhat):
    return np.stack([np.searchsorted(pm["thr"][j], Yhat[:, j], side="right") for j in range(Yhat.shape[1])], 1)


# ---------------- Phase 2: graded response model
def unpack(par, J):
    P = par.reshape(J, K)
    a = np.exp(P[:, 0]); b = np.cumsum(np.column_stack([P[:, 1], np.exp(P[:, 2:])]), 1)
    return a, b


def pstar(theta, a, b):
    """P*(Y >= k | theta) for k = 1..K-1: shape (Q, J, K-1)."""
    return expit(a[None, :, None] * (np.asarray(theta)[:, None, None] - b[None, :, :]))


def cat_probs(theta, a, b):
    ps = pstar(np.atleast_1d(theta), a, b)
    Q, J = ps.shape[:2]
    full = np.concatenate([np.ones((Q, J, 1)), ps, np.zeros((Q, J, 1))], 2)
    return full[:, :, :-1] - full[:, :, 1:]


def negloglik(par, O, J):
    a, b = unpack(par, J)
    P = np.clip(cat_probs(GH_X, a, b), 1e-300, None)        # (Q, J, K)
    L = np.zeros((len(GH_X), O.shape[0]))
    for j in range(J):
        L += np.log(P[:, j, :][:, O[:, j]])
    return -float(logsumexp(L + GH_LW[:, None], axis=0).sum())


def fit_grm(O):
    J = O.shape[1]
    init = np.zeros((J, K)); init[:, 0] = 0.0; init[:, 1] = -1.0; init[:, 2:] = np.log(1.0)
    res = minimize(negloglik, init.ravel(), args=(O, J), method="L-BFGS-B", options={"maxiter": 2000})
    a, b = unpack(res.x, J)
    return {"a": a, "b": b, "par": res.x, "converged": bool(res.success), "message": str(res.message), "nit": int(res.nit), "nll": float(res.fun)}


def item_info(theta, a, b):
    """Samejima GRM Fisher information per item at scalar theta: sum_k (dP_k/dtheta)^2 / P_k."""
    ps = pstar(np.array([theta]), a, b)[0]                  # (J, K-1)
    full = np.column_stack([np.ones(len(a)), ps, np.zeros(len(a))])
    P = full[:, :-1] - full[:, 1:]
    d = a[:, None] * full * (1 - full)
    dP = d[:, :-1] - d[:, 1:]
    return (dP ** 2 / np.clip(P, 1e-300, None)).sum(1)


def map_theta(items, resp, a, b, m0=0.0, s0=1.0):
    if len(items) == 0:
        return m0
    it = np.array(items); rs = np.array(resp)
    f = lambda th: -(np.log(np.clip(cat_probs(th, a[it], b[it])[0, np.arange(len(it)), rs], 1e-300, None)).sum() - 0.5 * ((th - m0) / s0) ** 2)
    return float(minimize_scalar(f, bounds=(-6, 6), method="bounded", options={"xatol": 1e-8}).x)


def alirt_cat(Orow, a, b, budget, m0=0.0):
    """One application author: MFI next item among unadministered, MAP update from all administered responses."""
    left = list(range(len(a))); items, resp, th = [], [], m0
    for _ in range(budget):
        info = item_info(th, a, b)
        j = max(left, key=lambda i: info[i])
        items.append(j); resp.append(int(Orow[j])); left.remove(j)
        th = map_theta(items, resp, a, b, m0)
    return items, th


# ---------------- operator-aware comparator (source-fitted Gaussian condition-response model on projected features)
def fit_operator(Xsrc, Xpoly, Ypoly, z):
    """Xsrc = D_tr features (N, J, p); Xpoly/Ypoly = D_poly (sign and Y bridge). Returns the measurement model."""
    pooled = np.concatenate([Xsrc.reshape(-1, Xsrc.shape[2]), Xpoly.reshape(-1, Xpoly.shape[2])])
    pooled = pooled - pooled.mean(0)
    w = np.linalg.svd(pooled, full_matrices=False)[2][0]
    if np.corrcoef((Xpoly @ w).mean(1), Ypoly)[0, 1] < 0:
        w = -w
    S = Xsrc @ w; mu = S.mean(0)
    Z = np.column_stack([np.ones(len(z)), z]); ZtZi = np.linalg.inv(Z.T @ Z)
    Bhat = (S - mu) @ Z @ ZtZi.T
    res = (S - mu) - Bhat @ Z.T; sig2 = float((res ** 2).sum() / (res.size - 2 * S.shape[0]))
    prior_cov = np.cov(Bhat.T) - sig2 * ZtZi
    ev, V = np.linalg.eigh(prior_cov); prior_cov = (V * np.maximum(ev, 1e-6)) @ V.T
    m = {"w": w, "mu": mu, "z": z, "sig2": sig2, "prior_cov": prior_cov, "prior_mean": np.zeros(2)}
    ap = posterior(m, Xpoly @ w, list(range(len(z))))[0][:, 0]
    m["bridge"] = np.polyfit(ap, Ypoly, 1)
    return m


def posterior(m, S, items):
    """Gaussian posterior of (alpha, beta) for each author from projected scores S (N, J) on the given items."""
    it = np.array(items); Z = np.column_stack([np.ones(len(it)), m["z"][it]])
    P0 = np.linalg.inv(m["prior_cov"]); prec = P0 + Z.T @ Z / m["sig2"]; cov = np.linalg.inv(prec)
    mean = ((S[:, it] - m["mu"][it]) @ Z / m["sig2"]) @ cov.T
    return mean, cov


def operator_targets(m, mean, cov, zstar):
    c1, c0 = m["bridge"]
    cs = np.array([1.0, zstar])
    return {"scalar": c0 + c1 * mean[:, 0], "newcond": c0 + c1 * (mean @ cs),
            "sd_scalar": abs(c1) * np.sqrt(cov[0, 0]), "sd_newcond": abs(c1) * np.sqrt(cs @ cov @ cs)}


def operator_select(m, budget, zstar):
    """Target-aware sequential design (Gaussian posterior variance does not depend on responses: same for all authors)."""
    left = list(range(len(m["z"]))); items = []; cs = np.array([1.0, zstar])
    for _ in range(budget):
        best = min(left, key=lambda j: cs @ np.linalg.inv(np.linalg.inv(m["prior_cov"]) + sum(np.outer([1, m["z"][i]], [1, m["z"][i]]) for i in items + [j]) / m["sig2"]) @ cs)
        items.append(best); left.remove(best)
    return items
