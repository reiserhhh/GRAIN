"""GRAIN-C4-02: continuous-time occasion state on residuals from a frozen author level.

r_t = s_t + e_t,  s_{t+1} = rho s_t + w,  rho = exp(-lambda * gap),  Var(w) = q (1 - rho^2),
Var(e) = sigma2, s_0 ~ N(0, q). Standard scalar Kalman filtering with irregular gaps (an OU
state); maximum likelihood on source residual sequences. Not an emotion model.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def loglik(params, R, G):
    """R (A, T) residuals, G (A, T-1) gaps in days; vectorized over authors."""
    lam, q, s2 = params
    A, T = R.shape
    m, P = np.zeros(A), np.full(A, q)
    ll = 0.0
    for t in range(T):
        S = P + s2
        ll += float(np.sum(-0.5 * (np.log(2 * np.pi * S) + (R[:, t] - m) ** 2 / S)))
        K = P / S
        m = m + K * (R[:, t] - m)
        P = (1 - K) * P
        if t < T - 1:
            rho = np.exp(-lam * G[:, t])
            m, P = rho * m, rho ** 2 * P + q * (1 - rho ** 2)
    return ll


def fit(R, G):
    v = float(np.var(R))
    x0 = np.log([1 / 30, 0.3 * v, 0.7 * v])
    res = minimize(lambda x: -loglik(np.exp(x), R, G), x0, method="L-BFGS-B",
                   bounds=[(np.log(1e-6), np.log(1e3)), (np.log(1e-9 * v), np.log(10 * v)), (np.log(1e-9 * v), np.log(10 * v))])
    lam, q, s2 = np.exp(res.x)
    return {"lambda_per_day": float(lam), "half_life_days": float(np.log(2) / lam), "q": float(q), "sigma2": float(s2),
            "state_share": float(q / (q + s2)), "converged": bool(res.success), "neg_loglik": float(res.fun), "var_resid": v}


def prequential(params, r, gaps, start_eval):
    """Filter one author's residual sequence r (T,) with gaps (T-1,); return predictive mean/var for t >= start_eval."""
    lam, q, s2 = params["lambda_per_day"], params["q"], params["sigma2"]
    m, P = 0.0, q
    means, vars_ = [], []
    for t in range(len(r)):
        if t >= start_eval:
            means.append(m)
            vars_.append(P + s2)
        S = P + s2
        K = P / S
        m, P = m + K * (r[t] - m), (1 - K) * P
        if t < len(r) - 1:
            rho = np.exp(-lam * gaps[t])
            m, P = rho * m, rho ** 2 * P + q * (1 - rho ** 2)
    return np.array(means), np.array(vars_)
