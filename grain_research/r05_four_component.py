"""R05 paired four-component identification (P03 pp.21-22): selection / interpretation / style / realization.

Model (declared assumptions, not a human law):
    y[u, arm, rep] = mu[arm] + L[arm] . theta[u] + e[u, arm, rep],  theta = (choice Q, interpretation I, style S, realization R)
Every arm observes the SAME person's theta; replicates are independent events given theta; errors are arm-specific and
independent. Estimators below consume observed arrays only (shape persons x arms x reps); truth never enters them.
"""
from __future__ import annotations

import numpy as np

COMPONENTS = ("choice", "interpretation", "style", "realization")
ARMS = ("A", "B", "C", "D", "E")
L_NOMINAL = np.array([[1, 1, 1, 1],   # A free choice
                      [0, 1, 1, 1],   # B fixed topic
                      [0, 1, 1, 1],   # C shared external material: interpretation RETAINED
                      [0, 1, 1, 0],   # D ideal realization clamp (idealized channel)
                      [0, 1, 0, 0]],  # E ideal interpretation anchor (idealized channel)
                     float)
BASIS = (0, 1, 3, 4)                  # A, B, D, E: invertible under nominal loadings
CONTRASTS = np.array([[1, -1, 0, 0, 0],   # A-B  -> choice
                      [0, 0, 0, 0, 1],    # E    -> interpretation
                      [0, 0, 0, 1, -1],   # D-E  -> style
                      [0, 1, 0, -1, 0]],  # B-D  -> realization
                     float)
NULL_WITNESS = np.array([0.0, 1.0, -1.0, 0.0])
RESTRICTION_NAMES = ("C_CC-C_BB", "C_BC-C_BB", "C_AC-C_AB", "C_CD-C_BD", "C_CE-C_BE")


# ------------------------------------------------------------------ algebra
def design_summary(L):
    L = np.asarray(L, float)
    sv = np.linalg.svd(L, compute_uv=False)
    rank = int(np.sum(sv > 1e-12 * max(1.0, sv.max())))
    _, _, vt = np.linalg.svd(L)
    null = vt[rank:].T if rank < L.shape[1] else np.zeros((L.shape[1], 0))
    return {"singular_values": sv.tolist(), "rank": rank, "null_basis": null.T.tolist()}


def implied_cov(L, Sigma):
    L = np.asarray(L, float)
    return L @ np.asarray(Sigma, float) @ L.T


def basis_transform(L_nominal, L_true, basis=BASIS):
    """T with L_nominal T = L_true on the basis arms; theta' = T theta reproduces every arm iff non-basis rows agree."""
    Mn, Mt = np.asarray(L_nominal, float)[list(basis)], np.asarray(L_true, float)[list(basis)]
    T = np.linalg.solve(Mn, Mt)
    return T, float(np.max(np.abs(np.asarray(L_nominal, float) @ T - np.asarray(L_true, float))))


def restriction_residuals(C):
    """Observable content of the nominal C = B restriction (5 moment equalities)."""
    C = np.asarray(C, float)
    return np.array([C[2, 2] - C[1, 1], C[1, 2] - C[1, 1], C[0, 2] - C[0, 1], C[2, 3] - C[1, 3], C[2, 4] - C[1, 4]])


# ------------------------------------------------------------------ estimators (observed data only)
def between_cov(Y):
    """Primary estimator: Cov(person means) - diag(two-way residual mean square / R), per arm.
    Y: (persons, arms, reps). Returns between-person arm covariance, per-arm noise variance, person means."""
    Y = np.asarray(Y, float)
    n, a, r = Y.shape
    ybar = Y.mean(2)
    cov_means = np.cov(ybar, rowvar=False, ddof=1)
    z = Y - Y.mean(0, keepdims=True)                     # column (replicate) centred
    resid = z - z.mean(2, keepdims=True)
    noise = (resid ** 2).sum(axis=(0, 2)) / ((n - 1) * (r - 1))
    return cov_means - np.diag(noise / r), noise, ybar


def between_cov_crossrep(Y):
    """Alternative formula: cross-replicate covariances (diagonal from r != r' pairs); exact identity with between_cov."""
    Y = np.asarray(Y, float)
    n, a, r = Y.shape
    flat = Y.reshape(n, a * r)
    S = np.cov(flat, rowvar=False, ddof=1).reshape(a, r, a, r)
    C = np.zeros((a, a))
    for i in range(a):
        for j in range(a):
            blk = S[i, :, j, :]
            C[i, j] = (blk.sum() - np.trace(blk)) / (r * (r - 1)) if i == j else blk.mean()
    return C


def sigma_hat(C, L):
    """Least-squares component covariance with declared L: L+ C L+'. Also the fitted-covariance residual."""
    P = np.linalg.pinv(np.asarray(L, float))
    S = P @ C @ P.T
    return S, C - implied_cov(L, S)


def naive_and_paired(C):
    return {"varA_minus_varB": float(C[0, 0] - C[1, 1]), "varB_minus_varC": float(C[1, 1] - C[2, 2]),
            "paired_var_A_minus_B": float(C[0, 0] + C[1, 1] - 2 * C[0, 1]), "var_between_B_minus_C": float(C[1, 1] + C[2, 2] - 2 * C[1, 2])}


def scores(Y_app, source_means):
    """Fixed A/B/D/E contrasts after SOURCE-only centring: A-B, E, D-E, B-D (no fitted model, no application truth)."""
    ybar = np.asarray(Y_app, float).mean(2) - np.asarray(source_means, float)[None, :]
    return ybar @ CONTRASTS.T


def summary_stats(Y, L=L_NOMINAL):
    C, noise, ybar = between_cov(Y)
    S, resid = sigma_hat(C, L)
    iu = np.triu_indices(4)
    return {"C": C, "noise": noise, "Sigma": S, "sigma_vec": S[iu], "resid_fro": float(np.linalg.norm(resid)),
            "restriction": restriction_residuals(C), **naive_and_paired(C), "means": ybar.mean(0)}


def bootstrap(Y, n_boot, seed, block=250, L=L_NOMINAL):
    """Deterministic whole-person bootstrap (all arms and replicates of a person move together)."""
    Y = np.asarray(Y, float)
    n = Y.shape[0]
    idx = np.random.default_rng(seed).integers(0, n, (n_boot, n))
    out = {k: [] for k in ("sigma_vec", "restriction", "varA_minus_varB", "varB_minus_varC", "paired_var_A_minus_B", "var_between_B_minus_C", "noise")}
    for s in range(0, n_boot, block):
        for b in idx[s:s + block]:
            st = summary_stats(Y[b], L)
            for k in out:
                out[k].append(st[k])
    return {k: np.asarray(v) for k, v in out.items()}, idx


def simulate(Sigma, L_true, mu, sd_err, n_src, n_app, reps, seed):
    """Data-generating side. Common random numbers: the same standard normals for every world with this seed.
    Returns observed arrays and a separate truth dict (evaluation only)."""
    rng = np.random.default_rng(seed)
    n = n_src + n_app
    Z = rng.standard_normal((n, 4))
    E = rng.standard_normal((n, len(mu), reps))
    theta = Z @ np.linalg.cholesky(np.asarray(Sigma, float)).T
    Y = np.asarray(mu, float)[None, :, None] + (theta @ np.asarray(L_true, float).T)[:, :, None] + sd_err * E
    return {"Y_src": Y[:n_src], "Y_app": Y[n_src:]}, {"theta_src": theta[:n_src], "theta_app": theta[n_src:]}
