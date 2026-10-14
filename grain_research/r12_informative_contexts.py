"""R12 packet 31: informative contexts versus prompt counts (declared world; known loadings).

y_ut = phi_{c(t)}' theta_u + delta_{u, a(c)} + eps_ut,  phi_c = (1, zA_c, zB_c),
delta persistent within an ACTUAL context a (Var tau2), eps independent readings (Var sigma2).
For author u the reading covariance is R = sigma2 I + tau2 * (same-actual-context indicator), so
    I = Phi' R^-1 Phi = sum_a n_a phi_a phi_a' / (sigma2 + n_a tau2)   (sum over ACTUAL contexts).
A target q' theta is data-identifiable iff q lies in the row space of I; then the known-covariance GLS variance is
q' I^+ q. Otherwise it is UNIDENTIFIED (a null-space witness w with q'w != 0 is returned); never a zero risk.
"""
from __future__ import annotations

import numpy as np

S2 = np.sqrt(2.0)
TOL = 1e-10


def design(name):
    """Contexts as dicts: phi, n, actual (delta identity), label (nominal), slots (reading indices), dslot (delta slot).
    N = 32 readings per author for every design."""
    corners = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    ctx = []
    if name == "D1":
        for c, za in enumerate([S2, S2, -S2, -S2]):
            ctx.append({"phi": np.array([1.0, za, 0.0]), "n": 8, "actual": c, "label": c, "slots": list(range(8 * c, 8 * c + 8)), "dslot": c})
    elif name == "D2":
        for c, (a, b) in enumerate(corners):
            ctx.append({"phi": np.array([1.0, a, b]), "n": 8, "actual": c, "label": c, "slots": list(range(8 * c, 8 * c + 8)), "dslot": c})
    elif name == "D3":
        for c, (a, b) in enumerate(corners):
            for h in range(2):
                j = 2 * c + h
                ctx.append({"phi": np.array([1.0, a, b]), "n": 4, "actual": j, "label": j, "slots": list(range(8 * c + 4 * h, 8 * c + 4 * h + 4)), "dslot": j})
    elif name == "D2_relabel":
        for c, (a, b) in enumerate(corners):
            for h in range(2):
                ctx.append({"phi": np.array([1.0, a, b]), "n": 4, "actual": c, "label": 2 * c + h, "slots": list(range(8 * c + 4 * h, 8 * c + 4 * h + 4)), "dslot": c})
    else:
        raise KeyError(name)
    assert sum(k["n"] for k in ctx) == 32
    return ctx


def actual_groups(ctx):
    """Merge nominal pieces sharing an ACTUAL context (same delta)."""
    g = {}
    for k in ctx:
        e = g.setdefault(k["actual"], {"phi": k["phi"], "n": 0, "slots": []})
        assert np.allclose(e["phi"], k["phi"])
        e["n"] += k["n"]; e["slots"] += k["slots"]
    return list(g.values())


def info(ctx, sigma2, tau2, by="actual"):
    """Aggregated information. by='actual' is correct; by='label' treats nominal labels as independent (deliberately wrong)."""
    groups = actual_groups(ctx) if by == "actual" else [{"phi": k["phi"], "n": k["n"]} for k in ctx]
    return sum(g["n"] * np.outer(g["phi"], g["phi"]) / (sigma2 + g["n"] * tau2) for g in groups)


def info_dense(ctx, sigma2, tau2):
    """Independent check: explicit Phi' R^-1 Phi with the full 32 x 32 block covariance."""
    N = 32; Phi = np.zeros((N, 3)); act = np.zeros(N, int)
    for k in ctx:
        for s in k["slots"]:
            Phi[s] = k["phi"]; act[s] = k["actual"]
    R = sigma2 * np.eye(N) + tau2 * (act[:, None] == act[None, :])
    return Phi.T @ np.linalg.solve(R, Phi)


def identifiable(I, q):
    w, V = np.linalg.eigh(I)
    keep = w > w.max() * TOL
    P = V[:, keep] @ V[:, keep].T
    ok = bool(np.linalg.norm(P @ q - q) <= 1e-9 * max(1.0, np.linalg.norm(q)))
    witness = None
    if not ok:
        N = V[:, ~keep]; w0 = N @ (N.T @ q); witness = w0 / np.linalg.norm(w0)
    return ok, witness, int(keep.sum()), w


def gls_risk(I, q):
    ok, wit, rank, eig = identifiable(I, q)
    return (float(q @ np.linalg.pinv(I, rcond=TOL) @ q) if ok else None), ok, wit, rank, eig


def context_means(Y, ctx):
    """Per-author means over the readings of each ACTUAL context."""
    G = actual_groups(ctx)
    return np.stack([Y[:, g["slots"]].mean(1) for g in G], 1), G


def source_fit(Y, ctx):
    """Estimate sigma2 and tau2 from source observations and known phi ONLY (no theta, no generating values).
    sigma2: pooled within-actual-context variance. tau2: per-author OLS of context means on phi; residual mean square
    minus sigma2/n (balanced n). Returns raw and constrained (>= 0) values with df and boundary flag."""
    G = actual_groups(ctx); ns = {g["n"] for g in G}
    if len(ns) != 1:
        raise ValueError("balanced source design required")
    n = ns.pop(); U = Y.shape[0]; K = len(G)
    within = sum(((Y[:, g["slots"]] - Y[:, g["slots"]].mean(1, keepdims=True)) ** 2).sum() for g in G)
    df_s = U * K * (n - 1); s2 = within / df_s
    M, _ = context_means(Y, ctx); Phi = np.stack([g["phi"] for g in G]); r = np.linalg.matrix_rank(Phi)
    H = Phi @ np.linalg.pinv(Phi); res = M - M @ H.T
    df_t = U * (K - r); ms = (res ** 2).sum() / df_t
    t2 = ms - s2 / n
    return {"sigma2": float(s2), "tau2_raw": float(t2), "tau2_used": float(max(t2, 0.0)), "tau2_boundary": bool(t2 < 0), "df_sigma2": int(df_s), "df_tau2": int(df_t),
            "ms_context_residual": float(ms)}


def gls_scores(Y, ctx, sigma2, tau2):
    """Per-author GLS estimate of theta (min-norm for rank-deficient designs) using ACTUAL-context covariance."""
    M, G = context_means(Y, ctx)
    Phi = np.stack([g["phi"] for g in G]); w = np.array([g["n"] / (sigma2 + g["n"] * tau2) for g in G])
    I = (Phi * w[:, None]).T @ Phi
    return (M * w[None, :]) @ Phi @ np.linalg.pinv(I, rcond=TOL).T, I
