"""R01 packet 27: common donor shift versus text-specific vector action (source-only; cached derivatives).

Crossed entries T_ij = d_i . u_j (u = sqrt(s) J) are cached DIRECTIONAL DERIVATIVES, not new persons, finite native
patch outcomes or labels. In an orthonormal basis Q of the training-recipient row space (U = V Q', D_Q = D Q, T = D_Q V'):

    ZERO    : T_hat = 0
    COMMON  : T_hat_ij = dbar_Q . v_j,              dbar_Q = mean_rows(D_Q)  (training-mean donor, projected)
    LINEAR  : T_hat_ij = z_i B v_j                    (packet-24 full-cross no-intercept ridge, replayed exactly)
    AFFINE  : T_hat_ij = (c + z_i B) . v_j,           c unpenalized, slope penalty identical to LINEAR

All fits use the full-cross mean loss (1/m) sum_ij (pred_ij - T_ij)^2 + lam ||B||_F^2 with m = n * n_recipients and
lam = 1/n. AFFINE: B solves the centered Sylvester equation (Zc'Zc) B (V'V) + m lam B = Zc' Tc V with Zc = Z - 1 zbar,
Tc = T - 1 mean_rows(T); then c = mean_rows(T) V (V'V)^-1 - zbar B, which equals mean_rows(D_Q) - zbar B when T = D_Q V'.
The intercept is NOT regularized (documented change relative to treating it as an extra penalized feature).
"""
from __future__ import annotations

import numpy as np

from grain_research import r01_source_crossed as x

RANK_TOL = x.RANK_TOL


def fit_linear(Z, V, T, lam):
    """Packet-24 full-cross no-intercept ridge (unchanged frozen solver)."""
    return x.fit_full(Z, V, T, lam)


def fit_affine(Z, V, T, lam):
    """Centered Sylvester slope + unpenalized intercept. Returns (B (p x r), c (r,))."""
    zbar = Z.mean(0)
    tbar = T.mean(0)                                   # mean over donor rows, per recipient column
    B = x.fit_full(Z - zbar, V, T - tbar[None, :], lam)
    c = np.linalg.solve(V.T @ V, V.T @ tbar) - zbar @ B
    return B, c


def fit_affine_joint(Z, V, T, lam):
    """Explicit joint primal solve (checks only): rows (i, j), features [v_j | kron(v_j, z_i)], penalty on B block only."""
    n, p = Z.shape
    nr, r = V.shape
    PhiB = np.einsum("jb,ia->ijba", V, Z).reshape(n * nr, r * p)
    Phic = np.broadcast_to(V[None, :, :], (n, nr, r)).reshape(n * nr, r)
    Phi = np.hstack([Phic, PhiB])
    m = n * nr
    pen = np.r_[np.zeros(r), np.full(r * p, m * lam)]
    sol = np.linalg.solve(Phi.T @ Phi + np.diag(pen), Phi.T @ T.reshape(-1))
    return sol[r:].reshape(r, p).T, sol[:r]


def common_coef(DQ):
    return DQ.mean(0)


def mean_loss(Z, V, T, B, c, lam):
    """Full-cross mean loss with slope penalty (c may be zeros)."""
    P = (c[None, :] + Z @ B) @ V.T
    return float(np.mean((P - T) ** 2) + lam * np.sum(B ** 2))


def affine_grad(Z, V, T, B, c, lam):
    """Max-abs gradient of the joint objective wrt (B, c); stationarity check."""
    n, nr = Z.shape[0], V.shape[0]
    E = (c[None, :] + Z @ B) @ V.T - T                # n x nr
    m = n * nr
    gB = (2 / m) * Z.T @ E @ V + 2 * lam * B
    gc = (2 / m) * E.sum(0) @ V
    return float(max(np.abs(gB).max(), np.abs(gc).max()))


def predict(z, B, c, v):
    return float((c + z @ B) @ v)


def _proj(M):
    return M @ np.linalg.pinv(M, rcond=RANK_TOL)


def projection_shares(Z, T):
    """Left (donor-space) projection shares of T. Orthogonal identity: P_[1,Z] = P_1 + P_Zc (Zc = centered Z)."""
    n = Z.shape[0]
    one = np.ones((n, 1))
    tot = float(np.sum(T ** 2))
    PZ, P1, P1Z = _proj(Z), _proj(one), _proj(np.hstack([one, Z]))
    Zc = Z - Z.mean(0)
    Tc = T - T.mean(0, keepdims=True)
    sZ = float(np.sum((PZ @ T) ** 2) / tot)
    s1 = float(np.sum((P1 @ T) ** 2) / tot)
    s1Z = float(np.sum((P1Z @ T) ** 2) / tot)
    sZc = float(np.sum((_proj(Zc) @ Tc) ** 2) / tot)
    resid = np.linalg.svd(T - P1Z @ T, compute_uv=False)
    return {"PZ_share": sZ, "P1_share_common": s1, "P1Z_share": s1Z, "affine_minus_linear_projection_gain": s1Z - sZ,
            "centered_energy_share": float(np.sum(Tc ** 2) / tot), "centered_in_colZc_share": sZc, "residual_share": 1.0 - s1Z,
            "orthogonal_identity_error": abs(s1Z - (s1 + sZc)), "residual_singular_values_top5": resid[:5].tolist()}


def oddness(z, B, c, v, sigma=None):
    """Orientation check at a FIXED recipient gradient v: reversing (d, z) -> (-d, -z) must negate a derivative.
    Untagged affine: pred(-z) + pred(z) = 2 c.v (oddness error). Tagged (sigma * c + z B) with sigma supplied as
    experiment metadata: exact oddness."""
    if sigma is None:
        return predict(-z, B, c, v) + predict(z, B, c, v)
    return float((-sigma * c - z @ B) @ v) + float((sigma * c + z @ B) @ v)


def split(cell_of, held_cell):
    """Source-cell hold-out: all seeds of the held cell leave BOTH the donor rows and the recipient columns."""
    h = [k for k, c in enumerate(cell_of) if c == held_cell]
    tr = [k for k, c in enumerate(cell_of) if c != held_cell]
    return tr, h


def cold_score(coef, Z, U, s):
    """Cold numerical scorer from saved coefficients only: Q (D x r), B_linear, B_affine, c_affine, dbar (r,).
    Returns native-unit diagonal predictions for rows (z_k, u_k): dict model -> array."""
    V = U @ coef["Q"]
    rs = np.sqrt(s)
    return {"ZERO": np.zeros(len(Z)),
            "COMMON": (V @ coef["dbar"]) / rs,
            "LINEAR": np.einsum("ia,ab,ib->i", Z, coef["B_linear"], V) / rs,
            "AFFINE": (np.einsum("ia,ab,ib->i", Z, coef["B_affine"], V) + V @ coef["c_affine"]) / rs}
