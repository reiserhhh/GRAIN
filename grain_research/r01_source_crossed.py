"""R01 source-crossed derivative design (packet 24 B).

T_ij = d_i . u_j with u_j = sqrt(s) J_j: exact cached DIRECTIONAL DERIVATIVES of recipient j's query score along donor
i's activation shift (same head/layer/query-suffix coordinates). NOT new independent samples, finite native patch
outcomes, or labels produced by a fitted W. All designs use the mean-row loss
    (1/m) sum_rows (z_i' W u_j - T_ij)^2 + lambda ||W||_F^2,  lambda = 1/n  (n = diagonal training pairs),
solved in an orthonormal basis Q of the training-recipient U row space (W = B Q', U = V Q').
"""
from __future__ import annotations

import numpy as np

RANK_TOL = 1e-10


def orth_basis(U):
    """U (n x D) = V Q' with Q (D x r) orthonormal; r = numerical rank."""
    L, S, Rt = np.linalg.svd(U, full_matrices=False)
    r = int((S > S.max() * RANK_TOL).sum())
    Q = Rt[:r].T
    return Q, U @ Q, S


def fit_rows(Z, V, rows, T, lam):
    """Dual ridge for an explicit row list [(i, j)] with targets T[i, j]; returns B (p x r)."""
    rows = np.asarray(rows)
    zi, vj = Z[rows[:, 0]], V[rows[:, 1]]
    t = T[rows[:, 0], rows[:, 1]]
    m = len(rows)
    K = (zi @ zi.T) * (vj @ vj.T)
    c = np.linalg.solve(K + m * lam * np.eye(m), t)
    return (zi * c[:, None]).T @ vj


def fit_full(Z, V, T, lam):
    """Full cross (all n x n rows): (Z'Z) B (V'V) + m lam B = Z' T V, m = n^2; eigen-factor (Sylvester) solution."""
    m = Z.shape[0] * V.shape[0]
    a, P = np.linalg.eigh(Z.T @ Z)
    c, R = np.linalg.eigh(V.T @ V)
    Bt = (P.T @ Z.T @ T @ V @ R) / (a[:, None] * c[None, :] + m * lam)
    return P @ Bt @ R.T


def fit_full_primal(Z, V, T, lam):
    """Explicit primal ridge on all n^2 rows (check only): features kron(v_j, z_i)."""
    n, p = Z.shape; r = V.shape[1]
    Phi = np.einsum("jb,ia->ijba", V, Z).reshape(n * n, r * p)  # row (i,j), column (b,a) -> B[a,b]
    t = T.reshape(-1)
    m = n * n
    x = np.linalg.solve(Phi.T @ Phi + m * lam * np.eye(r * p), Phi.T @ t)
    return x.reshape(r, p).T


def loss_grad(Z, V, rows, T, B, lam):
    rows = np.asarray(rows)
    zi, vj = Z[rows[:, 0]], V[rows[:, 1]]
    e = np.einsum("ra,ab,rb->r", zi, B, vj) - T[rows[:, 0], rows[:, 1]]
    m = len(rows)
    return float(np.abs((2 / m) * np.einsum("r,ra,rb->ab", e, zi, vj) + 2 * lam * B).max())


def predict(z, B, v):
    return float(z @ B @ v)


def euclid(Z, D):
    """Identity-metric donor regression reference: mean_rows ||z'W - d||^2 + (1/n)||W||^2 == summed alpha 1."""
    return np.linalg.solve(Z.T @ Z + np.eye(Z.shape[1]), Z.T @ D)


def pz_share(Z, T):
    """Share of T lying in col(Z) (T = Z A needs every COLUMN of T in col(Z)); rank-aware pseudoinverse."""
    Pz = Z @ np.linalg.pinv(Z, rcond=RANK_TOL)
    num = np.linalg.norm(Pz @ T) ** 2; den = np.linalg.norm(T) ** 2
    return float(num / den), np.linalg.svd(T - Pz @ T, compute_uv=False)


def kron_rank(Z, U):
    sz = np.linalg.svd(Z, compute_uv=False); su = np.linalg.svd(U, compute_uv=False)
    rz = int((sz > sz.max() * RANK_TOL).sum()); ru = int((su > su.max() * RANK_TOL).sum())
    return rz, ru, rz * ru


def cyclic_rows(train_cells, cell_of, seed_of):
    """Donor in training cell k (sorted) -> recipient in cell k+1 with the same seed label; all donors/recipients once."""
    cells = sorted(set(train_cells))
    nxt = {c: cells[(k + 1) % len(cells)] for k, c in enumerate(cells)}
    idx = {(cell_of[i], seed_of[i]): i for i in range(len(cell_of))}
    return [(i, idx[(nxt[cell_of[i]], seed_of[i])]) for i in range(len(cell_of))]


def random_rows(train_cells, cell_of, seed_of, seed):
    """One fixed-seed cell derangement + within-cell seed permutation; no same-cell match; all donors/recipients once."""
    rng = np.random.default_rng(seed)
    cells = sorted(set(train_cells))
    while True:
        perm = rng.permutation(len(cells))
        if not np.any(perm == np.arange(len(cells))):
            break
    to_cell = {c: cells[perm[k]] for k, c in enumerate(cells)}
    seeds = sorted(set(seed_of))
    seed_perm = {c: dict(zip(seeds, [seeds[x] for x in rng.permutation(len(seeds))])) for c in cells}
    idx = {(cell_of[i], seed_of[i]): i for i in range(len(cell_of))}
    return [(i, idx[(to_cell[cell_of[i]], seed_perm[cell_of[i]][seed_of[i]])]) for i in range(len(cell_of))]
