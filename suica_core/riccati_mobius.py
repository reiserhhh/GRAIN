#!/usr/bin/env python3
"""Constrained Riccati-Mobius ZOH flows, dual-pathway episodic retention, dynamic trajectory LP bounds,
and multi-window transition contrast stacking for standalone GRAIN pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linprog


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x_clip = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.where(x > 30.0, x, np.log1p(np.exp(np.clip(x, -40.0, 30.0))))


@dataclass
class ConstrainedRiccatiCell:
    """Exact bounded contractive Riccati-Mobius layer with O(log T) parallel prefix scan."""

    input_dim: int
    hidden_dim: int
    bound_B: float = 1.0
    s_min: float = 0.05
    seed: int = 431
    enable_quadratic: bool = True
    enable_input_speed: bool = True

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        scale = 1.0 / np.sqrt(max(1, self.input_dim))
        self.W_alpha = rng.normal(0.0, scale, (self.input_dim, self.hidden_dim))
        self.b_alpha = np.zeros(self.hidden_dim)
        self.W_beta = rng.normal(0.0, scale, (self.input_dim, self.hidden_dim))
        self.b_beta = np.zeros(self.hidden_dim)
        self.W_gamma = (
            rng.normal(0.0, scale, (self.input_dim, self.hidden_dim))
            if self.enable_quadratic
            else np.zeros((self.input_dim, self.hidden_dim))
        )
        self.b_gamma = np.zeros(self.hidden_dim)
        self.W_eps = (
            rng.normal(0.0, scale, (self.input_dim, self.hidden_dim))
            if self.enable_input_speed
            else np.zeros((self.input_dim, self.hidden_dim))
        )
        self.b_eps = np.zeros(self.hidden_dim)

    def coefficients(
        self, u: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Map inputs u (..., input_dim) to constrained Riccati (alpha, beta, gamma, eps, omega)."""
        u = np.asarray(u, dtype=np.float64)
        alpha_hat = u @ self.W_alpha + self.b_alpha
        beta_hat = u @ self.W_beta + self.b_beta
        gamma_hat = (
            u @ self.W_gamma + self.b_gamma
            if self.enable_quadratic
            else np.zeros_like(alpha_hat)
        )
        eps_hat = (
            u @ self.W_eps + self.b_eps
            if self.enable_input_speed
            else np.broadcast_to(self.b_eps, alpha_hat.shape).copy()
        )
        B = self.bound_B
        eps = _sigmoid(eps_hat)
        s = self.s_min + _softplus(beta_hat)
        gamma = gamma_hat if self.enable_quadratic else np.zeros_like(alpha_hat)
        abs_g = np.abs(gamma)
        alpha = (s * B + abs_g * (B**2)) * np.tanh(alpha_hat)
        beta = -(s + 2.0 * abs_g * B)
        disc = np.maximum(0.25 * (s**2), 0.25 * (beta**2) - alpha * gamma)
        omega = eps * np.sqrt(disc)
        return alpha, beta, gamma, eps, omega

    def step_matrices(self, u: np.ndarray, dt: np.ndarray | float) -> np.ndarray:
        """Compute normalized 2x2 exact Mobius flow matrices M(dt) of shape (..., D, 2, 2)."""
        alpha, beta, gamma, eps, omega = self.coefficients(u)
        dt_arr = np.asarray(dt, dtype=np.float64)
        if dt_arr.ndim == u.ndim - 1:
            dt_arr = dt_arr[..., None]
        dt_arr = np.broadcast_to(dt_arr, alpha.shape)

        z = np.maximum(0.0, omega * dt_arr)
        exp_neg2z = np.exp(-2.0 * z)
        c_norm = 0.5 * (1.0 + exp_neg2z)
        small = z < 1e-6
        s_over_w = np.empty_like(z)
        s_over_w[~small] = 0.5 * (1.0 - exp_neg2z[~small]) / omega[~small]
        if np.any(small):
            zs = z[small]
            dts = dt_arr[small]
            s_over_w[small] = dts * np.exp(-zs) * (1.0 + (zs**2) / 6.0)

        half_eb = 0.5 * eps * beta
        a = c_norm + s_over_w * half_eb
        b = s_over_w * (eps * alpha)
        c = s_over_w * (-eps * gamma)
        d = c_norm - s_over_w * half_eb

        M = np.empty(alpha.shape + (2, 2), dtype=np.float64)
        M[..., 0, 0] = a
        M[..., 0, 1] = b
        M[..., 1, 0] = c
        M[..., 1, 1] = d
        scale = np.maximum(np.max(np.abs(M), axis=(-2, -1), keepdims=True), 1e-30)
        return M / scale

    @staticmethod
    def apply_mobius(M: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Apply 2x2 projective matrices M (..., D, 2, 2) to state x (..., D)."""
        num = M[..., 0, 0] * x + M[..., 0, 1]
        den = M[..., 1, 0] * x + M[..., 1, 1]
        return num / den

    def rollout_sequential(
        self, u_seq: np.ndarray, dt_seq: np.ndarray | float, x0: np.ndarray | None = None
    ) -> np.ndarray:
        """Sequential exact Mobius step-by-step rollout."""
        squeeze_batch = False
        if u_seq.ndim == 2:
            u_seq = u_seq[None, ...]
            squeeze_batch = True
        B_sz, T, _ = u_seq.shape
        dt_arr = np.asarray(dt_seq, dtype=np.float64)
        if dt_arr.ndim == 0:
            dt_arr = np.full((B_sz, T), float(dt_arr))
        elif dt_arr.ndim == 1:
            dt_arr = np.broadcast_to(dt_arr[None, :], (B_sz, T))
        M_all = self.step_matrices(u_seq, dt_arr)
        x = (
            np.zeros((B_sz, self.hidden_dim), dtype=np.float64)
            if x0 is None
            else np.broadcast_to(np.asarray(x0, dtype=np.float64), (B_sz, self.hidden_dim)).copy()
        )
        xs = np.empty((B_sz, T, self.hidden_dim), dtype=np.float64)
        for t in range(T):
            x = self.apply_mobius(M_all[:, t], x)
            xs[:, t] = x
        return xs[0] if squeeze_batch else xs

    def rollout_parallel_scan(
        self, u_seq: np.ndarray, dt_seq: np.ndarray | float, x0: np.ndarray | None = None
    ) -> np.ndarray:
        """Associative Kogge-Stone parallel prefix scan of 2x2 Mobius matrices in O(log T) depth."""
        squeeze_batch = False
        if u_seq.ndim == 2:
            u_seq = u_seq[None, ...]
            squeeze_batch = True
        B_sz, T, _ = u_seq.shape
        dt_arr = np.asarray(dt_seq, dtype=np.float64)
        if dt_arr.ndim == 0:
            dt_arr = np.full((B_sz, T), float(dt_arr))
        elif dt_arr.ndim == 1:
            dt_arr = np.broadcast_to(dt_arr[None, :], (B_sz, T))

        P = self.step_matrices(u_seq, dt_arr).copy()
        offset = 1
        while offset < T:
            left = P[:, :-offset]
            right = P[:, offset:]
            prod = np.matmul(right, left)
            scale = np.maximum(np.max(np.abs(prod), axis=(-2, -1), keepdims=True), 1e-30)
            P[:, offset:] = prod / scale
            offset *= 2

        x0_arr = (
            np.zeros((B_sz, self.hidden_dim), dtype=np.float64)
            if x0 is None
            else np.broadcast_to(np.asarray(x0, dtype=np.float64), (B_sz, self.hidden_dim))
        )
        xs = self.apply_mobius(P, x0_arr[:, None, :])
        return xs[0] if squeeze_batch else xs


class DualPathwayRiccatiObserver:
    """Pairs fast continuous Riccati flow (Pathway A) with slow episodic retrieval (Pathway B)."""

    def __init__(
        self,
        dim: int,
        seed: int = 431,
        episodic_weight: float = 0.25,
        episodic_temperature: float = 4.0,
        gamma_scale: float = 0.35,
    ) -> None:
        self.dim = dim
        self.episodic_weight = episodic_weight
        self.episodic_temperature = episodic_temperature
        self.cell = ConstrainedRiccatiCell(
            input_dim=dim,
            hidden_dim=dim,
            bound_B=1.0,
            s_min=0.25,
            seed=seed,
            enable_quadratic=True,
            enable_input_speed=True,
        )
        self.cell.W_alpha = 0.9 * np.eye(dim)
        self.cell.b_alpha = np.zeros(dim)
        self.cell.W_beta = np.zeros((dim, dim))
        self.cell.b_beta = np.full(dim, 0.2)
        self.cell.W_gamma = gamma_scale * np.eye(dim)
        self.cell.b_gamma = np.zeros(dim)
        self.cell.W_eps = 0.5 * np.eye(dim)
        self.cell.b_eps = np.zeros(dim)

    def rollout_with_idle_gaps(
        self,
        x_events: np.ndarray,
        gaps: np.ndarray,
        pulse_dt: float = 1.0,
        idle_substeps: int = 1,
        mask_first: bool = False,
        enable_episodic: bool = True,
    ) -> np.ndarray:
        """Evolve simplex-valued observations over causal inter-event gaps + episodic memory."""
        x_events = np.asarray(x_events, dtype=np.float64)
        gaps = np.asarray(gaps, dtype=np.float64)
        T, D = x_events.shape
        x_eff = x_events.copy()
        if mask_first:
            x_eff[0] = 0.0

        zero_in = np.zeros((1, D), dtype=np.float64)
        h = np.zeros((1, D), dtype=np.float64)
        fast_traj = np.empty((T, D), dtype=np.float64)
        ep_traj = np.empty((T, D), dtype=np.float64)

        for t in range(T):
            dt_idle = float(gaps[t])
            if dt_idle > 0.0:
                sub_dt = dt_idle / max(1, idle_substeps)
                M_idle = self.cell.step_matrices(zero_in, np.full((1, D), sub_dt))
                for _ in range(max(1, idle_substeps)):
                    h = self.cell.apply_mobius(M_idle, h)
            u_t = x_eff[t : t + 1]
            M_ev = self.cell.step_matrices(u_t, np.full((1, D), pulse_dt))
            h = self.cell.apply_mobius(M_ev, h)
            fast_traj[t] = h[0]

            if enable_episodic and t > 0:
                hist = x_eff[:t]
                q = x_eff[t]
                q_norm = np.linalg.norm(q)
                h_norm = np.linalg.norm(hist, axis=1)
                denom = np.maximum(q_norm * h_norm, 1e-12)
                sims = (hist @ q) / denom
                if mask_first:
                    sims[0] = -1e9
                logits = self.episodic_temperature * sims
                logits -= np.max(logits)
                weights = np.exp(logits)
                if mask_first and np.all(h_norm == 0):
                    ep_traj[t] = 0.0
                else:
                    if mask_first:
                        weights[0] = 0.0
                    w_sum = weights.sum()
                    ep_traj[t] = (weights @ hist) / w_sum if w_sum > 0 else 0.0
            else:
                ep_traj[t] = x_eff[t]

        combined = fast_traj + self.episodic_weight * ep_traj if enable_episodic else fast_traj
        scaled = 2.0 * combined
        scaled -= np.max(scaled, axis=1, keepdims=True)
        exps = np.exp(scaled)
        return exps / np.sum(exps, axis=1, keepdims=True)


def estimate_cross_view_lag1_rates(
    y1: np.ndarray, y2: np.ndarray, rank: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Legacy clipped lag-1 moment ratios (small/negative denominators fall back to zero).

    Use dual_view_increment.moment_screen before interpreting unit-loading
    common/private rates. A fallback zero is not evidence of zero persistence.
    """
    rc = np.zeros(rank, dtype=np.float64)
    rp1 = np.zeros(rank, dtype=np.float64)
    rp2 = np.zeros(rank, dtype=np.float64)
    for j in range(rank):
        a0c = y1[:, :-1, j].ravel() - np.mean(y1[:, :-1, j])
        a1c = y1[:, 1:, j].ravel() - np.mean(y1[:, 1:, j])
        b0c = y2[:, :-1, j].ravel() - np.mean(y2[:, :-1, j])
        b1c = y2[:, 1:, j].ravel() - np.mean(y2[:, 1:, j])
        var_c = float(np.mean(a0c * b0c))
        cov_c1 = 0.5 * float(np.mean(a1c * b0c) + np.mean(b1c * a0c))
        rc[j] = float(np.clip(cov_c1 / var_c, -0.95, 0.95)) if var_c > 1e-4 else 0.0
        var_p1 = float(np.mean(a0c * a0c)) - var_c
        cov_p1 = float(np.mean(a1c * a0c)) - cov_c1
        rp1[j] = float(np.clip(cov_p1 / var_p1, -0.95, 0.95)) if var_p1 > 1e-4 else 0.0
        var_p2 = float(np.mean(b0c * b0c)) - var_c
        cov_p2 = float(np.mean(b1c * b0c)) - cov_c1
        rp2[j] = float(np.clip(cov_p2 / var_p2, -0.95, 0.95)) if var_p2 > 1e-4 else 0.0
    return rc, rp1, rp2


def solve_axis_trajectory_lp(
    y1_t: np.ndarray,
    y2_t: np.ndarray,
    r_c: float,
    r_p1: float,
    r_p2: float,
    D: float,
    E1: float,
    E2: float,
    static_half_widths: np.ndarray | None,
    c_bound: float = 20.0,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Solve exact HiGHS min/max trajectory polytope bounds on c_0..c_{T-1}."""
    T = len(y1_t)
    A_ub: list[np.ndarray] = []
    b_ub: list[float] = []

    def add_abs_bound(row: np.ndarray, target: float, tol: float) -> None:
        A_ub.append(row)
        b_ub.append(target + tol)
        A_ub.append(-row)
        b_ub.append(-target + tol)

    for t in range(T - 1):
        row_c = np.zeros(T, dtype=np.float64)
        row_c[t + 1] = 1.0
        row_c[t] = -r_c
        add_abs_bound(row_c, 0.0, D)

        row_p1 = np.zeros(T, dtype=np.float64)
        row_p1[t + 1] = 1.0
        row_p1[t] = -r_p1
        add_abs_bound(row_p1, float(y1_t[t + 1] - r_p1 * y1_t[t]), E1)

        row_p2 = np.zeros(T, dtype=np.float64)
        row_p2[t + 1] = 1.0
        row_p2[t] = -r_p2
        add_abs_bound(row_p2, float(y2_t[t + 1] - r_p2 * y2_t[t]), E2)

    if static_half_widths is not None:
        c_mid = 0.5 * (y1_t + y2_t)
        for t in range(T):
            hw = float(static_half_widths[t])
            if hw < 0.0:
                return None
            row_s = np.zeros(T, dtype=np.float64)
            row_s[t] = 1.0
            add_abs_bound(row_s, float(c_mid[t]), hw)

    A_arr = np.asarray(A_ub, dtype=np.float64) if A_ub else None
    b_arr = np.asarray(b_ub, dtype=np.float64) if b_ub else None
    bounds = [(-c_bound, c_bound)] * T

    lo = np.empty(T, dtype=np.float64)
    hi = np.empty(T, dtype=np.float64)
    for t in range(T):
        obj = np.zeros(T, dtype=np.float64)
        obj[t] = 1.0
        r_min = linprog(obj, A_ub=A_arr, b_ub=b_arr, bounds=bounds, method="highs")
        if r_min.status == 2:
            return None
        if not r_min.success:
            raise ArithmeticError(r_min.message)
        r_max = linprog(-obj, A_ub=A_arr, b_ub=b_arr, bounds=bounds, method="highs")
        if r_max.status == 2:
            return None
        if not r_max.success:
            raise ArithmeticError(r_max.message)
        lo[t] = float(r_min.x[t])
        hi[t] = float(r_max.x[t])
    return lo, hi


def stack_transition_contrasts(
    observations: ArrayLike,
    common_maps: ArrayLike,
    private_maps: ArrayLike,
    *,
    common_drive: ArrayLike | None = None,
    private_drive: ArrayLike | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return (A, b) with b = A @ common_initial in the exact multi-window affine case."""
    y = np.asarray(observations, dtype=np.float64)
    a = np.asarray(common_maps, dtype=np.float64)
    p = np.asarray(private_maps, dtype=np.float64)
    if y.ndim != 3 or y.shape[0] < 2 or not all(y.shape):
        raise ValueError("expected observations with at least two frames, one reader and one coordinate")
    frames, readers, d = y.shape
    steps = frames - 1
    if a.shape != (steps, d, d) or p.shape != (steps, readers, d, d):
        raise ValueError("transition maps must match observation steps, readers and coordinates")
    g = np.zeros((steps, d)) if common_drive is None else np.asarray(common_drive, dtype=np.float64)
    f = np.zeros(y[1:].shape) if private_drive is None else np.asarray(private_drive, dtype=np.float64)
    if g.shape != (steps, d) or f.shape != y[1:].shape:
        raise ValueError("drive shapes must match their respective states")
    if not all(np.isfinite(v).all() for v in (y, a, p, g, f)):
        raise ValueError("observations and calibration must be finite")

    phi = np.eye(d)
    offset = np.zeros(d)
    blocks = []
    residuals = []
    for t in range(steps):
        gap = a[t] - p[t]
        blocks.append((gap @ phi).reshape(-1, d))
        b = y[t + 1] - (p[t] @ y[t, :, :, None])[..., 0] - g[t] - f[t] - gap @ offset
        residuals.append(b.ravel())
        phi = a[t] @ phi
        offset = a[t] @ offset + g[t]
    return np.vstack(blocks), np.concatenate(residuals)


__all__ = [
    "ConstrainedRiccatiCell",
    "DualPathwayRiccatiObserver",
    "estimate_cross_view_lag1_rates",
    "solve_axis_trajectory_lp",
    "stack_transition_contrasts",
]
