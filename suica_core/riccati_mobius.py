#!/usr/bin/env python3
"""Constrained Riccati-Mobius ZOH flows, dual-pathway episodic retention, dynamic trajectory LP bounds,
and multi-window transition contrast stacking for standalone GRAIN pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linprog
from scipy.sparse import csr_matrix


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

    def event_path(
        self,
        events,
        times,
        *,
        start,
        cutoff,
        pulse=1.0,
        x0=None,
        clock="jump",
        parallel=True,
    ):
        """State after each retained arrival and at cutoff. Equal times follow supplied order.

        jump: instantaneous readout update of strength pulse, all wall time is idle.
        ending_pulse: the pulse occupies up to pulse of the preceding gap; short
        gaps still give one full update in operational (not wall-clock) time.
        """
        u, t = np.asarray(events, float), np.asarray(times, float)
        x = np.zeros(self.hidden_dim) if x0 is None else np.asarray(x0, float)
        if (
            u.shape != (len(t), self.input_dim)
            or t.ndim != 1
            or not np.isfinite(t).all()
            or not np.isfinite([start, cutoff, pulse]).all()
            or start > cutoff
            or pulse <= 0
            or np.any(np.diff(t) < 0)
            or np.any(t < start)
            or x.shape != (self.hidden_dim,)
            or not np.isfinite(x).all()
            or np.any(np.abs(x) > self.bound_B)
            or clock not in ("jump", "ending_pulse")
        ):
            raise ValueError("invalid event path, clock or initial state")
        k = np.searchsorted(t, cutoff, side="right")
        t, u = t[:k], u[:k]
        if not np.isfinite(u).all():
            raise ValueError("nonfinite arrival")
        gap = np.diff(t, prepend=start)
        idle = gap if clock == "jump" else np.maximum(0, gap - pulse)
        zero = np.zeros(self.input_dim)
        if k:
            matrices = self.step_matrices(u, pulse) @ self.step_matrices(
                np.zeros_like(u), idle
            )
            if parallel:
                stride = 1
                while stride < k:
                    prod = matrices[stride:] @ matrices[:-stride]
                    matrices[stride:] = prod / np.max(
                        np.abs(prod), axis=(-2, -1), keepdims=True
                    )
                    stride *= 2
                states = self.apply_mobius(matrices, x)
            else:
                states = np.empty((k, self.hidden_dim))
                for i, M in enumerate(matrices):
                    x = self.apply_mobius(M, x)
                    states[i] = x
            x = states[-1]
        else:
            states = np.empty((0, self.hidden_dim))
        end = self.apply_mobius(
            self.step_matrices(zero, cutoff - (t[-1] if k else start)), x
        )
        return states, end

    def window_delta_at_cutoff(self, events, times, *, window, cutoff, **kwargs):
        """Remove arrivals in the closed recent window, holding cutoff and start fixed."""
        if not np.isfinite(window) or window < 0:
            raise ValueError("invalid window")
        _, end = self.event_path(events, times, cutoff=cutoff, **kwargs)
        t = np.asarray(times)
        keep = t < cutoff - window
        if not np.any((~keep) & (t <= cutoff)):
            return np.zeros_like(end)
        _, base = self.event_path(
            np.asarray(events)[keep], t[keep], cutoff=cutoff, **kwargs
        )
        return end - base

    def rollout_sequential(
        self,
        u_seq: np.ndarray,
        dt_seq: np.ndarray | float,
        x0: np.ndarray | None = None,
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
            else np.broadcast_to(
                np.asarray(x0, dtype=np.float64), (B_sz, self.hidden_dim)
            ).copy()
        )
        xs = np.empty((B_sz, T, self.hidden_dim), dtype=np.float64)
        for t in range(T):
            x = self.apply_mobius(M_all[:, t], x)
            xs[:, t] = x
        return xs[0] if squeeze_batch else xs

    def rollout_parallel_scan(
        self,
        u_seq: np.ndarray,
        dt_seq: np.ndarray | float,
        x0: np.ndarray | None = None,
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
            scale = np.maximum(
                np.max(np.abs(prod), axis=(-2, -1), keepdims=True), 1e-30
            )
            P[:, offset:] = prod / scale
            offset *= 2

        x0_arr = (
            np.zeros((B_sz, self.hidden_dim), dtype=np.float64)
            if x0 is None
            else np.broadcast_to(
                np.asarray(x0, dtype=np.float64), (B_sz, self.hidden_dim)
            )
        )
        xs = self.apply_mobius(P, x0_arr[:, None, :])
        return xs[0] if squeeze_batch else xs

    def impulse_step_matrices(
        self,
        u: np.ndarray,
        dt_gap: np.ndarray | float,
        *,
        pulse_tau: float = 1.0,
        subtract_pulse: bool = True,
        is_first_step: np.ndarray | bool = False,
    ) -> np.ndarray:
        """Two-stage exact Idle-Then-Impulse 2x2 Mobius matrices M(u, tau) @ M(0, dt_idle)."""
        if pulse_tau <= 0.0:
            raise ValueError("pulse_tau must be positive")
        u_arr = np.asarray(u, dtype=np.float64)
        dt_arr = np.asarray(dt_gap, dtype=np.float64)
        if dt_arr.ndim == u_arr.ndim - 1:
            dt_arr = dt_arr[..., None]
        first_arr = np.asarray(is_first_step, dtype=bool)
        if first_arr.ndim > 0 and first_arr.ndim == u_arr.ndim - 1:
            first_arr = first_arr[..., None]
        target_shape = u_arr.shape[:-1] + (self.hidden_dim,)
        dt_b = np.broadcast_to(dt_arr, target_shape)
        first_b = np.broadcast_to(first_arr, target_shape)
        idle_raw = (
            np.maximum(0.0, dt_b - float(pulse_tau))
            if subtract_pulse
            else np.maximum(0.0, dt_b)
        )
        dt_idle = np.where(first_b, 0.0, idle_raw)
        M_idle = self.step_matrices(np.zeros_like(u_arr), dt_idle)
        M_pulse = self.step_matrices(u_arr, float(pulse_tau))
        prod = np.matmul(M_pulse, M_idle)
        scale = np.maximum(np.max(np.abs(prod), axis=(-2, -1), keepdims=True), 1e-30)
        return prod / scale

    def _prepare_impulse_inputs(
        self,
        u_seq: np.ndarray,
        dt_gaps: np.ndarray | float,
        *,
        pulse_tau: float,
        subtract_pulse: bool,
        first_has_idle: bool,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        squeeze_batch = False
        u_arr = np.asarray(u_seq, dtype=np.float64)
        if u_arr.ndim == 2:
            u_arr = u_arr[None, ...]
            squeeze_batch = True
        B_sz, T, _ = u_arr.shape
        dt_arr = np.asarray(dt_gaps, dtype=np.float64)
        if dt_arr.ndim == 0:
            dt_arr = np.full((B_sz, T), float(dt_arr))
        elif dt_arr.ndim == 1:
            dt_arr = np.broadcast_to(dt_arr[None, :], (B_sz, T))
        first_mask = np.zeros((B_sz, T), dtype=bool)
        if not first_has_idle and T > 0:
            first_mask[:, 0] = True
        M_all = self.impulse_step_matrices(
            u_arr,
            dt_arr,
            pulse_tau=pulse_tau,
            subtract_pulse=subtract_pulse,
            is_first_step=first_mask,
        )
        return M_all, u_arr, squeeze_batch

    def rollout_impulse_sequential(
        self,
        u_seq: np.ndarray,
        dt_gaps: np.ndarray | float,
        *,
        pulse_tau: float = 1.0,
        subtract_pulse: bool = True,
        first_has_idle: bool = False,
        dt_cut: np.ndarray | float | None = None,
        x0: np.ndarray | None = None,
        return_terminal: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Sequential two-stage Idle-Then-Impulse Mobius rollout (with optional T_cut decay)."""
        M_all, u_arr, squeeze_batch = self._prepare_impulse_inputs(
            u_seq,
            dt_gaps,
            pulse_tau=pulse_tau,
            subtract_pulse=subtract_pulse,
            first_has_idle=first_has_idle,
        )
        B_sz, T, _, _, _ = M_all.shape
        x = (
            np.zeros((B_sz, self.hidden_dim), dtype=np.float64)
            if x0 is None
            else np.broadcast_to(
                np.asarray(x0, dtype=np.float64), (B_sz, self.hidden_dim)
            ).copy()
        )
        xs = np.empty((B_sz, T, self.hidden_dim), dtype=np.float64)
        for t in range(T):
            x = self.apply_mobius(M_all[:, t], x)
            xs[:, t] = x
        xs_out = xs[0] if squeeze_batch else xs
        if not return_terminal:
            return xs_out
        if dt_cut is None:
            x_term = x
        else:
            zero_u = np.zeros((B_sz, u_arr.shape[-1]), dtype=np.float64)
            M_cut = self.step_matrices(zero_u, np.asarray(dt_cut, dtype=np.float64))
            x_term = self.apply_mobius(M_cut, x)
        return xs_out, (x_term[0] if squeeze_batch else x_term)

    def rollout_impulse_parallel_scan(
        self,
        u_seq: np.ndarray,
        dt_gaps: np.ndarray | float,
        *,
        pulse_tau: float = 1.0,
        subtract_pulse: bool = True,
        first_has_idle: bool = False,
        dt_cut: np.ndarray | float | None = None,
        x0: np.ndarray | None = None,
        return_terminal: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Associative O(log T) prefix scan for Idle-Then-Impulse Mobius flows."""
        M_all, u_arr, squeeze_batch = self._prepare_impulse_inputs(
            u_seq,
            dt_gaps,
            pulse_tau=pulse_tau,
            subtract_pulse=subtract_pulse,
            first_has_idle=first_has_idle,
        )
        B_sz, T, _, _, _ = M_all.shape
        P = M_all.copy()
        offset = 1
        while offset < T:
            left = P[:, :-offset]
            right = P[:, offset:]
            prod = np.matmul(right, left)
            scale = np.maximum(
                np.max(np.abs(prod), axis=(-2, -1), keepdims=True), 1e-30
            )
            P[:, offset:] = prod / scale
            offset *= 2

        x0_arr = (
            np.zeros((B_sz, self.hidden_dim), dtype=np.float64)
            if x0 is None
            else np.broadcast_to(
                np.asarray(x0, dtype=np.float64), (B_sz, self.hidden_dim)
            )
        )
        xs = self.apply_mobius(P, x0_arr[:, None, :])
        xs_out = xs[0] if squeeze_batch else xs
        if not return_terminal:
            return xs_out
        if T == 0:
            x_term = x0_arr.copy()
        elif dt_cut is None:
            x_term = xs[:, -1]
        else:
            zero_u = np.zeros((B_sz, u_arr.shape[-1]), dtype=np.float64)
            M_cut = self.step_matrices(zero_u, np.asarray(dt_cut, dtype=np.float64))
            x_term = self.apply_mobius(M_cut, xs[:, -1])
        return xs_out, (x_term[0] if squeeze_batch else x_term)

    def counterfactual_window_delta(
        self,
        u_seq: np.ndarray,
        timestamps: np.ndarray,
        *,
        t_cut: float,
        window_W: float,
        pulse_tau: float = 1.0,
        x0: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Exact counterfactual arrival innovation in (t_cut - window_W, t_cut]; 0.0 when empty."""
        u_arr = np.asarray(u_seq, dtype=np.float64)
        ts = np.asarray(timestamps, dtype=np.float64)
        if u_arr.ndim != 2 or ts.ndim != 1 or len(u_arr) != len(ts):
            raise ValueError("expected 2D u_seq (T, input_dim) and 1D timestamps (T,)")
        if (
            not np.isfinite(ts).all()
            or not np.isfinite([t_cut, window_W, pulse_tau]).all()
            or window_W < 0
            or pulse_tau <= 0
            or np.any(np.diff(ts) < 0)
        ):
            raise ValueError("timestamps must be monotonically non-decreasing")
        x0_arr = (
            np.zeros(self.hidden_dim, dtype=np.float64)
            if x0 is None
            else np.asarray(x0, dtype=np.float64).copy()
        )
        act_mask = ts <= float(t_cut)
        pre_mask = ts <= (float(t_cut) - float(window_W))

        def _eval_subset(mask: np.ndarray) -> np.ndarray:
            if not np.any(mask):
                idle = float(t_cut - ts[0]) if act_mask.any() else 0.0
                return self.apply_mobius(
                    self.step_matrices(np.zeros(self.input_dim), idle), x0_arr
                )
            t_sub = ts[mask]
            u_sub = u_arr[mask]
            dt_g = np.r_[0.0, np.diff(t_sub)]
            dt_c = max(0.0, float(t_cut) - float(t_sub[-1]))
            _, term = self.rollout_impulse_parallel_scan(
                u_sub,
                dt_g,
                pulse_tau=pulse_tau,
                subtract_pulse=False,
                first_has_idle=False,
                dt_cut=dt_c,
                x0=x0_arr,
                return_terminal=True,
            )
            return term

        x_actual = _eval_subset(act_mask)
        x_cf = _eval_subset(pre_mask)
        if np.array_equal(act_mask, pre_mask):
            x_cf = x_actual.copy()
        return {
            "actual_state": x_actual,
            "counterfactual_state": x_cf,
            "delta": x_actual - x_cf,
        }


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

        combined = (
            fast_traj + self.episodic_weight * ep_traj if enable_episodic else fast_traj
        )
        scaled = 2.0 * combined
        scaled -= np.max(scaled, axis=1, keepdims=True)
        exps = np.exp(scaled)
        return exps / np.sum(exps, axis=1, keepdims=True)


def _weighted_mean(vals: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(weights * vals) / max(float(np.sum(weights)), 1e-12))


def estimate_cross_view_lag1_rates(
    y1: np.ndarray,
    y2: np.ndarray,
    rank: int,
    *,
    counts1: np.ndarray | None = None,
    counts2: np.ndarray | None = None,
    sigma_eps1: np.ndarray | float | None = None,
    sigma_eps2: np.ndarray | float | None = None,
    se1=None,
    se2=None,
    cross_error_cov=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lag-1 moment ratios with optional count-weighted errors-in-variables de-attenuation."""
    if se1 is not None or se2 is not None or cross_error_cov is not None:
        from .stream_precision import lag_rates

        if (
            se1 is None
            or se2 is None
            or not 0 < rank <= np.shape(y1)[-1]
            or any(v is not None for v in (counts1, counts2, sigma_eps1, sigma_eps2))
        ):
            raise ValueError(
                "provide either calibrated paired SEs or count calibration"
            )
        return tuple(
            lag_rates(y1, y2, se1, se2, cross_error_cov=cross_error_cov)[:, :rank]
        )
    if (
        counts1 is None
        and counts2 is None
        and sigma_eps1 is None
        and sigma_eps2 is None
    ):
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
            rp1[j] = (
                float(np.clip(cov_p1 / var_p1, -0.95, 0.95)) if var_p1 > 1e-4 else 0.0
            )
            var_p2 = float(np.mean(b0c * b0c)) - var_c
            cov_p2 = float(np.mean(b1c * b0c)) - cov_c1
            rp2[j] = (
                float(np.clip(cov_p2 / var_p2, -0.95, 0.95)) if var_p2 > 1e-4 else 0.0
            )
        return rc, rp1, rp2

    c1 = (
        np.ones(y1.shape[:2], dtype=np.float64)
        if counts1 is None
        else np.asarray(counts1, dtype=float)
    )
    c2 = (
        np.ones(y2.shape[:2], dtype=np.float64)
        if counts2 is None
        else np.asarray(counts2, dtype=float)
    )
    if any(
        v.shape != y1.shape[:2] or not np.isfinite(v).all() or (v <= 0).any()
        for v in (c1, c2)
    ):
        raise ValueError("positive aligned counts required; missing is not one event")
    w1_lag = np.minimum(c1[:, :-1], c1[:, 1:]) / (
        np.minimum(c1[:, :-1], c1[:, 1:]) + 1.0
    )
    w2_lag = np.minimum(c2[:, :-1], c2[:, 1:]) / (
        np.minimum(c2[:, :-1], c2[:, 1:]) + 1.0
    )
    wc_lag = np.sqrt(w1_lag * w2_lag)
    w1_flat = (w1_lag / np.sum(w1_lag)).ravel()
    w2_flat = (w2_lag / np.sum(w2_lag)).ravel()
    wc_flat = (wc_lag / np.sum(wc_lag)).ravel()
    inv_c1 = (1.0 / c1[:, :-1]).ravel()
    inv_c2 = (1.0 / c2[:, :-1]).ravel()

    rc = np.zeros(rank, dtype=np.float64)
    rp1 = np.zeros(rank, dtype=np.float64)
    rp2 = np.zeros(rank, dtype=np.float64)
    for j in range(rank):
        a0 = y1[:, :-1, j].ravel()
        a1 = y1[:, 1:, j].ravel()
        b0 = y2[:, :-1, j].ravel()
        b1 = y2[:, 1:, j].ravel()
        a0c = a0 - _weighted_mean(a0, w1_flat)
        a1c = a1 - _weighted_mean(a1, w1_flat)
        b0c = b0 - _weighted_mean(b0, w2_flat)
        b1c = b1 - _weighted_mean(b1, w2_flat)

        var_c = float(np.sum(wc_flat * a0c * b0c))
        cov_c1 = 0.5 * float(np.sum(wc_flat * (a1c * b0c + b1c * a0c)))
        rc[j] = float(np.clip(cov_c1 / var_c, -0.95, 0.95)) if var_c > 1e-4 else 0.0

        def _sig_sq(
            supplied: np.ndarray | float | None,
            dev: np.ndarray,
            inv_c: np.ndarray,
            cross_term: np.ndarray,
            col_idx: int = j,
        ) -> float:
            if supplied is not None:
                arr = np.asarray(supplied, dtype=np.float64)
                val = float(arr) if arr.ndim == 0 else float(arr[col_idx])
                return max(0.0, val) ** 2
            if float(np.std(inv_c)) <= 1e-6:
                return 0.0
            resid_var = dev * dev - cross_term
            ic = inv_c - np.mean(inv_c)
            denom = float(np.sum(ic * ic))
            return (
                max(0.0, float(np.sum(ic * (resid_var - np.mean(resid_var))) / denom))
                if denom > 1e-12
                else 0.0
            )

        cross_ab = a0c * b0c
        sig1_sq = _sig_sq(sigma_eps1, a0c, inv_c1, cross_ab)
        sig2_sq = _sig_sq(sigma_eps2, b0c, inv_c2, cross_ab)

        var_c_1 = float(np.sum(w1_flat * cross_ab))
        cov_c1_1 = 0.5 * float(np.sum(w1_flat * (a1c * b0c + b1c * a0c)))
        var_p1 = (
            float(np.sum(w1_flat * a0c * a0c))
            - var_c_1
            - sig1_sq * float(np.sum(w1_flat * inv_c1))
        )
        cov_p1 = float(np.sum(w1_flat * a1c * a0c)) - cov_c1_1
        rp1[j] = float(np.clip(cov_p1 / var_p1, -0.95, 0.95)) if var_p1 > 1e-4 else 0.0

        var_c_2 = float(np.sum(w2_flat * cross_ab))
        cov_c1_2 = 0.5 * float(np.sum(w2_flat * (a1c * b0c + b1c * a0c)))
        var_p2 = (
            float(np.sum(w2_flat * b0c * b0c))
            - var_c_2
            - sig2_sq * float(np.sum(w2_flat * inv_c2))
        )
        cov_p2 = float(np.sum(w2_flat * b1c * b0c)) - cov_c1_2
        rp2[j] = float(np.clip(cov_p2 / var_p2, -0.95, 0.95)) if var_p2 > 1e-4 else 0.0
    return rc, rp1, rp2


def _solve_axis_lp_array_tols(
    y1_t: np.ndarray,
    y2_t: np.ndarray,
    r_c: float,
    r_p1: float,
    r_p2: float,
    D_arr: np.ndarray,
    E1_arr: np.ndarray,
    E2_arr: np.ndarray,
    static_half_widths: np.ndarray | None,
    c_bound: float,
    endpoint_only: bool,
) -> tuple[np.ndarray, np.ndarray] | None:
    T = len(y1_t)
    if not T:
        if endpoint_only:
            raise ValueError("empty trajectory has no endpoint")
        return np.empty(0), np.empty(0)
    n = T - 1
    rates = np.repeat([r_c, r_p1, r_p2], 2)
    signs = np.tile([1.0, -1.0], 3)
    data = np.tile(np.column_stack((-rates * signs, signs)).ravel(), n)
    left = np.repeat(np.arange(n), 6)
    cols = np.column_stack((left, left + 1)).ravel()
    indptr = np.arange(6 * n + 1) * 2
    targets = np.column_stack(
        (np.zeros(n), y1_t[1:] - r_p1 * y1_t[:-1], y2_t[1:] - r_p2 * y2_t[:-1])
    )
    tols = np.column_stack((D_arr, E1_arr, E2_arr))
    b_arr = (np.repeat(targets, 2, axis=1) * signs + np.repeat(tols, 2, axis=1)).ravel()
    if static_half_widths is not None:
        if np.any(np.asarray(static_half_widths) < 0):
            return None
        c_mid = 0.5 * (y1_t + y2_t)
        data = np.r_[data, np.tile([1.0, -1.0], T)]
        cols = np.r_[cols, np.repeat(np.arange(T), 2)]
        indptr = np.r_[indptr, indptr[-1] + np.arange(1, 2 * T + 1)]
        b_arr = np.r_[
            b_arr,
            np.column_stack(
                (c_mid + static_half_widths, -c_mid + static_half_widths)
            ).ravel(),
        ]
    A_arr = (
        csr_matrix((data, cols, indptr), shape=(len(b_arr), T)) if len(b_arr) else None
    )
    rhs = b_arr if A_arr is not None else None
    bounds = [(-c_bound, c_bound)] * T

    positions = range(T - 1, T) if endpoint_only else range(T)
    lo, hi = np.empty(len(positions)), np.empty(len(positions))
    for i, t in enumerate(positions):
        obj = np.zeros(T, dtype=np.float64)
        obj[t] = 1.0
        r_min = linprog(obj, A_ub=A_arr, b_ub=rhs, bounds=bounds, method="highs")
        if r_min.status == 2:
            return None
        if not r_min.success:
            raise ArithmeticError(r_min.message)
        r_max = linprog(-obj, A_ub=A_arr, b_ub=rhs, bounds=bounds, method="highs")
        if r_max.status == 2:
            return None
        if not r_max.success:
            raise ArithmeticError(r_max.message)
        lo[i] = float(r_min.x[t])
        hi[i] = float(r_max.x[t])
    return lo, hi


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
    *,
    endpoint_only: bool = False,
    counts1: np.ndarray | None = None,
    counts2: np.ndarray | None = None,
    sigma_eps1: float = 0.0,
    sigma_eps2: float = 0.0,
    z_alpha: float = 1.6448536269514722,
    return_diagnostics: bool = False,
) -> tuple[np.ndarray, np.ndarray] | None | dict[str, object]:
    """Bound every coordinate with optional count-scaled conformal tolerances (G14)."""
    y1_arr = np.asarray(y1_t, dtype=float)
    y2_arr = np.asarray(y2_t, dtype=float)
    if y1_arr.ndim != 1 or y2_arr.shape != y1_arr.shape:
        raise ValueError("expected two aligned trajectory vectors")
    hw_arr = (
        None
        if static_half_widths is None
        else np.asarray(static_half_widths, dtype=float)
    )
    if hw_arr is not None and hw_arr.shape != y1_arr.shape:
        raise ValueError("one static half width per time step is required")
    T = len(y1_arr)
    n = max(0, T - 1)
    D_base, E1_base, E2_base = (
        np.broadcast_to(np.asarray(v, float), (n,)) for v in (D, E1, E2)
    )
    if (
        not np.isfinite([D_base, E1_base, E2_base]).all()
        or min(np.min(v, initial=0.0) for v in (D_base, E1_base, E2_base)) < 0
    ):
        raise ValueError("invalid transition allowance")

    if counts1 is None and counts2 is None and not return_diagnostics:
        return _solve_axis_lp_array_tols(
            y1_arr,
            y2_arr,
            r_c,
            r_p1,
            r_p2,
            D_base,
            E1_base,
            E2_base,
            hw_arr,
            c_bound,
            endpoint_only,
        )

    c1 = (
        np.ones(T, dtype=float) if counts1 is None else np.asarray(counts1, dtype=float)
    )
    c2 = (
        np.ones(T, dtype=float) if counts2 is None else np.asarray(counts2, dtype=float)
    )
    if any(
        v.shape != (T,) or not np.isfinite(v).all() or (v <= 0).any() for v in (c1, c2)
    ):
        raise ValueError("counts1 and counts2 must match length T")
    se1 = float(sigma_eps1) / np.sqrt(c1)
    se2 = float(sigma_eps2) / np.sqrt(c2)
    E1_eff = np.sqrt(
        E1_base**2
        + (float(z_alpha) ** 2) * (se1[1:] ** 2 + (r_p1**2) * (se1[:-1] ** 2))
    )
    E2_eff = np.sqrt(
        E2_base**2
        + (float(z_alpha) ** 2) * (se2[1:] ** 2 + (r_p2**2) * (se2[:-1] ** 2))
    )
    hw_eff = (
        None
        if hw_arr is None
        else np.sqrt(
            np.maximum(0.0, hw_arr) ** 2
            + ((0.5 * float(z_alpha)) ** 2) * (se1**2 + se2**2)
        )
    )
    adapted = _solve_axis_lp_array_tols(
        y1_arr,
        y2_arr,
        r_c,
        r_p1,
        r_p2,
        D_base,
        E1_eff,
        E2_eff,
        hw_eff,
        c_bound,
        endpoint_only,
    )
    if not return_diagnostics:
        return adapted
    if adapted is None:
        return {"status": "structural_infeasible", "bounds": None}
    strict = _solve_axis_lp_array_tols(
        y1_arr,
        y2_arr,
        r_c,
        r_p1,
        r_p2,
        D_base,
        E1_base,
        E2_base,
        hw_arr,
        c_bound,
        endpoint_only,
    )
    status = "compatible" if strict is not None else "sampling_underpowered"
    return {"status": status, "bounds": adapted, "strict_bounds": strict}


def stack_transition_contrasts(
    observations: ArrayLike,
    common_maps: ArrayLike,
    private_maps: ArrayLike,
    *,
    common_drive: ArrayLike | None = None,
    private_drive: ArrayLike | None = None,
    counts: ArrayLike | None = None,
    obs_noise_var: float = 1.0,
    struct_var: float = 0.05,
    contrast_error=None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return (A, b) with b = A @ common_initial, optionally precision-weighted by counts (GLS)."""
    y = np.asarray(observations, dtype=np.float64)
    a = np.asarray(common_maps, dtype=np.float64)
    p = np.asarray(private_maps, dtype=np.float64)
    if y.ndim != 3 or y.shape[0] < 2 or not all(y.shape):
        raise ValueError(
            "expected observations with at least two frames, one reader and one coordinate"
        )
    frames, readers, d = y.shape
    steps = frames - 1
    if a.shape != (steps, d, d) or p.shape != (steps, readers, d, d):
        raise ValueError(
            "transition maps must match observation steps, readers and coordinates"
        )
    g = (
        np.zeros((steps, d))
        if common_drive is None
        else np.asarray(common_drive, dtype=np.float64)
    )
    f = (
        np.zeros(y[1:].shape)
        if private_drive is None
        else np.asarray(private_drive, dtype=np.float64)
    )
    if g.shape != (steps, d) or f.shape != y[1:].shape:
        raise ValueError("drive shapes must match their respective states")
    if not all(np.isfinite(v).all() for v in (y, a, p, g, f)):
        raise ValueError("observations and calibration must be finite")
    c_arr = None if counts is None else np.asarray(counts, dtype=np.float64)
    if c_arr is not None and (
        c_arr.shape != (frames, readers)
        or np.any(c_arr <= 0)
        or not np.isfinite(c_arr).all()
    ):
        raise ValueError(
            "counts must be positive finite array of shape (frames, readers)"
        )

    phi = np.eye(d)
    offset = np.zeros(d)
    blocks = []
    residuals = []
    for t in range(steps):
        gap = a[t] - p[t]
        blk = gap @ phi
        b = y[t + 1] - (p[t] @ y[t, :, :, None])[..., 0] - g[t] - f[t] - gap @ offset
        if c_arr is not None:
            for r in range(readers):
                p_norm_sq = float(np.sum(p[t, r] ** 2)) / max(1, d)
                var_r = float(struct_var) + float(obs_noise_var) * (
                    1.0 / float(c_arr[t + 1, r]) + p_norm_sq / float(c_arr[t, r])
                )
                scale_r = 1.0 / np.sqrt(max(var_r, 1e-12))
                blk[r] *= scale_r
                b[r] *= scale_r
        blocks.append(blk.reshape(-1, d))
        residuals.append(b.ravel())
        phi = a[t] @ phi
        offset = a[t] @ offset + g[t]
    A, b = np.vstack(blocks), np.concatenate(residuals)
    if contrast_error is not None:
        from .stream_precision import whiten_system

        if counts is not None:
            raise ValueError("specify contrast covariance or counts, not both")
        return whiten_system(A, b, contrast_error)
    return A, b


__all__ = [
    "ConstrainedRiccatiCell",
    "DualPathwayRiccatiObserver",
    "estimate_cross_view_lag1_rates",
    "solve_axis_trajectory_lp",
    "stack_transition_contrasts",
]
