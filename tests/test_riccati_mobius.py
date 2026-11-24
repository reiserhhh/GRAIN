#!/usr/bin/env python3
"""Unit tests for grain_core.riccati_mobius standalone GRAIN extensions."""

from __future__ import annotations

import numpy as np
from grain_core.riccati_mobius import (
    ConstrainedRiccatiCell,
    DualPathwayRiccatiObserver,
    estimate_cross_view_lag1_rates,
    solve_axis_trajectory_lp,
    stack_transition_contrasts,
)


def test_riccati_zero_time_subdivision_bounds_and_scan() -> None:
    rng = np.random.default_rng(7)
    bound_B = 1.25
    cell = ConstrainedRiccatiCell(input_dim=5, hidden_dim=8, bound_B=bound_B, s_min=0.05)
    u = rng.normal(size=(5,))
    x0 = rng.uniform(-0.9, 0.9, size=(8,))

    M0 = cell.step_matrices(u, 0.0)
    assert np.max(np.abs(cell.apply_mobius(M0, x0) - x0)) < 1e-15

    dt_total = 1.6
    x_one = cell.apply_mobius(cell.step_matrices(u, dt_total), x0)
    M_quarter = cell.step_matrices(u, dt_total / 4.0)
    x_sub = x0.copy()
    for _ in range(4):
        x_sub = cell.apply_mobius(M_quarter, x_sub)
    assert np.all(np.abs(x_one) <= bound_B + 1e-12)
    assert np.max(np.abs(x_one - x_sub)) < 1e-12

    u_seq = rng.normal(size=(12, 5))
    dt_seq = rng.uniform(0.1, 1.5, size=(12,))
    assert np.max(np.abs(cell.rollout_sequential(u_seq, dt_seq, x0) - cell.rollout_parallel_scan(u_seq, dt_seq, x0))) < 1e-12


def test_dual_pathway_episodic_preservation() -> None:
    obs = DualPathwayRiccatiObserver(dim=6, episodic_weight=0.35, gamma_scale=0.35)
    events = np.full((16, 6), 1.0 / 6.0)
    events[0] = np.array([0.70, 0.06, 0.06, 0.06, 0.06, 0.06])
    events[-1] = np.array([0.70, 0.06, 0.06, 0.06, 0.06, 0.06])
    gaps = np.full((16,), 2.0)
    gaps[0] = 0.0

    fast_diff = 0.5 * float(
        np.sum(
            np.abs(
                obs.rollout_with_idle_gaps(events, gaps, mask_first=False, enable_episodic=False)[-1]
                - obs.rollout_with_idle_gaps(events, gaps, mask_first=True, enable_episodic=False)[-1]
            )
        )
    )
    dual_diff = 0.5 * float(
        np.sum(
            np.abs(
                obs.rollout_with_idle_gaps(events, gaps, mask_first=False, enable_episodic=True)[-1]
                - obs.rollout_with_idle_gaps(events, gaps, mask_first=True, enable_episodic=True)[-1]
            )
        )
    )
    assert fast_diff < 1e-4
    assert dual_diff > 0.01


def test_dynamic_lp_shrinkage_vs_equal_rate_degeneracy_and_mom() -> None:
    rng = np.random.default_rng(123)
    N, T = 40, 6
    rc_true, rp1_true, rp2_true = 0.85, 0.25, 0.40
    c = np.zeros((N, T))
    p1 = np.zeros((N, T))
    p2 = np.zeros((N, T))
    c[:, 0] = rng.normal(0.0, 1.0, N)
    p1[:, 0] = rng.normal(0.0, 1.2, N)
    p2[:, 0] = rng.normal(0.0, 1.2, N)
    for t in range(T - 1):
        c[:, t + 1] = rc_true * c[:, t] + rng.uniform(-0.02, 0.02, N)
        p1[:, t + 1] = rp1_true * p1[:, t] + rng.uniform(-0.02, 0.02, N)
        p2[:, t + 1] = rp2_true * p2[:, t] + rng.uniform(-0.02, 0.02, N)
    y1 = (c + p1)[:, :, None]
    y2 = (c + p2)[:, :, None]
    est_c, est_p1, est_p2 = estimate_cross_view_lag1_rates(y1, y2, rank=1)
    assert abs(float(est_c[0]) - rc_true) < 0.12
    assert abs(float(est_p1[0]) - float(est_p2[0])) > 0.02

    res_sep = solve_axis_trajectory_lp(y1[0, :, 0], y2[0, :, 0], rc_true, rp1_true, rp2_true, 0.021, 0.021, 0.021, None, c_bound=10.0)
    c_eq = np.zeros(T)
    p1_eq = np.zeros(T)
    p2_eq = np.zeros(T)
    c_eq[0], p1_eq[0], p2_eq[0] = c[0, 0], p1[0, 0], p2[0, 0]
    for t in range(T - 1):
        c_eq[t + 1] = 0.5 * c_eq[t] + rng.uniform(-0.02, 0.02)
        p1_eq[t + 1] = 0.5 * p1_eq[t] + rng.uniform(-0.02, 0.02)
        p2_eq[t + 1] = 0.5 * p2_eq[t] + rng.uniform(-0.02, 0.02)
    res_eq = solve_axis_trajectory_lp(c_eq + p1_eq, c_eq + p2_eq, 0.5, 0.5, 0.5, 0.021, 0.021, 0.021, None, c_bound=10.0)
    assert res_sep is not None and res_eq is not None
    w_sep = float(np.median(res_sep[1][1:-1] - res_sep[0][1:-1]))
    w_eq = float(np.median(res_eq[1][1:-1] - res_eq[0][1:-1]))
    assert w_sep < 0.25
    assert w_eq > 5.0 * w_sep


def test_stack_transition_contrasts_lifts_rank() -> None:
    rng = np.random.default_rng(31)
    c = np.zeros((3, 2))
    c[0] = np.array([1.2, -0.7])
    A = np.array([[[0.8, 0.1], [-0.05, 0.75]], [[0.7, 0.0], [0.1, 0.85]]])
    P = np.array(
        [
            [[[0.8, 0.1], [-0.05, 0.2]], [[0.8, 0.1], [-0.05, 0.3]]],
            [[[0.2, 0.0], [0.1, 0.85]], [[0.3, 0.0], [0.1, 0.85]]],
        ]
    )
    p = rng.normal(0.0, 0.8, size=(3, 2, 2))
    g = np.array([[0.1, -0.05], [0.0, 0.15]])
    for t in range(2):
        c[t + 1] = A[t] @ c[t] + g[t]
        for r in range(2):
            p[t + 1, r] = P[t, r] @ p[t, r]
    y = c[:, None, :] + p
    H, b = stack_transition_contrasts(y, A, P, common_drive=g)
    assert np.linalg.matrix_rank(H) == 2
    c0_hat = np.linalg.lstsq(H, b, rcond=None)[0]
    assert np.max(np.abs(c0_hat - c[0])) < 1e-12
