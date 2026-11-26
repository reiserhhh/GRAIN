#!/usr/bin/env python3
"""Unit tests and G6 counterexample checks for G-1..G-6 open-stream operators."""

from __future__ import annotations

import numpy as np
import pytest
from grain_core.open_cohort import PrefixHygiene, decompose_open_cohort_step
from grain_core.pragmatic_register import (
    CrossFittedRegisterCalibrator,
    extract_pragmatic_acts,
    mtmm_validity_audit,
    register_channel_probs,
)
from grain_core.riccati_mobius import (
    ConstrainedRiccatiCell,
    estimate_cross_view_lag1_rates,
    solve_axis_trajectory_lp,
    stack_transition_contrasts,
)
from grain_core.selective_silence import compute_selective_silence_panel
from grain_core.transport_gate import (
    ConditionScopedRiccatiStream,
    evaluate_transportability_gate,
)

from grain_core import liquid_axis_series as ax
from grain_core import liquid_common_range as cr
from grain_core import liquid_reference_error as lre


def test_g1_heteroscedastic_eb_deattenuation_and_g14_diagnostics() -> None:
    rng = np.random.default_rng(101)
    # 1. Empirical-Bayes chart weighting reduces center error on n=1 vs n=100 units
    n_u, dim = 60, 3
    counts = np.where(np.arange(n_u) < 30, 1.0, 100.0)
    mu_true = np.array([1.2, -0.8, 0.5])
    sig_x = rng.normal(0.0, 0.25, size=(n_u, dim))
    eps_x = (1.5 / np.sqrt(counts))[:, None] * rng.normal(size=(n_u, dim))
    x = mu_true + sig_x + eps_x
    ch_unw = lre.fit_chart(x, ["a", "b", "c"], population="test", basis="raw")
    ch_eb = lre.fit_chart(
        x, ["a", "b", "c"], population="test", basis="raw", counts=counts
    )
    assert ch_eb["eb_kappa"] > 5.0
    assert np.linalg.norm(np.array(ch_eb["center"]) - mu_true) < np.linalg.norm(
        np.array(ch_unw["center"]) - mu_true
    )

    # 2. Lag-1 de-attenuation prevents false rate attenuation on sparse cells
    N, T = 80, 6
    c_cnt = rng.choice([1.0, 80.0], size=(N, T), p=[0.65, 0.35])
    rc_t, rp1_t, rp2_t = 0.80, 0.82, 0.82
    c_lat = np.zeros((N, T))
    p1_lat = np.zeros((N, T))
    p2_lat = np.zeros((N, T))
    c_lat[:, 0] = rng.normal(0.0, 1.0, N)
    p1_lat[:, 0] = rng.normal(0.0, 1.0, N)
    p2_lat[:, 0] = rng.normal(0.0, 1.0, N)
    for t in range(T - 1):
        c_lat[:, t + 1] = rc_t * c_lat[:, t] + rng.normal(0.0, 0.1, N)
        p1_lat[:, t + 1] = rp1_t * p1_lat[:, t] + rng.normal(0.0, 0.1, N)
        p2_lat[:, t + 1] = rp2_t * p2_lat[:, t] + rng.normal(0.0, 0.1, N)
    sigma_e = 1.4
    y1 = (c_lat + p1_lat + (sigma_e / np.sqrt(c_cnt)) * rng.normal(size=(N, T)))[
        :, :, None
    ]
    y2 = (c_lat + p2_lat + (sigma_e / np.sqrt(c_cnt)) * rng.normal(size=(N, T)))[
        :, :, None
    ]
    _, rp1_raw, _ = estimate_cross_view_lag1_rates(y1, y2, rank=1)
    _, rp1_deb, _ = estimate_cross_view_lag1_rates(
        y1, y2, rank=1, counts1=c_cnt, counts2=c_cnt
    )
    assert abs(float(rp1_deb[0]) - rp1_t) < abs(float(rp1_raw[0]) - rp1_t)

    # 3. Conformal LP & common-range separate sampling_underpowered from structural_infeasible
    y1_t = np.array([0.0, 0.0, 0.0, 0.0])
    y2_t = np.array([0.0, 0.55, 0.55, 0.0])
    diag_under = solve_axis_trajectory_lp(
        y1_t,
        y2_t,
        0.8,
        0.5,
        0.5,
        0.05,
        0.1,
        0.1,
        None,
        counts1=np.array([50.0, 1.0, 50.0, 50.0]),
        counts2=np.array([50.0, 1.0, 50.0, 50.0]),
        sigma_eps1=0.6,
        sigma_eps2=0.6,
        return_diagnostics=True,
    )
    assert (
        isinstance(diag_under, dict) and diag_under["status"] == "sampling_underpowered"
    )
    diag_bad = solve_axis_trajectory_lp(
        y1_t,
        np.array([0.0, 9.0, 9.0, 0.0]),
        0.8,
        0.5,
        0.5,
        0.05,
        0.1,
        0.1,
        None,
        counts1=np.full(4, 100.0),
        counts2=np.full(4, 100.0),
        sigma_eps1=0.1,
        sigma_eps2=0.1,
        return_diagnostics=True,
    )
    assert isinstance(diag_bad, dict) and diag_bad["status"] == "structural_infeasible"

    # Common range 3-way partition
    X_syn = rng.normal(size=(36, 5, 4))
    X_syn[:, :, 2] = X_syn[:, :, 0] + 0.05 * rng.normal(size=(36, 5))
    ref_cr = cr.fit(
        X_syn, list("abcd"), 2, "syn", "event", counts=np.full((36, 5, 2), 25.0)
    )
    assert ref_cr["heteroscedastic"] is True
    view_a = X_syn[0, 0, :2]
    view_b0 = X_syn[0, 0, 2:]
    base_e = float(cr.estimate(ref_cr, [view_a, view_b0], None)["minimum_private_norm"])
    view_b = view_b0 + np.array([1.1, 0.0])
    est_compat = cr.estimate(
        ref_cr,
        [view_a, view_b0],
        budget=base_e + 0.05,
        counts=[50.0, 50.0],
        event_sigma=[0.5, 0.5],
    )
    assert est_compat["status"] == "compatible"
    est_under = cr.estimate(
        ref_cr,
        [view_a, view_b],
        budget=base_e + 0.05,
        counts=[50.0, 1.0],
        event_sigma=[1.0, 1.0],
    )
    assert est_under["status"] == "sampling_underpowered"
    est_out = cr.estimate(
        ref_cr,
        [view_a, view_b0 + np.array([6.0, 0.0])],
        budget=base_e + 0.05,
        counts=[50.0, 50.0],
        event_sigma=[0.2, 0.2],
    )
    assert est_out["status"] == "outside_private_budget"

    # 4. Precision-weighted GLS stack_transition_contrasts reduces error vs OLS
    A = np.array([[[0.8, 0.0], [0.0, 0.75]], [[0.7, 0.0], [0.0, 0.8]]])
    P = np.array(
        [
            [[[0.2, 0.0], [0.0, 0.25]], [[0.3, 0.0], [0.0, 0.2]]],
            [[[0.25, 0.0], [0.0, 0.2]], [[0.2, 0.0], [0.0, 0.3]]],
        ]
    )
    c0_true = np.array([1.5, -1.0])
    c_path = np.zeros((3, 2))
    c_path[0] = c0_true
    for t in range(2):
        c_path[t + 1] = A[t] @ c_path[t]
    p_path = np.zeros((3, 2, 2))
    cnts = np.array([[100.0, 1.0], [100.0, 1.0], [100.0, 1.0]])
    y_clean = c_path[:, None, :] + p_path
    y_noisy = y_clean.copy()
    y_noisy[:, 1, :] += 1.5 * rng.normal(size=(3, 2))
    y_noisy[:, 0, :] += 0.1 * rng.normal(size=(3, 2))
    H_ols, b_ols = stack_transition_contrasts(y_noisy, A, P)
    H_gls, b_gls = stack_transition_contrasts(
        y_noisy, A, P, counts=cnts, obs_noise_var=2.25, struct_var=1e-4
    )
    c0_ols = np.linalg.lstsq(H_ols, b_ols, rcond=None)[0]
    c0_gls = np.linalg.lstsq(H_gls, b_gls, rcond=None)[0]
    assert np.linalg.norm(c0_gls - c0_true) < np.linalg.norm(c0_ols - c0_true)


def test_g2_impulse_after_idle_and_bipolar_symmetry() -> None:
    cell = ConstrainedRiccatiCell(
        input_dim=4, hidden_dim=6, bound_B=1.0, s_min=0.1, seed=29
    )
    rng = np.random.default_rng(29)
    u_seq = rng.normal(size=(3, 14, 4))
    dt_gaps = rng.uniform(0.5, 80.0, size=(3, 14))
    x0 = rng.uniform(-0.5, 0.5, size=(3, 6))

    seq = cell.rollout_impulse_sequential(u_seq, dt_gaps, pulse_tau=1.0, x0=x0)
    par = cell.rollout_impulse_parallel_scan(u_seq, dt_gaps, pulse_tau=1.0, x0=x0)
    assert np.max(np.abs(seq - par)) < 1e-12
    assert np.all(np.abs(seq) <= 1.0 + 1e-12)

    # Empty window counterfactual delta is strictly zero; arrival yields non-zero bounded delta
    ts = np.array([0.0, 2.0, 5.0])
    u_one = rng.normal(size=(3, 4))
    cf_empty = cell.counterfactual_window_delta(
        u_one, ts, t_cut=10.0, window_W=4.0, pulse_tau=1.0
    )
    assert np.max(np.abs(cf_empty["delta"])) == 0.0
    cf_hit = cell.counterfactual_window_delta(
        u_one, ts, t_cut=5.5, window_W=1.5, pulse_tau=1.0
    )
    assert np.max(np.abs(cf_hit["delta"])) > 1e-3

    # Bipolar origin symmetry: exact zero at mean and odd flip under z -> -z
    ref = {"mean": np.array([1.0, -2.0]), "scale": np.array([2.0, 0.5])}
    vals = np.array([[1.0, -2.0], [3.0, -1.0], [-1.0, -3.0]])
    p_bp = ax.encode_bipolar_origin(vals, ref)
    assert p_bp.shape == (3, 5)
    np.testing.assert_allclose(ax.decode_bipolar_origin(p_bp, ref), vals, atol=1e-12)
    pol = ax.bipolar_polarization(p_bp)
    np.testing.assert_allclose(pol[0], np.zeros(2), atol=1e-15)
    np.testing.assert_allclose(pol[1], -pol[2], atol=1e-13)


def test_g3_selective_silence_mar_vs_mnar_and_null() -> None:
    # Build two conditions A, B over windows 0..3; in window 3, positive-prior authors
    # go online-silent in A (active in B -> MNAR), while in B absenteeism is full-offline (MAR).
    ts, au, co, vecs = [], [], [], []
    for w in range(3):
        t0 = float(w * 10)
        for i in range(10):
            ts.extend([t0 + 1.0, t0 + 2.0])
            au.extend([f"A_{i}", f"A_{i}"])
            co.extend(["A", "A"])
            val = 1.5 if i < 5 else -1.5
            vecs.extend([[val], [val]])

            ts.extend([t0 + 3.0, t0 + 4.0])
            au.extend([f"B_{i}", f"B_{i}"])
            co.extend(["B", "B"])
            vecs.extend([[val], [val]])

    # Window 3: Positive A incumbents (0..4) avoid A but speak in B; negative A incumbents stay in A.
    # For B, half incumbents go completely offline (MAR).
    t3 = 30.0
    for i in range(10):
        if i < 5:
            ts.append(t3 + 1.0)
            au.append(f"A_{i}")
            co.append("B")
            vecs.append([0.5])
        else:
            ts.append(t3 + 2.0)
            au.append(f"A_{i}")
            co.append("A")
            vecs.append([-1.5])
        if i >= 5:
            ts.append(t3 + 3.0)
            au.append(f"B_{i}")
            co.append("B")
            vecs.append([-1.5])

    res = compute_selective_silence_panel(
        ts,
        au,
        co,
        np.asarray(vecs),
        window_edges=[0.0, 10.0, 20.0, 30.0, 40.0],
        min_incumbent_events=2,
    )
    w3_A = next(r for r in res if r["window_index"] == 3 and r["condition_id"] == "A")
    w3_B = next(r for r in res if r["window_index"] == 3 and r["condition_id"] == "B")
    assert np.isclose(
        w3_A["active_retention_rate"]
        + w3_A["online_avoid_rate"]
        + w3_A["offline_absent_rate"],
        1.0,
    )
    assert w3_A["online_avoid_rate"] > 0.45 and w3_A["offline_absent_rate"] < 0.05
    assert float(w3_A["asym_signed_abstention"][0]) > 0.5
    assert w3_B["offline_absent_rate"] > 0.45 and w3_B["online_avoid_rate"] < 0.05


def test_g4_kitagawa_oaxaca_and_causal_prefix_hygiene() -> None:
    prev_m = {f"s{i}": np.array([0.0, 1.0]) for i in range(10)}
    cur_m = {
        **{f"s{i}": np.array([0.0, 1.0]) for i in range(10)},
        **{f"e{i}": np.array([2.0, -1.0]) for i in range(10)},
    }
    dec = decompose_open_cohort_step(prev_m, cur_m)
    assert dec["identity_max_abs_err"] < 1e-14
    assert np.allclose(dec["stayer_evolution"], np.zeros(2))
    assert np.isclose(dec["turnover_to_total_norm_ratio"], 1.0)

    # Bitwise causal PrefixHygiene under future perturbations
    ts1 = [1.0, 5.0, 10.0, 12.0, 13.0]
    au1 = ["u1", "u2", "u1", "u1", "u1"]
    tx1 = ["hello", "world", "hi there", "spam", "spam2"]
    res1 = PrefixHygiene(burst_window=10.0, max_burst=2).filter_stream(ts1, au1, tx1)

    ts2 = [1.0, 5.0, 10.0, 12.0, 12.1, 12.2, 12.3]
    au2 = ["u1", "u2", "u1", "u1", "u1", "u1", "u1"]
    tx2 = ["hello", "world", "hi there", "spam", "spam", "spam", "spam"]
    res2 = PrefixHygiene(burst_window=10.0, max_burst=2).filter_stream(ts2, au2, tx2)
    assert res1[:4] == res2[:4]
    assert res2[-1]["is_clean"] is False

    with pytest.raises(ValueError, match="monotonically"):
        PrefixHygiene().filter_stream([2.0, 1.0], ["a", "a"], ["x", "y"])


def test_g5_pragmatic_acts_kronecker_probe_and_mtmm() -> None:
    acts = extract_pragmatic_acts(
        [
            "Personally I definitely expect this will succeed next year!",
            "According to someone else, maybe it could work if conditions hold.",
        ]
    )
    assert acts.shape == (2, 5)
    assert (
        acts[0, 0] > acts[1, 0] and acts[0, 1] > acts[1, 1] and acts[0, 2] > acts[1, 2]
    )

    # Same author's latent trait observed across short vs long replies with opposite channel offsets
    rng = np.random.default_rng(77)
    n_au = 40
    z_a = rng.normal(size=n_au)
    z_b = rng.normal(size=n_au)
    au_ids, embs, texts, is_rep, y_a, y_b = [], [], [], [], [], []
    for i in range(n_au):
        # Reg 1: short reply (sign inversion bias +1.5 on raw dim 1)
        au_ids.append(f"u{i}")
        texts.append("Short reply here.")
        is_rep.append(True)
        e1 = np.array([z_a[i], 0.3 * z_a[i] + 1.5, z_b[i], -1.2]) + 0.1 * rng.normal(
            size=4
        )
        embs.append(e1)
        y_a.append(z_a[i])
        y_b.append(z_b[i])

        # Reg 2: long standalone post (opposite baseline -1.5, rotated relation)
        au_ids.append(f"u{i}")
        texts.append("Long analytical essay " * 20)
        is_rep.append(False)
        e2 = np.array([-z_a[i], -0.3 * z_a[i] - 1.5, -z_b[i], 1.2]) + 0.1 * rng.normal(
            size=4
        )
        embs.append(e2)
        y_a.append(z_a[i])
        y_b.append(z_b[i])

    E = np.vstack(embs)
    rp = register_channel_probs(texts, is_reply=is_rep)
    cal_a = CrossFittedRegisterCalibrator(alphas=(0.1, 1.0, 10.0), n_folds=5, seed=11)
    res_a = cal_a.fit_predict(E, rp, np.asarray(y_a), au_ids)
    cal_b = CrossFittedRegisterCalibrator(alphas=(0.1, 1.0, 10.0), n_folds=5, seed=11)
    res_b = cal_b.fit_predict(E, rp, np.asarray(y_b), au_ids)

    pred_a = res_a["oof_predictions"]
    pred_b = res_b["oof_predictions"]
    audit = mtmm_validity_audit(pred_a[0::2], pred_a[1::2], pred_b[0::2], pred_b[1::2])
    assert audit["convergent_mean"] > 0.85
    assert audit["mtmm_margin"] > 0.5

    # G6 null: permuting targets breaks OOF prediction
    perm_res = cal_a.fit_predict(
        E, rp, np.asarray(y_a)[rng.permutation(len(y_a))], au_ids
    )
    assert perm_res["oof_mse"] > 3.0 * res_a["oof_mse"]


def test_g6_condition_scoped_stream_and_transport_gate() -> None:
    cell = ConstrainedRiccatiCell(
        input_dim=3, hidden_dim=4, bound_B=1.0, s_min=0.1, seed=19
    )
    stream = ConditionScopedRiccatiStream(cell, pulse_tau=1.0)
    r_a1 = stream.step("au1", "condA", 1.0, np.array([1.0, 0.0, 0.0]))
    stream.step("au1", "condB", 2.0, np.array([-3.0, -3.0, -3.0]))
    r_a2 = stream.step("au1", "condA", 3.0, np.array([1.0, 0.0, 0.0]))
    assert r_a1["is_fresh_start"] is True and r_a2["is_fresh_start"] is False
    # condB shock must not perturb condA trajectory vs an isolated stream
    iso = ConditionScopedRiccatiStream(cell, pulse_tau=1.0)
    iso.step("au1", "condA", 1.0, np.array([1.0, 0.0, 0.0]))
    r_iso = iso.step("au1", "condA", 3.0, np.array([1.0, 0.0, 0.0]))
    np.testing.assert_allclose(
        r_a2["pre_arrival_state"], r_iso["pre_arrival_state"], atol=1e-14
    )

    # Gate blocks HM281 negative transfer & passes true benefit
    aus = [f"u{i}" for i in range(40)]
    cids = ["regimeA"] * 20 + ["regimeB"] * 20
    rng = np.random.default_rng(5)
    lf = np.full(40, 1.0)
    lc_bad = lf + 0.12 + 0.05 * rng.normal(size=40)
    g_closed = evaluate_transportability_gate(aus, lc_bad, lf, condition_ids=cids)
    assert g_closed["gate_open"] is False and g_closed["transport_weight"] == 0.0

    # Pooled positive but harms regimeB > max_subcondition_harm -> gate stays closed
    lc_split = np.concatenate([np.full(20, 0.7), np.full(20, 1.15)])
    g_harm = evaluate_transportability_gate(
        aus, lc_split, lf, condition_ids=cids, max_subcondition_harm=0.01
    )
    assert g_harm["gate_open"] is False

    lc_good = lf - 0.15 + 0.03 * rng.normal(size=40)
    g_open = evaluate_transportability_gate(aus, lc_good, lf, condition_ids=cids)
    assert g_open["gate_open"] is True and g_open["transport_weight"] == 1.0
