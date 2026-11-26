#!/usr/bin/env python3
"""Tests for GRAIN Open-Stream Psychometric Measurement (P1..P6 & CLI)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from grain_core import psychometric_open_stream as pos
from grain_core import response_profile as rp


def test_p1_reader_misclassification_dif_and_unidentified_refusal() -> None:
    # Exact CaSiNo-style DIF distortion: true rates b1=0.60, b0=0.20 (diff=+0.40)
    # Condition-specific reader calibration: cue has higher FPR & lower sensitivity
    cal = {
        "cue": {"sensitivity": 0.65, "fpr": 0.25},
        "other": {"sensitivity": 0.80, "fpr": 0.10},
    }
    p1 = 0.25 + (0.65 - 0.25) * 0.60  # 0.49
    p0 = 0.10 + (0.80 - 0.10) * 0.20  # 0.24
    res = pos.calibrate_reader_misclassification(
        {"cue": p1, "other": p0},
        cal,
        weights={"cue": 0.5, "other": 0.5},
        contrast=["cue", "other"],
    )
    decomp = res["contrast_calibration"]["dif_decomposition"]
    assert decomp["closure_error"] < 1e-14
    assert np.isclose(decomp["unclipped_corrected_contrast"], 0.40, atol=1e-12)
    assert np.isclose(decomp["uniform_fpr_dif"], 0.15, atol=1e-12)
    assert np.isclose(decomp["nonuniform_discrimination_dif"], -0.30 * 0.40, atol=1e-12)

    # Also test end-to-end compatibility with response_profile.measure output + unidentified refusal
    prof = rp.measure(
        {
            "behavior": "b",
            "reader": "r",
            "condition_rule": "cr",
            "unit": "u",
            "frame": "observed_replies",
            "weights": {"cue": 0.5, "other": 0.5},
            "contrast": ["cue", "other"],
        },
        [
            {"focal": "a", "dialogue": "d1", "id": "1", "condition": "cue", "status": "support", "response_id": "r1"},
            {"focal": "a", "dialogue": "d1", "id": "2", "condition": "cue", "status": "refute", "response_id": "r2"},
            {"focal": "a", "dialogue": "d1", "id": "3", "condition": "other", "status": "refute", "response_id": "r3"},
        ],
    )["cards"][0]
    unid = pos.calibrate_reader_misclassification(
        prof,
        {
            "cue": {"sensitivity": [0.35, 0.55], "fpr": [0.40, 0.50]},  # s_lo <= f_hi
            "other": {"tp": 40, "fp": 10, "tn": 90, "fn": 10},
        },
        contrast=["cue", "other"],
    )
    assert unid["conditions"]["cue"]["status"] == "reader_unidentified"
    assert unid["conditions"]["cue"]["corrected_bounds"] == [0.0, 1.0]
    assert unid["contrast_calibration"]["dif_decomposition"]["status"] == "reader_unidentified"


def test_p2_bivariate_caps_kelley_eb_and_rouder_deattenuation() -> None:
    rng = np.random.default_rng(7)
    n_u = 80
    theta_1 = rng.normal(0.0, 0.40, n_u)
    theta_0 = -0.8 * theta_1 + rng.normal(0.0, 0.25, n_u)
    n_a1 = rng.choice([2, 3, 40, 60], size=n_u, p=[0.25, 0.25, 0.25, 0.25])
    n_a0 = rng.choice([2, 3, 40, 60], size=n_u, p=[0.25, 0.25, 0.25, 0.25])
    n_b1 = rng.integers(30, 50, size=n_u)
    n_b0 = rng.integers(30, 50, size=n_u)
    ev1, ev0 = 0.8, 0.8

    recs_a = []
    recs_b = []
    for i in range(n_u):
        recs_a.append(
            {
                "focal": f"u{i}",
                "mean_1": float(theta_1[i] + rng.normal(0.0, np.sqrt(ev1 / n_a1[i]))),
                "n_1": int(n_a1[i]),
                "mean_0": float(theta_0[i] + rng.normal(0.0, np.sqrt(ev0 / n_a0[i]))),
                "n_0": int(n_a0[i]),
            }
        )
        recs_b.append(
            {
                "focal": f"u{i}",
                "mean_1": float(theta_1[i] + rng.normal(0.0, np.sqrt(ev1 / n_b1[i]))),
                "n_1": int(n_b1[i]),
                "mean_0": float(theta_0[i] + rng.normal(0.0, np.sqrt(ev0 / n_b0[i]))),
                "n_0": int(n_b0[i]),
            }
        )

    fit_a = pos.estimate_bivariate_caps_reliability(
        recs_a[:40],
        event_var_c1=ev1,
        event_var_c0=ev0,
        min_reliability=0.50,
        fit_records=recs_a[40:],
    )
    fit_b = pos.estimate_bivariate_caps_reliability(
        recs_b[:40],
        event_var_c1=ev1,
        event_var_c0=ev0,
        min_reliability=0.50,
        fit_records=recs_b[40:],
    )
    # Sparse units (n <= 3) should have lower reliability and EB beats raw against true contrast
    true_con = theta_1[:40] - theta_0[:40]
    raw_mse = float(np.mean([(p["raw_contrast"] - t) ** 2 for p, t in zip(fit_a["persons"], true_con, strict=True)]))
    eb_mse = float(np.mean([(p["eb_contrast"] - t) ** 2 for p, t in zip(fit_a["persons"], true_con, strict=True)]))
    assert eb_mse < raw_mse
    sparse_rels = [p["reliability_contrast"] for p in fit_a["persons"] if p["n_1"] <= 3]
    dense_rels = [p["reliability_contrast"] for p in fit_a["persons"] if p["n_1"] >= 40]
    assert np.mean(sparse_rels) < np.mean(dense_rels)

    cross = pos.deattenuate_cross_panel_correlation(fit_a, fit_b, metric="contrast", n_perm=200, seed=11)
    assert cross["pearson_r_eb"] > cross["pearson_r_raw"] > 0.4
    assert cross["r_deattenuated_unclipped"] is not None
    assert cross["stranger_null_perm_p_eb"] < 0.02

    # Null variance world must flag no_latent_contrast_variance
    flat = [{"focal": f"z{i}", "mean_1": 0.5, "n_1": 10, "mean_0": 0.5, "n_0": 10} for i in range(6)]
    zero_out = pos.estimate_bivariate_caps_reliability(flat, event_var_c1=0.1, event_var_c0=0.1)
    assert zero_out["latent_contrast_variance"] == 0.0
    assert all(p["reporting_status"] == "no_latent_contrast_variance" for p in zero_out["persons"])


def test_p3_ct_lsto_regime_separation_and_carry_threshold() -> None:
    rng = np.random.default_rng(19)
    ts_list, au_list, co_list, val_list = [], [], [], []
    for i in range(25):
        trait = rng.normal(0.0, 0.45)
        zeta = rng.normal(0.0, 0.70)
        t = 0.0
        for step in range(8):
            dt = 0.1 if step % 2 == 1 else 2.5
            t += dt
            zeta = np.exp(-0.5 * dt) * zeta + rng.normal(0.0, np.sqrt(max(1e-6, 0.49 * (1.0 - np.exp(-dt)))))
            ts_list.append(t)
            au_list.append(f"a{i}")
            co_list.append("c1")
            val_list.append(float(trait + zeta + rng.normal(0.0, 0.25)))

    res_state = pos.fit_ct_lsto_variance_components(ts_list, au_list, co_list, val_list, n_starts=6)
    assert res_state["sigma_state_sq"] > res_state["sigma_occ_sq"]
    assert res_state["naive_carry_hurts_when_occ_exceeds_state"] is False
    assert res_state["regime"] == "bursty_state_trait_active"

    # Pure trait + iid occasion noise world must detect occ > state & naive_carry_hurts
    val_iid = [
        float((i % 5 - 2) * 0.25 + rng.normal(0.0, 1.0))
        for i in range(25)
        for _ in range(8)
    ]
    res_iid = pos.fit_ct_lsto_variance_components(ts_list, au_list, co_list, val_iid, n_starts=6)
    assert res_iid["sigma_occ_sq"] > res_iid["sigma_state_sq"]
    assert res_iid["naive_carry_hurts_when_occ_exceeds_state"] is True
    assert res_iid["regime"] == "trait_plus_iid_occasion"


def test_p4_irtree_selection_vs_expression_and_manski_bounds() -> None:
    rng = np.random.default_rng(31)
    ts, aus, cos, vals = [], [], [], []
    edges = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    for i in range(20):
        # Selection profile uncorrelated with expression profile
        p_sel = np.array([0.85, 0.15]) if i % 2 == 0 else np.array([0.15, 0.85])
        expr_mean = 0.7 if i < 10 else -0.7
        for w in range(6):
            # Force at least one venue active so author is online every window
            active_c = "c1" if rng.random() < p_sel[0] else "c2"
            ts.append(w + 0.3)
            aus.append(f"u{i}")
            cos.append(active_c)
            vals.append(float(expr_mean + rng.normal(0.0, 0.15)))
    res = pos.measure_irtree_selection_expression(
        ts, aus, cos, vals, window_edges=edges, min_active_windows=4, eta_min=0.5, n_perm=200, seed=4
    )
    assert res["node1_selection_contiguous_repro"]["split_half_profile_r"] > 0.35
    assert res["node1_selection_contiguous_repro"]["stranger_null_perm_p"] < 0.05
    assert res["node2_expression_contiguous_repro"]["split_half_r"] > 0.80
    assert abs(res["cross_node_selection_vs_expression_r"]) < 0.35
    for au in res["authors"]:
        for c_row in au["conditions"].values():
            lo, hi = c_row["manski_participation_bounds"]
            assert 0.0 <= lo <= hi <= 1.0


def test_p5_hierarchical_five_term_mix_reverses_stayer_sign() -> None:
    # Exact within-person Simpson's paradox witness (plus cohort turnover)
    prev_cells = {
        ("stayer1", "c1"): (1, [0.2, 0.0]),
        ("stayer1", "c2"): (9, [0.8, 0.0]),
        ("leaver", "c1"): (5, [0.9, 0.2]),
    }
    cur_cells = {
        ("stayer1", "c1"): (9, [0.3, 0.1]),
        ("stayer1", "c2"): (1, [0.9, 0.1]),
        ("entrant", "c2"): (5, [-0.5, -0.3]),
    }
    dec = pos.decompose_hierarchical_cohort_person_mix(prev_cells, cur_cells)
    assert dec["status"] == "decomposed"
    assert dec["closure_max_abs_err"] < 1e-14
    # stayer1 improved by +0.10 in BOTH c1 and c2, yet raw mean fell 0.74 -> 0.36 (-0.38)
    assert dec["within_person_within_condition"][0] > 0.09
    assert dec["within_person_condition_mix"][0] < -0.37
    assert dec["n_stayers_sign_reversed_by_mix"] == 1


def test_p6_register_dif_invariance_and_g12_license_upgrade() -> None:
    rng = np.random.default_rng(42)
    n_authors = 30
    au_ids, grp_list, emb_list, sp_list, y_list = [], [], [], [], []
    for a in range(n_authors):
        trait = rng.normal(0.0, 1.0)
        for rep in range(8):
            is_r1 = rep >= 4
            z = np.array([trait + rng.normal(0.0, 0.15), rng.normal(0.0, 0.5)])
            if is_r1:
                # Register 1 reverses sign AND shifts offset unless interaction is modeled
                emb = np.array([-z[0], z[1], rng.normal(0.0, 0.2)])
                sp = np.array([0.05, 0.95])
                y_val = trait + 0.05 * z[1] + 0.05 * rng.normal()
            else:
                emb = np.array([z[0], z[1], rng.normal(0.0, 0.2)])
                sp = np.array([0.95, 0.05])
                y_val = trait + 0.05 * z[1] + 0.60 + 0.05 * rng.normal()
            au_ids.append(f"a{a}")
            grp_list.append("r1" if is_r1 else "r0")
            emb_list.append(emb)
            sp_list.append(sp)
            y_list.append(float(y_val))

    audit = pos.audit_register_dif_invariance(
        np.array(emb_list),
        np.array(sp_list),
        grp_list,
        np.array(y_list),
        au_ids,
        n_folds=5,
        seed=9,
    )
    assert audit["blind"]["earned_comparison_license"] == "L0_unlicensed"
    assert audit["kronecker"]["earned_comparison_license"] in (
        "L2_metric_contrast_comparable",
        "L3_scalar_level_comparable",
    )
    assert audit["kronecker"]["overall_mse"] < 0.25 * audit["blind"]["overall_mse"]
    assert audit["nonuniform_slope_reduction_ratio"] > 3.0
    assert audit["kronecker"]["nonuniform_slope_gap"] <= 0.15


def test_cli_demo_smoke() -> None:
    demo = Path(__file__).resolve().parents[1] / "examples" / "psychometric_open_stream_demo.json"
    proc = subprocess.run(
        [sys.executable, "-m", "grain_core.psychometric_open_stream", str(demo)],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["schema"] == pos.VERSION
    assert set(payload["result"]) == {
        "reader_dif",
        "caps_eb",
        "ct_lsto",
        "irtree",
        "hierarchical_mix",
    }
    assert payload["result"]["hierarchical_mix"]["closure_max_abs_err"] < 1e-14
