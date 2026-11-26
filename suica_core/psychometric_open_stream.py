#!/usr/bin/env python3
"""Unified open-stream personality psychometric measurement (G-1..G-6 & D88/D89 synthesis).

Deterministic scoring (`R-SCORE`, zero LLM calls) composing:
- P1: Rogan-Gladen reader misclassification inversion & 3-way Reader-DIF contrast decomposition
- P2: Bivariate 2D Kelley Empirical-Bayes CAPS reliability, MDD95 & Rouder de-attenuation
- P3: Continuous-Time Latent State-Trait-Occasion (CT-LSTO) semivariogram & carry threshold
- P4: Two-node IRTree (Venue Selection vs Conditional Expression) + Manski exposure bounds
- P5: Exact 5-term Hierarchical Cohort x Within-Person Condition-Mix Kitagawa-Oaxaca identity
- P6: Meredith-style Register DIF hierarchy & G12 comparison license gate

Supports layer-M psychometric auditing without fabricating construct-level (Theta) claims.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from .open_cohort import decompose as _decompose_cohort_1d
from .pragmatic_register import CrossFittedRegisterCalibrator, mtmm_validity_audit
from .response_profile import VERSION as _RP_VERSION

VERSION = "grain-psychometric-open-stream-v2"


def _as_scalar_or_vec(val: Any) -> np.ndarray:
    arr = np.asarray(val, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim != 1 or not np.isfinite(arr).all():
        raise ValueError("finite scalar or 1D coordinate vector required")
    return arr


def _wilson_interval(k: int, n: int, confidence: float) -> tuple[float, float, float]:
    if n <= 0 or k < 0 or k > n:
        raise ValueError("invalid binomial counts for reader calibration")
    p_hat = float(k) / float(n)
    z = float(norm.ppf(0.5 + 0.5 * confidence))
    z2 = z * z
    denom = 1.0 + z2 / float(n)
    center = (p_hat + 0.5 * z2 / float(n)) / denom
    radius = (
        z
        * math.sqrt((p_hat * (1.0 - p_hat) + 0.25 * z2 / float(n)) / float(n))
        / denom
    )
    return p_hat, max(0.0, center - radius), min(1.0, center + radius)


def _parse_reader_spec(spec: Mapping[str, Any], confidence: float) -> dict[str, Any]:
    if {"tp", "fp", "tn", "fn"}.issubset(spec):
        tp, fp, tn, fn = (int(spec[k]) for k in ("tp", "fp", "tn", "fn"))
        if min(tp, fp, tn, fn) < 0 or (tp + fn) <= 0 or (fp + tn) <= 0:
            raise ValueError("confusion counts require positive positives and negatives")
        s_pt, s_lo, s_hi = _wilson_interval(tp, tp + fn, confidence)
        f_pt, f_lo, f_hi = _wilson_interval(fp, fp + tn, confidence)
        return {
            "sensitivity": [float(s_lo), float(s_hi)],
            "fpr": [float(f_lo), float(f_hi)],
            "sensitivity_point": float(s_pt),
            "fpr_point": float(f_pt),
        }

    if "sensitivity" not in spec or "fpr" not in spec:
        raise ValueError(
            "reader_calibration requires either {tp, fp, tn, fn} or {sensitivity, fpr}"
        )

    def _bounds(raw: Any, name: str) -> tuple[float, float, float]:
        if isinstance(raw, (int, float)):
            v = float(raw)
            if not math.isfinite(v) or not (0.0 <= v <= 1.0):
                raise ValueError(f"invalid {name} probability")
            return v, v, v
        seq = list(raw)
        if len(seq) != 2:
            raise ValueError(f"{name} interval must have length 2")
        lo, hi = float(seq[0]), float(seq[1])
        if (
            not (math.isfinite(lo) and math.isfinite(hi))
            or lo < 0.0
            or hi > 1.0
            or lo > hi
        ):
            raise ValueError(f"invalid {name} interval bounds")
        return 0.5 * (lo + hi), lo, hi

    s_pt, s_lo, s_hi = _bounds(spec["sensitivity"], "sensitivity")
    f_pt, f_lo, f_hi = _bounds(spec["fpr"], "fpr")
    return {
        "sensitivity": [s_lo, s_hi],
        "fpr": [f_lo, f_hi],
        "sensitivity_point": s_pt,
        "fpr_point": f_pt,
    }


def _rogan_gladen_box_bounds(
    p_lo: float, p_hi: float, s_lo: float, s_hi: float, f_lo: float, f_hi: float
) -> tuple[list[float], bool]:
    if s_lo <= f_hi:
        return [0.0, 1.0], False
    # g(p, s, f) = (p - f) / (s - f) is monotone in each variable separately on
    # s > f, so extrema over the compact box lie among the 8 box vertices (clipped to [0, 1]).
    vals = [
        min(1.0, max(0.0, (p - f) / (s - f)))
        for p, s, f in itertools.product((p_lo, p_hi), (s_lo, s_hi), (f_lo, f_hi))
    ]
    return [float(min(vals)), float(max(vals))], True


def calibrate_reader_misclassification(
    observed_cards_or_rates: Mapping[str, Any],
    reader_calibration: Mapping[str, Mapping[str, Any]],
    *,
    weights: Mapping[str, float] | None = None,
    contrast: Sequence[str] | None = None,
    confidence: float = 0.95,
    calibration_denominator: str | None = None,
) -> dict[str, Any]:
    """Invert Rogan-Gladen misclassification bias and decompose condition-dependent Reader DIF."""
    if not (0.0 < float(confidence) < 1.0):
        raise ValueError("confidence must lie strictly in (0, 1)")
    if calibration_denominator not in (None, "among_replies"):
        raise ValueError("only calibration on observed replies is supported")
    frame = (
        observed_cards_or_rates.get("denominator")
        if "conditions" in observed_cards_or_rates
        else None
    )
    if "conditions" in observed_cards_or_rates and frame not in (
        "among_replies",
        "per_opportunity",
    ):
        raise ValueError("response profile needs a recognized denominator")
    if frame == "per_opportunity":
        if calibration_denominator is None:
            raise ValueError(
                "per_opportunity card requires explicit calibration_denominator"
            )
        return _calibrate_opportunity_card(
            observed_cards_or_rates, reader_calibration, weights, contrast, confidence
        )
    # Accept either response_profile card or flat {condition: scalar|[lo, hi]} map
    if "conditions" in observed_cards_or_rates and isinstance(
        observed_cards_or_rates["conditions"], Mapping
    ):
        cond_map = observed_cards_or_rates["conditions"]
        raw_rates: dict[str, list[float]] = {}
        obs_shares: dict[str, float] = {}
        for c, row in cond_map.items():
            rb = row.get("reading_bounds")
            if rb is not None and row.get("n", 0) > 0:
                raw_rates[str(c)] = [float(rb[0]), float(rb[1])]
                if row.get("observed_share") is not None:
                    obs_shares[str(c)] = float(row["observed_share"])
        if weights is None:
            weights = {
                str(c): float(row["reference_weight"])
                for c, row in cond_map.items()
                if float(row.get("reference_weight", 0.0)) > 0.0
            } or None
    else:
        raw_rates = {}
        obs_shares = {}
        for c, val in observed_cards_or_rates.items():
            if isinstance(val, (int, float)):
                v = float(val)
                raw_rates[str(c)] = [v, v]
            else:
                seq = list(val)
                if len(seq) != 2:
                    raise ValueError(
                        "each observed condition rate must be scalar or [lo, hi]"
                    )
                raw_rates[str(c)] = [float(seq[0]), float(seq[1])]

    if not raw_rates:
        raise ValueError("at least one observed condition rate required")

    cal_conds: dict[str, Any] = {}
    true_bounds: dict[str, list[float]] = {}
    for c, (p_lo, p_hi) in sorted(raw_rates.items()):
        if (
            not (math.isfinite(p_lo) and math.isfinite(p_hi))
            or p_lo < 0.0
            or p_hi > 1.0
            or p_lo > p_hi
        ):
            raise ValueError(f"invalid observed rate bounds for condition {c!r}")
        if c not in reader_calibration:
            raise ValueError(f"missing reader_calibration entry for condition {c!r}")
        parsed = _parse_reader_spec(reader_calibration[c], float(confidence))
        s_lo, s_hi = parsed["sensitivity"]
        f_lo, f_hi = parsed["fpr"]
        s_pt, f_pt = parsed["sensitivity_point"], parsed["fpr_point"]
        p_mid = 0.5 * (p_lo + p_hi)
        j_pt = s_pt - f_pt
        bounds, identified = _rogan_gladen_box_bounds(
            p_lo, p_hi, s_lo, s_hi, f_lo, f_hi
        )
        if not identified:
            status = "reader_unidentified"
            b_point_unclipped = None
            b_point = None
        else:
            status = "identified"
            b_point_unclipped = (p_mid - f_pt) / j_pt
            b_point = min(1.0, max(0.0, b_point_unclipped))
        true_bounds[c] = bounds
        cal_conds[c] = {
            "observed_bounds": [float(p_lo), float(p_hi)],
            "observed_mid": float(p_mid),
            "sensitivity": parsed["sensitivity"],
            "fpr": parsed["fpr"],
            "sensitivity_point": float(s_pt),
            "fpr_point": float(f_pt),
            "youden_j_point": float(j_pt),
            "corrected_bounds": bounds,
            "corrected_point_clipped": b_point,
            "corrected_point_unclipped": b_point_unclipped,
            "status": status,
        }

    if frame == "among_replies":
        for c in cond_map.keys() - raw_rates.keys():
            true_bounds[c] = [0.0, 1.0]
            cal_conds[c] = {"status": "unsupported", "corrected_bounds": [0.0, 1.0]}

    def _weighted_interval(w_map: Mapping[str, float]) -> list[float] | None:
        if not w_map:
            return None
        w_sum = math.fsum(float(v) for v in w_map.values())
        if w_sum <= 0.0:
            return None
        lo = math.fsum(
            (float(w) / w_sum) * true_bounds.get(str(c), [0.0, 1.0])[0]
            for c, w in w_map.items()
        )
        hi = math.fsum(
            (float(w) / w_sum) * true_bounds.get(str(c), [0.0, 1.0])[1]
            for c, w in w_map.items()
        )
        return [float(lo), float(hi)]

    standardized = _weighted_interval(weights) if weights is not None else None
    natural = _weighted_interval(obs_shares) if obs_shares else None
    mix_gap = None
    if weights and obs_shares:
        from .response_profile import _linear

        domain = set(weights) | set(obs_shares)
        mass = math.fsum(weights.values())
        if any(not math.isfinite(w) or w < 0 for w in weights.values()) or mass <= 0:
            raise ValueError("invalid reference weights")
        mix_gap = _linear(
            {c: obs_shares.get(c, 0.0) - weights.get(c, 0.0) / mass for c in domain},
            {c: true_bounds.get(c, [0.0, 1.0]) for c in domain},
        )

    contrast_out = None
    if contrast is not None:
        if len(contrast) != 2:
            raise ValueError(
                "contrast must specify [higher_condition, lower_condition]"
            )
        c1, c0 = str(contrast[0]), str(contrast[1])
        if c1 == c0:
            raise ValueError("contrast needs distinct conditions")
        if c1 not in cal_conds or c0 not in cal_conds:
            raise ValueError("contrast conditions must appear in observed rates")
        r1, r0 = cal_conds[c1], cal_conds[c0]
        b1_lo, b1_hi = r1["corrected_bounds"]
        b0_lo, b0_hi = r0["corrected_bounds"]
        corrected_contrast_bounds = [float(b1_lo - b0_hi), float(b1_hi - b0_lo)]
        if (
            r1["status"] == "identified"
            and r0["status"] == "identified"
            and r1["corrected_point_unclipped"] is not None
            and r0["corrected_point_unclipped"] is not None
        ):
            p1, p0 = r1["observed_mid"], r0["observed_mid"]
            j1, j0 = r1["youden_j_point"], r0["youden_j_point"]
            f1, f0 = r1["fpr_point"], r0["fpr_point"]
            bu1 = float(r1["corrected_point_unclipped"])
            bu0 = float(r0["corrected_point_unclipped"])
            obs_diff = p1 - p0
            scaled_true = 0.5 * (j1 + j0) * (bu1 - bu0)
            uniform_fpr = f1 - f0
            nonunif_disc = (j1 - j0) * 0.5 * (bu1 + bu0)
            closure_err = abs(obs_diff - (scaled_true + uniform_fpr + nonunif_disc))
            dif_decomp = {
                "observed_contrast": float(obs_diff),
                "unclipped_corrected_contrast": float(bu1 - bu0),
                "clipped_corrected_contrast": float(
                    r1["corrected_point_clipped"] - r0["corrected_point_clipped"]
                ),
                "scaled_true_contrast": float(scaled_true),
                "uniform_fpr_dif": float(uniform_fpr),
                "nonuniform_discrimination_dif": float(nonunif_disc),
                "closure_error": float(closure_err),
            }
        elif "unsupported" in (r1["status"], r0["status"]):
            dif_decomp = {"status": "unsupported"}
        else:
            dif_decomp = {
                "observed_contrast": float(r1["observed_mid"] - r0["observed_mid"]),
                "status": "reader_unidentified",
            }
        contrast_out = {
            "pair": [c1, c0],
            "corrected_bounds": corrected_contrast_bounds,
            "dif_decomposition": dif_decomp,
        }

    return {
        "schema": VERSION,
        "upstream_response_profile_schema": _RP_VERSION,
        "calibration_denominator": "among_replies",
        "denominator": frame or "among_replies",
        "confidence": float(confidence),
        "conditions": cal_conds,
        "natural_corrected_bounds": natural,
        "mix_gap_corrected_bounds": mix_gap,
        "standardized_corrected_bounds": standardized,
        "contrast_calibration": contrast_out,
    }


def _calibrate_opportunity_card(card, calibration, weights, contrast, confidence):
    """Invert the reader on confirmed speech only, then restore opportunity mass."""
    from .response_profile import _linear

    rows = card["conditions"]
    weights = dict(
        weights
        if weights is not None
        else {c: r["reference_weight"] for c, r in rows.items()}
    )
    if (
        not weights
        or any(not math.isfinite(w) or w < 0 for w in weights.values())
        or not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-12, rel_tol=0)
    ):
        raise ValueError("opportunity reference weights must sum to one")
    cells, detail, ns = {}, {}, {}
    for c in sorted(set(rows) | set(weights)):
        row = rows.get(c, {})
        counts = row.get("counts", {})
        k, neg, amb, silent, unknown = (
            counts.get(s, 0)
            for s in ("support", "refute", "unresolved", "no_reply", "reply_unknown")
        )
        if any(type(v) is not int or v < 0 for v in (k, neg, amb, silent, unknown)):
            raise ValueError("invalid opportunity counts")
        spoken = k + neg + amb
        n = spoken + silent + unknown
        if n != row.get("n", 0) or spoken != row.get("confirmed_replies", 0):
            raise ValueError("opportunity counts disagree with denominators")
        reading = None
        if spoken:
            reading = calibrate_reader_misclassification(
                {c: [k / spoken, (k + amb) / spoken]},
                calibration,
                confidence=confidence,
            )["conditions"][c]
        lo, hi = reading["corrected_bounds"] if reading else (0.0, 1.0)
        # A missed reply may be absent or may contain the behavior; a known silence cannot.
        bounds = [spoken * lo / n, (spoken * hi + unknown) / n] if n else [0.0, 1.0]
        status = (
            "unsupported"
            if not n
            else "unknown_reply"
            if unknown
            else reading["status"]
            if reading
            else "confirmed_silence"
        )
        cells[c], ns[c] = bounds, n
        detail[c] = {
            "status": status,
            "corrected_bounds": bounds,
            "reply_reader": reading,
            "confirmed_replies": spoken,
            "unknown_reply": unknown,
            "n": n,
        }
    total = sum(ns.values())
    q = {c: n / total for c, n in ns.items()} if total else {}
    pair = None
    if contrast is not None:
        if (
            len(contrast) != 2
            or contrast[0] == contrast[1]
            or any(c not in cells for c in contrast)
        ):
            raise ValueError("expected two distinct known conditions")
        pair = {
            "pair": list(contrast),
            "corrected_bounds": _linear({contrast[0]: 1.0, contrast[1]: -1.0}, cells),
            "dif_decomposition": {"status": "reply_dif_not_opportunity_dif"},
        }
    return {
        "schema": VERSION,
        "upstream_response_profile_schema": _RP_VERSION,
        "confidence": confidence,
        "denominator": "per_opportunity",
        "calibration_denominator": "among_replies",
        "conditions": detail,
        "natural_corrected_bounds": _linear(q, cells) if q else None,
        "standardized_corrected_bounds": _linear(weights, cells),
        "mix_gap_corrected_bounds": _linear(
            {c: q[c] - weights.get(c, 0.0) for c in cells}, cells
        )
        if q
        else None,
        "unconfirmed_excluded": card.get("unconfirmed_excluded", 0),
        "contrast_calibration": pair,
    }


def _fit_2d_caps_prior(
    records: Sequence[Mapping[str, Any]],
    *,
    event_var_c1: float,
    event_var_c0: float,
) -> dict[str, Any]:
    if len(records) < 3:
        raise ValueError("at least 3 paired focal units required to fit 2D CAPS prior")
    y = np.array(
        [[float(r["mean_1"]), float(r["mean_0"])] for r in records], dtype=np.float64
    )
    n_mat = np.array(
        [[float(r["n_1"]), float(r["n_0"])] for r in records], dtype=np.float64
    )
    if not (np.isfinite(y).all() and np.isfinite(n_mat).all()) or np.any(n_mat <= 0.0):
        raise ValueError("finite condition means and positive counts required")
    v_bar = np.array(
        [
            float(event_var_c1) * float(np.mean(1.0 / n_mat[:, 0])),
            float(event_var_c0) * float(np.mean(1.0 / n_mat[:, 1])),
        ],
        dtype=np.float64,
    )
    mu = np.mean(y, axis=0)
    raw_cov = np.cov(y, rowvar=False, ddof=1)
    eiv_cov = 0.5 * (raw_cov + raw_cov.T) - np.diag(v_bar)
    evals, evecs = np.linalg.eigh(eiv_cov)
    psd_evals = np.maximum(evals, 0.0)
    sigma_person = (evecs * psd_evals) @ evecs.T
    sigma_person = 0.5 * (sigma_person + sigma_person.T)
    return {
        "mu": mu,
        "raw_cov": raw_cov,
        "raw_eiv_evals": evals,
        "sigma_person": sigma_person,
        "mean_diag_noise": v_bar,
    }


def estimate_bivariate_caps_reliability(
    records: Sequence[Mapping[str, Any]],
    *,
    event_var_c1: float,
    event_var_c0: float,
    weights: Sequence[float] = (0.5, 0.5),
    min_reliability: float = 0.30,
    confidence: float = 0.95,
    fit_records: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """2D Kelley Empirical-Bayes CAPS estimator with variable-length G-coefficients & MDD."""
    ev1, ev0 = float(event_var_c1), float(event_var_c0)
    if not (math.isfinite(ev1) and math.isfinite(ev0)) or ev1 < 0.0 or ev0 < 0.0:
        raise ValueError("non-negative finite event_var_c1 and event_var_c0 required")
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (2,) or not np.isfinite(w).all() or np.any(w < 0.0) or float(np.sum(w)) <= 0.0:
        raise ValueError("non-negative 2-vector weights with positive sum required")
    w = w / float(np.sum(w))
    if not (0.0 < float(min_reliability) < 1.0) or not (0.0 < float(confidence) < 1.0):
        raise ValueError("min_reliability and confidence must lie in (0, 1)")
    if not records:
        raise ValueError("non-empty records required")

    prior = _fit_2d_caps_prior(
        fit_records if fit_records is not None else records,
        event_var_c1=ev1,
        event_var_c0=ev0,
    )
    mu = prior["mu"]
    sigma = prior["sigma_person"]
    c_vec = np.array([1.0, -1.0], dtype=np.float64)
    var_std_lat = float(max(0.0, w @ sigma @ w))
    var_con_lat = float(max(0.0, c_vec @ sigma @ c_vec))
    z_crit = float(norm.ppf(0.5 + 0.5 * float(confidence)))

    persons: list[dict[str, Any]] = []
    seen_focal: set[str] = set()
    for r in records:
        focal = str(r["focal"])
        if focal in seen_focal:
            raise ValueError(f"duplicate focal unit {focal!r} in panel")
        seen_focal.add(focal)
        y_u = np.array([float(r["mean_1"]), float(r["mean_0"])], dtype=np.float64)
        n1, n0 = float(r["n_1"]), float(r["n_0"])
        if not np.isfinite(y_u).all() or n1 <= 0.0 or n0 <= 0.0:
            raise ValueError(f"invalid counts/means for focal {focal!r}")
        d_u = np.diag([ev1 / n1, ev0 / n0])
        total_u = sigma + d_u
        # Use pinv if both sigma and d_u degenerate (e.g. event_var == 0 with rank-0 sigma)
        if np.linalg.cond(total_u) < 1e12:
            gain = sigma @ np.linalg.inv(total_u)
        else:
            gain = sigma @ np.linalg.pinv(total_u, rcond=1e-12)
        theta_eb = mu + gain @ (y_u - mu)
        post_cov = sigma - gain @ sigma
        post_cov = 0.5 * (post_cov + post_cov.T)

        std_raw = float(w @ y_u)
        std_eb = float(w @ theta_eb)
        con_raw = float(c_vec @ y_u)
        con_eb = float(c_vec @ theta_eb)

        sem_std_raw = float(math.sqrt(max(0.0, float(w @ d_u @ w))))
        sem_std_eb = float(math.sqrt(max(0.0, float(w @ post_cov @ w))))
        sem_con_raw = float(math.sqrt(max(0.0, float(c_vec @ d_u @ c_vec))))
        sem_con_eb = float(math.sqrt(max(0.0, float(c_vec @ post_cov @ c_vec))))

        err_std_u = float(w @ d_u @ w)
        err_con_u = float(c_vec @ d_u @ c_vec)
        rho_std = (
            float(var_std_lat / (var_std_lat + err_std_u))
            if (var_std_lat + err_std_u) > 0.0
            else 0.0
        )
        rho_con = (
            float(var_con_lat / (var_con_lat + err_con_u))
            if (var_con_lat + err_con_u) > 0.0
            else 0.0
        )
        mdd_con = z_crit * math.sqrt(2.0) * sem_con_raw
        if var_con_lat <= 1e-12:
            status = "no_latent_contrast_variance"
        elif rho_con >= float(min_reliability):
            status = "reportable"
        else:
            status = "underpowered_for_individual_reporting"

        persons.append(
            {
                "focal": focal,
                "n_1": n1,
                "n_0": n0,
                "raw_standardized": std_raw,
                "eb_standardized": std_eb,
                "raw_contrast": con_raw,
                "eb_contrast": con_eb,
                "sem_raw_standardized": sem_std_raw,
                "sem_eb_standardized": sem_std_eb,
                "sem_raw_contrast": sem_con_raw,
                "sem_eb_contrast": sem_con_eb,
                "reliability_standardized": rho_std,
                "reliability_contrast": rho_con,
                "mdd_contrast": float(mdd_con),
                "reporting_status": status,
            }
        )

    mean_rho_std = float(np.mean([p["reliability_standardized"] for p in persons]))
    mean_rho_con = float(np.mean([p["reliability_contrast"] for p in persons]))
    n_reportable = sum(1 for p in persons if p["reporting_status"] == "reportable")
    return {
        "schema": VERSION,
        "prior_source": "disjoint_fit_records" if fit_records is not None else "in_panel_moments",
        "mu_population": [float(mu[0]), float(mu[1])],
        "sigma_person": [[float(sigma[0, 0]), float(sigma[0, 1])], [float(sigma[1, 0]), float(sigma[1, 1])]],
        "raw_eiv_eigenvalues": [float(v) for v in prior["raw_eiv_evals"]],
        "latent_standardized_variance": var_std_lat,
        "latent_contrast_variance": var_con_lat,
        "mean_reliability_standardized": mean_rho_std,
        "mean_reliability_contrast": mean_rho_con,
        "n_persons": len(persons),
        "n_reportable_contrast": int(n_reportable),
        "reportable_share": float(n_reportable / len(persons)),
        "min_reliability": float(min_reliability),
        "persons": persons,
    }


def deattenuate_cross_panel_correlation(
    panel_a: Mapping[str, Any],
    panel_b: Mapping[str, Any],
    *,
    metric: str = "contrast",
    n_perm: int = 300,
    seed: int = 0,
) -> dict[str, Any]:
    """Cross-panel correlation with Spearman/Rouder (PR14) unclipped de-attenuation & stranger null."""
    if metric not in ("contrast", "standardized"):
        raise ValueError("metric must be 'contrast' or 'standardized'")
    if n_perm < 200:
        raise ValueError("G3 requires n_perm >= 200 for empirical stranger null")

    map_a = {row["focal"]: row for row in panel_a["persons"]}
    map_b = {row["focal"]: row for row in panel_b["persons"]}
    shared = sorted(set(map_a) & set(map_b))
    if len(shared) < 3:
        raise ValueError("at least 3 shared focal units required across panels")

    raw_key = f"raw_{metric}"
    eb_key = f"eb_{metric}"
    rel_key = f"reliability_{metric}"

    xa_raw = np.array([map_a[u][raw_key] for u in shared], dtype=np.float64)
    xb_raw = np.array([map_b[u][raw_key] for u in shared], dtype=np.float64)
    xa_eb = np.array([map_a[u][eb_key] for u in shared], dtype=np.float64)
    xb_eb = np.array([map_b[u][eb_key] for u in shared], dtype=np.float64)
    rho_a = float(np.mean([map_a[u][rel_key] for u in shared]))
    rho_b = float(np.mean([map_b[u][rel_key] for u in shared]))

    def _corr(u: np.ndarray, v: np.ndarray) -> float:
        uc, vc = u - np.mean(u), v - np.mean(v)
        den = float(np.linalg.norm(uc) * np.linalg.norm(vc))
        return float(np.dot(uc, vc) / den) if den > 1e-12 else 0.0

    r_raw = _corr(xa_raw, xb_raw)
    r_eb = _corr(xa_eb, xb_eb)
    geom_rel = math.sqrt(max(0.0, rho_a * rho_b))
    r_deatt = (r_raw / geom_rel) if geom_rel > 1e-12 else None

    rng = np.random.default_rng(int(seed))
    ge_raw = 0
    ge_eb = 0
    null_raw = []
    for _ in range(int(n_perm)):
        perm = rng.permutation(len(shared))
        nr = _corr(xa_raw, xb_raw[perm])
        ne = _corr(xa_eb, xb_eb[perm])
        null_raw.append(nr)
        if abs(nr) >= abs(r_raw) - 1e-12:
            ge_raw += 1
        if abs(ne) >= abs(r_eb) - 1e-12:
            ge_eb += 1

    return {
        "schema": VERSION,
        "metric": metric,
        "n_shared": len(shared),
        "pearson_r_raw": float(r_raw),
        "pearson_r_eb": float(r_eb),
        "panel_a_mean_reliability": rho_a,
        "panel_b_mean_reliability": rho_b,
        "geometric_mean_reliability": float(geom_rel),
        "r_deattenuated_unclipped": float(r_deatt) if r_deatt is not None else None,
        "exceeds_unity_misspecification_flag": bool(
            r_deatt is not None and abs(r_deatt) > 1.0
        ),
        "stranger_null_perm_p_raw": float((ge_raw + 1) / (int(n_perm) + 1)),
        "stranger_null_perm_p_eb": float((ge_eb + 1) / (int(n_perm) + 1)),
        "stranger_null_median_r_raw": float(np.median(null_raw)),
    }


def fit_ct_lsto_variance_components(
    timestamps: Sequence[float] | np.ndarray,
    author_ids: Sequence[Any],
    condition_ids: Sequence[Any],
    values: Sequence[float] | np.ndarray,
    *,
    max_pairs_per_key: int = 64,
    n_starts: int = 6,
) -> dict[str, Any]:
    """Multi-start REML-style semivariogram decomposition into CT-LSTO trait/state/occasion."""
    ts = np.asarray(timestamps, dtype=np.float64)
    vals = np.asarray(values, dtype=np.float64)
    if (
        ts.ndim != 1
        or vals.ndim != 1
        or len(ts) != len(vals)
        or len(ts) != len(author_ids)
        or len(ts) != len(condition_ids)
        or len(ts) < 4
        or not (np.isfinite(ts).all() and np.isfinite(vals).all())
    ):
        raise ValueError("aligned finite 1D timestamps, keys, and values (>=4) required")
    if n_starts < 5:
        raise ValueError("G5 requires at least 5 optimizer starts")

    groups: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for t, au, co, v in zip(ts, author_ids, condition_ids, vals, strict=True):
        groups[(str(au), str(co))].append((float(t), float(v)))

    dts: list[float] = []
    semis: list[float] = []
    adj_dts: list[float] = []
    key_counts: dict[tuple[str, str], int] = {}
    for key, pairs in groups.items():
        if len(pairs) < 2:
            continue
        pairs.sort(key=lambda item: item[0])
        key_counts[key] = len(pairs)
        t_g = np.array([p[0] for p in pairs], dtype=np.float64)
        v_g = np.array([p[1] for p in pairs], dtype=np.float64)
        adj_diffs = np.diff(t_g)
        adj_dts.extend(float(d) for d in adj_diffs if d > 0.0)
        # Deterministic evenly strided within-key lag pairs capped at max_pairs_per_key
        m = len(pairs)
        idx_pairs = [
            (i, j)
            for i in range(m)
            for j in range(i + 1, m)
            if t_g[j] > t_g[i]
        ]
        if len(idx_pairs) > max_pairs_per_key:
            step = len(idx_pairs) / float(max_pairs_per_key)
            idx_pairs = [idx_pairs[int(k * step)] for k in range(max_pairs_per_key)]
        for i, j in idx_pairs:
            dts.append(float(t_g[j] - t_g[i]))
            semis.append(float(0.5 * (v_g[j] - v_g[i]) ** 2))

    if len(dts) < 3:
        raise ValueError("need at least 3 positive within-(author, condition) time lags")

    dt_arr = np.asarray(dts, dtype=np.float64)
    gamma_arr = np.asarray(semis, dtype=np.float64)
    sigma_total_sq = float(np.var(vals, ddof=1))

    dt_pos = np.sort(dt_arr)
    dt_min = max(1e-6, float(dt_pos[0]))
    dt_max = max(dt_min * 10.0, float(dt_pos[-1]))
    lam_lo = math.log(2.0) / (10.0 * dt_max)
    lam_hi = math.log(2.0) / (0.1 * dt_min)
    log_bounds = [(math.log(lam_lo), math.log(lam_hi))]

    # Use binned lag means weighted by bin counts so semivariogram NNLS doesn't overfit noisy single pairs
    n_bins = min(12, max(4, int(math.sqrt(len(dt_arr)))))
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(dt_arr, qs)
    bin_dt_list: list[float] = []
    bin_gamma_list: list[float] = []
    bin_w_list: list[float] = []
    for bi in range(n_bins):
        if bi < n_bins - 1:
            mask = (dt_arr >= edges[bi]) & (dt_arr < edges[bi + 1])
        else:
            mask = (dt_arr >= edges[bi]) & (dt_arr <= edges[bi + 1])
        if np.any(mask):
            bin_dt_list.append(float(np.mean(dt_arr[mask])))
            bin_gamma_list.append(float(np.mean(gamma_arr[mask])))
            bin_w_list.append(float(np.sum(mask)))
    b_dt = np.asarray(bin_dt_list, dtype=np.float64)
    b_gam = np.asarray(bin_gamma_list, dtype=np.float64)
    b_sqrt_w = np.sqrt(np.asarray(bin_w_list, dtype=np.float64))
    occ_flat_global = max(
        0.0, float(np.sum((b_sqrt_w ** 2) * b_gam) / max(1e-15, float(np.sum(b_sqrt_w ** 2))))
    )

    def _fit_occ_state(lam: float) -> tuple[float, float, float]:
        phi = 1.0 - np.exp(-lam * b_dt)
        # Solve min_{occ>=0, state>=0} || sqrt(w) * (occ + state * phi - gamma) ||^2
        x_mat = np.column_stack([b_sqrt_w, b_sqrt_w * phi])
        y_vec = b_sqrt_w * b_gam
        candidates = [(0.0, 0.0)]
        sol_u, *_ = np.linalg.lstsq(x_mat, y_vec, rcond=None)
        if sol_u[0] >= 0.0 and sol_u[1] >= 0.0:
            candidates.append((float(sol_u[0]), float(sol_u[1])))
        den_p = float(np.dot(b_sqrt_w * phi, b_sqrt_w * phi))
        if den_p > 1e-15:
            candidates.append((0.0, max(0.0, float(np.dot(b_sqrt_w * phi, y_vec) / den_p))))
        candidates.append((occ_flat_global, 0.0))
        best_sse = float("inf")
        best_pair = (occ_flat_global, 0.0)
        for o, s in candidates:
            res = b_sqrt_w * ((o + s * phi) - b_gam)
            sse = float(np.dot(res, res)) + 1e-4 * s * s
            if sse < best_sse:
                best_sse = sse
                best_pair = (o, s)
        return best_pair[0], best_pair[1], best_sse

    starts = np.linspace(log_bounds[0][0], log_bounds[0][1], int(n_starts))
    best_opt = None
    best_obj = float("inf")
    for s0 in starts:
        res = minimize(
            lambda z: _fit_occ_state(math.exp(float(z[0])))[2],
            x0=np.array([float(s0)]),
            bounds=log_bounds,
            method="L-BFGS-B",
        )
        lam_c = math.exp(float(res.x[0]))
        occ_c, st_c, sse_c = _fit_occ_state(lam_c)
        if sse_c < best_obj:
            best_obj = sse_c
            best_opt = (lam_c, occ_c, st_c, sse_c)

    # Check finite-sample model selection against pure flat i.i.d. occasion noise
    sse_flat = float(np.sum((b_sqrt_w * (occ_flat_global - b_gam)) ** 2))
    lam_hat, occ_sq, state_sq, sse_best = best_opt
    # Only accept dynamic state component when binned SSE improves materially (>15%) over flat occ noise
    if sse_best > 0.85 * sse_flat:
        occ_sq = occ_flat_global
        state_sq = 0.0
    # Cap within-key variance sum at total variance when finite-sample within > between
    raw_within = occ_sq + state_sq
    if raw_within > sigma_total_sq and raw_within > 1e-15:
        scale = sigma_total_sq / raw_within
        occ_sq *= scale
        state_sq *= scale
        trait_sq = 0.0
    else:
        trait_sq = max(0.0, sigma_total_sq - occ_sq - state_sq)

    half_life = math.log(2.0) / max(lam_hat, 1e-15)
    median_adj_dt = float(np.median(adj_dts)) if adj_dts else float(np.median(dt_arr))
    active_frac = float(np.mean(np.asarray(adj_dts or dt_arr) <= half_life))
    naive_hurts = bool(occ_sq > state_sq + 1e-12)
    critical_k = (
        math.ceil(occ_sq / state_sq)
        if state_sq > 1e-12 and occ_sq > state_sq
        else (1 if state_sq > 1e-12 else None)
    )
    alpha_med = math.exp(-lam_hat * median_adj_dt)
    if (state_sq + occ_sq) > 1e-12:
        kalman_gain_med = float(
            ((state_sq * alpha_med) ** 2) / ((state_sq + occ_sq) ** 2)
        )
    else:
        kalman_gain_med = 0.0

    regime = (
        "bursty_state_trait_active"
        if (state_sq > 0.1 * sigma_total_sq and active_frac >= 0.25 and state_sq >= occ_sq)
        else "trait_plus_iid_occasion"
    )
    denom_tot = max(sigma_total_sq, 1e-15)
    return {
        "schema": VERSION,
        "n_events": len(ts),
        "n_keys_with_repeats": len(key_counts),
        "n_lag_pairs": len(dt_arr),
        "sigma_total_sq": float(sigma_total_sq),
        "sigma_trait_sq": float(trait_sq),
        "sigma_state_sq": float(state_sq),
        "sigma_occ_sq": float(occ_sq),
        "trait_share": float(trait_sq / denom_tot),
        "state_share": float(state_sq / denom_tot),
        "occasion_share": float(occ_sq / denom_tot),
        "decay_lambda": float(lam_hat),
        "state_half_life": float(half_life),
        "median_adjacent_dt": float(median_adj_dt),
        "active_gap_fraction": float(active_frac),
        "naive_carry_hurts_when_occ_exceeds_state": naive_hurts,
        "critical_pool_size_k_for_naive_carry": critical_k,
        "expected_kalman_relative_gain_at_median_dt": kalman_gain_med,
        "kalman_shrinkage_weight_single_event": float(
            (state_sq * alpha_med) / (state_sq + occ_sq)
        )
        if (state_sq + occ_sq) > 1e-12
        else 0.0,
        "regime": regime,
    }


def measure_irtree_selection_expression(
    timestamps: Sequence[float] | np.ndarray,
    author_ids: Sequence[Any],
    condition_ids: Sequence[Any],
    values: Sequence[float] | np.ndarray,
    *,
    window_edges: Sequence[float] | np.ndarray,
    min_active_windows: int = 4,
    eta_min: float = 0.5,
    n_perm: int = 300,
    seed: int = 0,
) -> dict[str, Any]:
    """Two-node IRTree measurement: Node 1 (Venue Selection) x Node 2 (Conditional Expression)."""
    ts = np.asarray(timestamps, dtype=np.float64)
    vals = np.asarray(values, dtype=np.float64)
    edges = np.asarray(window_edges, dtype=np.float64)
    if (
        ts.ndim != 1
        or vals.ndim != 1
        or len(ts) != len(vals)
        or len(ts) != len(author_ids)
        or len(ts) != len(condition_ids)
        or edges.ndim != 1
        or len(edges) < 3
        or np.any(np.diff(edges) <= 0.0)
    ):
        raise ValueError("aligned streams and strictly increasing window_edges (>=2 bins) required")
    if not (0.0 < float(eta_min) <= 1.0) or min_active_windows < 2 or n_perm < 200:
        raise ValueError("eta_min in (0,1], min_active_windows >= 2, and n_perm >= 200 required")

    n_win = len(edges) - 1
    win_idx = np.searchsorted(edges, ts, side="right") - 1
    all_conds = sorted({str(c) for c in condition_ids})
    c_idx = {c: idx for idx, c in enumerate(all_conds)}
    m_c = len(all_conds)
    if m_c < 2:
        raise ValueError("at least 2 distinct conditions required for IRTree profile")

    # Bucket (win, author) -> active set, and (win, author, cond) -> values
    win_author_active: list[set[str]] = [set() for _ in range(n_win)]
    bucket_vals: dict[tuple[int, str, str], list[float]] = defaultdict(list)
    author_active_wins: dict[str, set[int]] = defaultdict(set)
    for i in range(len(ts)):
        w = int(win_idx[i])
        if 0 <= w < n_win:
            au = str(author_ids[i])
            co = str(condition_ids[i])
            win_author_active[w].add(au)
            author_active_wins[au].add(w)
            bucket_vals[(w, au, co)].append(float(vals[i]))

    eligible = sorted(au for au, ws in author_active_wins.items() if len(ws) >= min_active_windows)
    if len(eligible) < 3:
        raise ValueError("fewer than 3 authors meet min_active_windows")

    mid_w = n_win // 2

    def _profiles(win_range: range, authors: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        sel = np.zeros((len(authors), m_c), dtype=np.float64)
        expr = np.full((len(authors), m_c), np.nan, dtype=np.float64)
        valid_expr = np.zeros((len(authors), m_c), dtype=bool)
        card_rows: list[dict[str, Any]] = []
        for row_i, au in enumerate(authors):
            online_ws = [w for w in win_range if au in win_author_active[w]]
            n_on = len(online_ws)
            per_c: dict[str, Any] = {}
            for co in all_conds:
                j = c_idx[co]
                k_spoke = 0
                ev_vals: list[float] = []
                for w in online_ws:
                    vs = bucket_vals.get((w, au, co))
                    if vs:
                        k_spoke += 1
                        ev_vals.extend(vs)
                m_silent = n_on - k_spoke
                rate_full = float(k_spoke / n_on) if n_on > 0 else 0.0
                rate_hi = (
                    float(k_spoke / (k_spoke + float(eta_min) * m_silent))
                    if (k_spoke + float(eta_min) * m_silent) > 0.0
                    else 0.0
                )
                sel[row_i, j] = rate_full
                if ev_vals:
                    expr[row_i, j] = float(np.mean(ev_vals))
                    valid_expr[row_i, j] = True
                per_c[co] = {
                    "spoke_windows": k_spoke,
                    "online_silent_windows": m_silent,
                    "manski_participation_bounds": [rate_full, min(1.0, rate_hi)],
                    "mean_expression": float(expr[row_i, j]) if valid_expr[row_i, j] else None,
                    "n_events": len(ev_vals),
                }
            card_rows.append({"focal": au, "online_windows": n_on, "conditions": per_c})
        return sel, expr, valid_expr, card_rows

    sel_all, expr_all, _, cards_all = _profiles(range(n_win), eligible)
    sel_h1, expr_h1, v_h1, _ = _profiles(range(mid_w), eligible)
    sel_h2, expr_h2, v_h2, _ = _profiles(range(mid_w, n_win), eligible)

    def _mean_profile_corr(a_mat: np.ndarray, b_mat: np.ndarray, mask: np.ndarray | None) -> list[float]:
        out = []
        for i in range(len(a_mat)):
            if mask is not None:
                ok = mask[i]
                if int(np.sum(ok)) < 2:
                    continue
                u, v = a_mat[i, ok], b_mat[i, ok]
            else:
                u, v = a_mat[i], b_mat[i]
            uc, vc = u - np.mean(u), v - np.mean(v)
            den = float(np.linalg.norm(uc) * np.linalg.norm(vc))
            if den > 1e-12:
                out.append(float(np.dot(uc, vc) / den))
        return out

    # Flatten across (author, condition) pairs where both halves are active for global split-half
    def _stranger_null_corr(a_mat: np.ndarray, b_mat: np.ndarray, mask_a: np.ndarray | None = None, mask_b: np.ndarray | None = None) -> dict[str, Any]:
        if mask_a is None or mask_b is None:
            r_list = _mean_profile_corr(a_mat, b_mat, None)
            obs_r = float(np.mean(r_list)) if r_list else 0.0
            n_u = len(a_mat)
            rng = np.random.default_rng(int(seed))
            ge = 0
            null_rs = []
            for _ in range(int(n_perm)):
                perm = rng.permutation(n_u)
                nr_list = _mean_profile_corr(a_mat, b_mat[perm], None)
                nr = float(np.mean(nr_list)) if nr_list else 0.0
                null_rs.append(nr)
                if nr >= obs_r - 1e-12:
                    ge += 1
            return {
                "split_half_profile_r": obs_r,
                "n_authors_valid": len(r_list),
                "stranger_null_median_r": float(np.median(null_rs)),
                "stranger_null_perm_p": float((ge + 1) / (int(n_perm) + 1)),
            }

        # Expression profile: correlate across shared active (author, condition) cells
        both = mask_a & mask_b
        xa, xb = a_mat[both], b_mat[both]
        if len(xa) < 3:
            return {
                "split_half_r": None,
                "n_shared_cells": len(xa),
                "stranger_null_median_r": None,
                "stranger_null_perm_p": None,
            }
        uc, vc = xa - np.mean(xa), xb - np.mean(xb)
        den = float(np.linalg.norm(uc) * np.linalg.norm(vc))
        obs_r = float(np.dot(uc, vc) / den) if den > 1e-12 else 0.0
        rng = np.random.default_rng(int(seed) + 13)
        ge = 0
        null_rs = []
        n_u = len(a_mat)
        for _ in range(int(n_perm)):
            perm = rng.permutation(n_u)
            bm = mask_a & mask_b[perm]
            if int(np.sum(bm)) < 3:
                nr = 0.0
            else:
                pxa, pxb = a_mat[bm], b_mat[perm][bm]
                puc, pvc = pxa - np.mean(pxa), pxb - np.mean(pxb)
                pden = float(np.linalg.norm(puc) * np.linalg.norm(pvc))
                nr = float(np.dot(puc, pvc) / pden) if pden > 1e-12 else 0.0
            null_rs.append(nr)
            if nr >= obs_r - 1e-12:
                ge += 1
        return {
            "split_half_r": obs_r,
            "n_shared_cells": len(xa),
            "stranger_null_median_r": float(np.median(null_rs)),
            "stranger_null_perm_p": float((ge + 1) / (int(n_perm) + 1)),
        }

    sel_repro = _stranger_null_corr(sel_h1, sel_h2)
    expr_repro = _stranger_null_corr(expr_h1, expr_h2, v_h1, v_h2)

    valid_all = np.isfinite(expr_all)
    if int(np.sum(valid_all)) >= 3:
        s_vec = sel_all[valid_all] - np.mean(sel_all[valid_all])
        e_vec = expr_all[valid_all] - np.mean(expr_all[valid_all])
        c_den = float(np.linalg.norm(s_vec) * np.linalg.norm(e_vec))
        cross_node_r = float(np.dot(s_vec, e_vec) / c_den) if c_den > 1e-12 else 0.0
    else:
        cross_node_r = None

    return {
        "schema": VERSION,
        "conditions": all_conds,
        "eta_min": float(eta_min),
        "n_eligible_authors": len(eligible),
        "node1_selection_contiguous_repro": sel_repro,
        "node2_expression_contiguous_repro": expr_repro,
        "cross_node_selection_vs_expression_r": cross_node_r,
        "authors": cards_all,
    }


def decompose_hierarchical_cohort_person_mix(
    prev_cells: Mapping[tuple[Any, Any], tuple[float, Any]],
    cur_cells: Mapping[tuple[Any, Any], tuple[float, Any]],
) -> dict[str, Any]:
    """Exact 5-term Cohort Turnover x Within-Person Condition-Mix Kitagawa-Oaxaca identity."""
    if not prev_cells or not cur_cells:
        raise ValueError("non-empty prev_cells and cur_cells required")

    def _aggregate_authors(
        cells: Mapping[tuple[Any, Any], tuple[float, Any]],
    ) -> tuple[dict[str, dict[str, tuple[float, np.ndarray]]], dict[str, tuple[float, np.ndarray]], int]:
        by_author: dict[str, dict[str, tuple[float, np.ndarray]]] = defaultdict(dict)
        dim: int | None = None
        for (au, co), (cnt, val) in cells.items():
            c_f = float(cnt)
            v = _as_scalar_or_vec(val)
            if dim is None:
                dim = len(v)
            elif len(v) != dim:
                raise ValueError("inconsistent coordinate dimensions across cells")
            if not math.isfinite(c_f) or c_f <= 0.0:
                raise ValueError("positive cell counts required")
            by_author[str(au)][str(co)] = (c_f, v)
        assert dim is not None
        tot_map: dict[str, tuple[float, np.ndarray]] = {}
        for au, conds in by_author.items():
            w_tot = math.fsum(cnt for cnt, _ in conds.values())
            mean_u = np.sum(
                [(cnt / w_tot) * vec for cnt, vec in conds.values()], axis=0
            )
            tot_map[au] = (w_tot, mean_u)
        return by_author, tot_map, dim

    prev_by_au, prev_tot, dim_p = _aggregate_authors(prev_cells)
    cur_by_au, cur_tot, dim_c = _aggregate_authors(cur_cells)
    if dim_p != dim_c:
        raise ValueError("coordinate dimensions must match between prev_cells and cur_cells")
    dim = dim_p

    # Reuse 1D symmetric cohort decomposer dimension-by-dimension for outer terms
    # while computing exact inner stayer condition-mix vs within-condition splits.
    w_p_sum = math.fsum(w for w, _ in prev_tot.values())
    w_c_sum = math.fsum(w for w, _ in cur_tot.values())
    mu_prev = np.sum([(w / w_p_sum) * v for w, v in prev_tot.values()], axis=0)
    mu_cur = np.sum([(w / w_c_sum) * v for w, v in cur_tot.values()], axis=0)
    total_change = mu_cur - mu_prev

    stayers = sorted(set(prev_tot) & set(cur_tot))
    if not stayers:
        return {
            "schema": VERSION,
            "status": "no_stayer_reference",
            "total_change": total_change.tolist(),
            "n_prev_authors": len(prev_tot),
            "n_cur_authors": len(cur_tot),
            "n_stayers": 0,
        }

    # Verify consistency with open_cohort.decompose on axis 0
    _chk = _decompose_cohort_1d(
        {u: (w, float(v[0])) for u, (w, v) in prev_tot.items()},
        {u: (w, float(v[0])) for u, (w, v) in cur_tot.items()},
    )

    ws_p_tot = math.fsum(prev_tot[u][0] for u in stayers)
    ws_c_tot = math.fsum(cur_tot[u][0] for u in stayers)
    w_sp = {u: prev_tot[u][0] / ws_p_tot for u in stayers}
    w_sc = {u: cur_tot[u][0] / ws_c_tot for u in stayers}

    mu_sp = np.sum([w_sp[u] * prev_tot[u][1] for u in stayers], axis=0)
    mu_sc = np.sum([w_sc[u] * cur_tot[u][1] for u in stayers], axis=0)

    within_person_within_cond = np.zeros(dim, dtype=np.float64)
    within_person_cond_mix = np.zeros(dim, dtype=np.float64)
    n_sign_reversed = 0

    for u in stayers:
        w_bar_u = 0.5 * (w_sp[u] + w_sc[u])
        c_p = prev_by_au[u]
        c_c = cur_by_au[u]
        wp_u = prev_tot[u][0]
        wc_u = cur_tot[u][0]
        q_p = {co: cnt / wp_u for co, (cnt, _) in c_p.items()}
        q_c = {co: cnt / wc_u for co, (cnt, _) in c_c.items()}
        shared_c = sorted(set(c_p) & set(c_c))

        if shared_c:
            # Exact symmetric Kitagawa split inside author u's condition space:
            # delta_u = within_shared_cond + shared_reweight + new_cond_entry + old_cond_exit
            # Group all condition-portfolio terms into condition_mix_u so that
            # within_cond_u + condition_mix_u == z_{u,1} - z_{u,0} identically.
            sp_u = math.fsum(q_p[c] for c in shared_c)
            sc_u = math.fsum(q_c[c] for c in shared_c)
            qs_p = {c: q_p[c] / sp_u for c in shared_c}
            qs_c = {c: q_c[c] / sc_u for c in shared_c}
            within_u = np.sum(
                [0.5 * (qs_p[c] + qs_c[c]) * (c_c[c][1] - c_p[c][1]) for c in shared_c],
                axis=0,
            )
            cond_mix_u = (cur_tot[u][1] - prev_tot[u][1]) - within_u
        else:
            within_u = np.zeros(dim, dtype=np.float64)
            cond_mix_u = cur_tot[u][1] - prev_tot[u][1]

        within_person_within_cond += w_bar_u * within_u
        within_person_cond_mix += w_bar_u * cond_mix_u

        raw_u_delta0 = float(cur_tot[u][1][0] - prev_tot[u][1][0])
        within_u0 = float(within_u[0])
        if raw_u_delta0 * within_u0 < -1e-20:
            n_sign_reversed += 1

    stayer_reweight = np.sum(
        [(w_sc[u] - w_sp[u]) * 0.5 * (prev_tot[u][1] + cur_tot[u][1]) for u in stayers],
        axis=0,
    )
    entrants = [u for u in cur_tot if u not in prev_tot]
    leavers = [u for u in prev_tot if u not in cur_tot]
    entrant_takeover = (
        np.sum([(cur_tot[u][0] / w_c_sum) * (cur_tot[u][1] - mu_sc) for u in entrants], axis=0)
        if entrants
        else np.zeros(dim, dtype=np.float64)
    )
    leaver_selection = (
        np.sum([(prev_tot[u][0] / w_p_sum) * (mu_sp - prev_tot[u][1]) for u in leavers], axis=0)
        if leavers
        else np.zeros(dim, dtype=np.float64)
    )
    recon = (
        within_person_within_cond
        + within_person_cond_mix
        + stayer_reweight
        + entrant_takeover
        + leaver_selection
    )
    closure_err = float(np.max(np.abs(total_change - recon)))
    assert abs((within_person_within_cond[0] + within_person_cond_mix[0]) - _chk["within"]) < 1e-11
    return {
        "schema": VERSION,
        "status": "decomposed",
        "n_prev_authors": len(prev_tot),
        "n_cur_authors": len(cur_tot),
        "n_stayers": len(stayers),
        "n_entrants": len(entrants),
        "n_leavers": len(leavers),
        "n_stayers_sign_reversed_by_mix": int(n_sign_reversed),
        "total_change": total_change.tolist(),
        "within_person_within_condition": within_person_within_cond.tolist(),
        "within_person_condition_mix": within_person_cond_mix.tolist(),
        "stayer_weight_reallocation": stayer_reweight.tolist(),
        "entrant_takeover": entrant_takeover.tolist(),
        "leaver_selection": leaver_selection.tolist(),
        "closure_max_abs_err": closure_err,
    }


def _cross_fit_blind_linear(
    embeddings: np.ndarray,
    targets: np.ndarray,
    author_ids: Sequence[Any],
    *,
    alphas: Sequence[float],
    n_folds: int,
    seed: int,
) -> np.ndarray:
    unique_au = sorted({str(a) for a in author_ids})
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(len(unique_au))
    k_folds = max(2, min(int(n_folds), len(unique_au)))
    au_to_fold = {unique_au[idx]: int(pos % k_folds) for pos, idx in enumerate(perm)}
    folds = np.array([au_to_fold[str(a)] for a in author_ids], dtype=int)
    ids = np.asarray(author_ids)
    oof = np.empty_like(targets, dtype=np.float64)
    for outer in sorted(set(folds.tolist())):
        tr = folds != outer
        train_x, train_y = embeddings[tr], targets[tr]
        inner_au = sorted({str(a) for a in ids[tr]})
        in_perm = np.random.default_rng(int(seed) + outer + 1).permutation(len(inner_au))
        in_k = max(2, min(int(n_folds), len(inner_au)))
        in_map = {inner_au[idx]: int(pos % in_k) for pos, idx in enumerate(in_perm)}
        inner = np.array([in_map[str(a)] for a in ids[tr]], dtype=int)

        def _inner_loss(
            alpha: float,
            _inner: np.ndarray = inner,
            _tx: np.ndarray = train_x,
            _ty: np.ndarray = train_y,
        ) -> float:
            tot = 0.0
            for kf in np.unique(_inner):
                keep = _inner != kf
                w, b = CrossFittedRegisterCalibrator._fit_ridge(
                    _tx[keep], _ty[keep], alpha
                )
                tot += float(np.sum((_ty[~keep] - _tx[~keep] @ w - b) ** 2))
            return tot

        al = min(alphas, key=_inner_loss) if len(np.unique(inner)) > 1 else alphas[0]
        w, b = CrossFittedRegisterCalibrator._fit_ridge(train_x, train_y, al)
        oof[~tr] = embeddings[~tr] @ w + b
    return oof


def _dif_metrics(preds: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict[str, Any]:
    uniq = sorted({str(g) for g in groups})
    bias_by_g: dict[str, float] = {}
    slope_by_g: dict[str, float] = {}
    res_var_by_g: dict[str, float] = {}
    corr_by_g: dict[str, float] = {}
    mse_by_g: dict[str, float] = {}
    for g in uniq:
        sel = groups == g
        pg, yg = preds[sel], y[sel]
        res = pg - yg
        bias_by_g[g] = float(np.mean(res))
        mse_by_g[g] = float(np.mean(res ** 2))
        res_var_by_g[g] = float(np.var(res, ddof=1)) if len(res) > 1 else 0.0
        var_p = float(np.var(pg, ddof=1)) if len(pg) > 1 else 0.0
        if var_p > 1e-12:
            cov_py = float(np.cov(pg, yg, ddof=1)[0, 1])
            slope_by_g[g] = cov_py / var_p
        else:
            slope_by_g[g] = 0.0
        pc, yc = pg - np.mean(pg), yg - np.mean(yg)
        den = float(np.linalg.norm(pc) * np.linalg.norm(yc))
        corr_by_g[g] = float(np.dot(pc, yc) / den) if den > 1e-12 else 0.0

    biases = list(bias_by_g.values())
    slopes = list(slope_by_g.values())
    rvars = [v for v in res_var_by_g.values() if v > 1e-15]
    sd_y = float(np.std(y, ddof=1)) if len(y) > 1 else 1.0
    uniform_gap = float(max(biases) - min(biases)) if biases else 0.0
    nonunif_slope_gap = float(max(slopes) - min(slopes)) if slopes else 0.0
    strict_var_ratio = float(max(rvars) / min(rvars)) if len(rvars) >= 2 else 1.0
    min_corr = min(corr_by_g.values()) if corr_by_g else 0.0

    if min_corr <= 0.0:
        license_tier = "L0_unlicensed"
    elif nonunif_slope_gap <= 0.15 and uniform_gap <= 0.05 * max(sd_y, 1e-12):
        license_tier = "L3_scalar_level_comparable"
    elif nonunif_slope_gap <= 0.15:
        license_tier = "L2_metric_contrast_comparable"
    else:
        license_tier = "L1_within_register_rank_only"

    return {
        "overall_mse": float(np.mean((preds - y) ** 2)),
        "uniform_dif_gap": uniform_gap,
        "nonuniform_slope_gap": nonunif_slope_gap,
        "strict_residual_var_ratio": strict_var_ratio,
        "mean_bias_by_register": bias_by_g,
        "slope_by_register": slope_by_g,
        "residual_var_by_register": res_var_by_g,
        "within_register_corr": corr_by_g,
        "mse_by_register": mse_by_g,
        "earned_comparison_license": license_tier,
    }


def audit_register_dif_invariance(
    embeddings: np.ndarray,
    surface_probs: np.ndarray,
    register_groups: Sequence[Any],
    targets: Sequence[float] | np.ndarray,
    author_ids: Sequence[Any],
    *,
    trait_b_predictions: Sequence[float] | np.ndarray | None = None,
    trait_b_blind_predictions: Sequence[float] | np.ndarray | None = None,
    alphas: Sequence[float] = (0.1, 1.0, 10.0, 100.0),
    n_folds: int = 5,
    seed: int = 0,
) -> dict[str, Any]:
    """Audit Meredith (1993) DIF hierarchy & G12 license before vs after Kronecker probe."""
    emb = np.asarray(embeddings, dtype=np.float64)
    sp = np.asarray(surface_probs, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    grp = np.asarray([str(g) for g in register_groups])
    if (
        emb.ndim != 2
        or sp.ndim != 2
        or y.ndim != 1
        or len(emb) != len(sp)
        or len(emb) != len(y)
        or len(emb) != len(grp)
        or len(emb) != len(author_ids)
        or len(set(grp.tolist())) < 2
    ):
        raise ValueError("aligned finite embeddings, surface_probs, >=2 register groups, targets, and author_ids required")

    blind_oof = _cross_fit_blind_linear(
        emb, y, author_ids, alphas=alphas, n_folds=n_folds, seed=seed
    )
    cal = CrossFittedRegisterCalibrator(alphas=alphas, n_folds=n_folds, seed=seed)
    kron_fit = cal.fit_predict(emb, sp, y, author_ids)
    kron_oof = np.asarray(kron_fit["oof_predictions"], dtype=np.float64)

    blind_metrics = _dif_metrics(blind_oof, y, grp)
    kron_metrics = _dif_metrics(kron_oof, y, grp)

    mtmm_out = None
    regs = sorted(set(grp.tolist()))
    if len(regs) == 2 and trait_b_predictions is not None and trait_b_blind_predictions is not None:
        r0, r1 = grp == regs[0], grp == regs[1]
        n_pair = min(int(np.sum(r0)), int(np.sum(r1)))
        if n_pair >= 3:
            tb_k = np.asarray(trait_b_predictions, dtype=np.float64)
            tb_b = np.asarray(trait_b_blind_predictions, dtype=np.float64)
            mtmm_out = {
                "blind": mtmm_validity_audit(
                    blind_oof[r0][:n_pair],
                    blind_oof[r1][:n_pair],
                    tb_b[r0][:n_pair],
                    tb_b[r1][:n_pair],
                ),
                "kronecker": mtmm_validity_audit(
                    kron_oof[r0][:n_pair],
                    kron_oof[r1][:n_pair],
                    tb_k[r0][:n_pair],
                    tb_k[r1][:n_pair],
                ),
            }

    return {
        "schema": VERSION,
        "register_groups": regs,
        "n_events": len(y),
        "n_authors": len({str(a) for a in author_ids}),
        "blind": blind_metrics,
        "kronecker": kron_metrics,
        "best_alpha": float(kron_fit["best_alpha"]),
        "uniform_dif_reduction_ratio": (
            float(blind_metrics["uniform_dif_gap"] / max(kron_metrics["uniform_dif_gap"], 1e-12))
            if blind_metrics["uniform_dif_gap"] > 0.0
            else 1.0
        ),
        "nonuniform_slope_reduction_ratio": (
            float(blind_metrics["nonuniform_slope_gap"] / max(kron_metrics["nonuniform_slope_gap"], 1e-12))
            if blind_metrics["nonuniform_slope_gap"] > 0.0
            else 1.0
        ),
        "mtmm": mtmm_out,
    }


def _dispatch_task(task: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    if task == "reader_calibration":
        return calibrate_reader_misclassification(
            payload["observed"],
            payload["reader_calibration"],
            weights=payload.get("weights"),
            contrast=payload.get("contrast"),
            confidence=float(payload.get("confidence", 0.95)),
            calibration_denominator=payload.get("calibration_denominator"),
        )
    if task == "caps_reliability":
        res = {
            "panel_a": estimate_bivariate_caps_reliability(
                payload["records"],
                event_var_c1=float(payload["event_var_c1"]),
                event_var_c0=float(payload["event_var_c0"]),
                weights=payload.get("weights", (0.5, 0.5)),
                min_reliability=float(payload.get("min_reliability", 0.30)),
                confidence=float(payload.get("confidence", 0.95)),
                fit_records=payload.get("fit_records"),
            )
        }
        if "records_b" in payload:
            res["panel_b"] = estimate_bivariate_caps_reliability(
                payload["records_b"],
                event_var_c1=float(payload["event_var_c1"]),
                event_var_c0=float(payload["event_var_c0"]),
                weights=payload.get("weights", (0.5, 0.5)),
                min_reliability=float(payload.get("min_reliability", 0.30)),
                confidence=float(payload.get("confidence", 0.95)),
                fit_records=payload.get("fit_records"),
            )
            res["cross_panel"] = deattenuate_cross_panel_correlation(
                res["panel_a"],
                res["panel_b"],
                metric=str(payload.get("metric", "contrast")),
                n_perm=int(payload.get("n_perm", 300)),
                seed=int(payload.get("seed", 0)),
            )
        return res
    if task == "ct_lsto":
        return fit_ct_lsto_variance_components(
            payload["timestamps"],
            payload["author_ids"],
            payload["condition_ids"],
            payload["values"],
            max_pairs_per_key=int(payload.get("max_pairs_per_key", 64)),
            n_starts=int(payload.get("n_starts", 6)),
        )
    if task == "irtree_selection_expression":
        return measure_irtree_selection_expression(
            payload["timestamps"],
            payload["author_ids"],
            payload["condition_ids"],
            payload["values"],
            window_edges=payload["window_edges"],
            min_active_windows=int(payload.get("min_active_windows", 4)),
            eta_min=float(payload.get("eta_min", 0.5)),
            n_perm=int(payload.get("n_perm", 300)),
            seed=int(payload.get("seed", 0)),
        )
    if task == "hierarchical_mix":
        prev_cells = {(row["focal"], row["condition"]): (row["count"], row["value"]) for row in payload["prev_cells"]}
        cur_cells = {(row["focal"], row["condition"]): (row["count"], row["value"]) for row in payload["cur_cells"]}
        return decompose_hierarchical_cohort_person_mix(prev_cells, cur_cells)
    if task == "register_dif":
        return audit_register_dif_invariance(
            payload["embeddings"],
            payload["surface_probs"],
            payload["register_groups"],
            payload["targets"],
            payload["author_ids"],
            n_folds=int(payload.get("n_folds", 5)),
            seed=int(payload.get("seed", 0)),
        )
    raise ValueError(f"unsupported task {task!r}")


def measure_psychometric_card(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Unified front-door psychometric measurement interface (layer M; deterministic R-SCORE)."""
    task = str(spec["task"])
    encoded = json.dumps(
        spec, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    )
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    if task == "bundle":
        sections = spec.get("sections", {})
        if not isinstance(sections, Mapping) or not sections:
            raise ValueError("bundle task requires non-empty sections mapping")
        results = {
            name: _dispatch_task(str(sec["task"]), sec)
            for name, sec in sections.items()
        }
    else:
        results = _dispatch_task(task, spec)

    return {
        "schema": VERSION,
        "task": task,
        "spec_digest": digest,
        "claim_boundary": (
            "Layer-M psychometric calibration, reliability, and invariance audit only; "
            "external construct validity (Theta) or clinical diagnosis is not implied."
        ),
        "result": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    spec = json.loads(args.input.read_text(encoding="utf-8"))
    out = json.dumps(
        measure_psychometric_card(spec),
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )
    if args.out is not None:
        with args.out.open("x", encoding="utf-8") as handle:
            print(out, file=handle)
    else:
        print(out)


__all__ = [
    "VERSION",
    "audit_register_dif_invariance",
    "calibrate_reader_misclassification",
    "deattenuate_cross_panel_correlation",
    "decompose_hierarchical_cohort_person_mix",
    "estimate_bivariate_caps_reliability",
    "fit_ct_lsto_variance_components",
    "main",
    "measure_irtree_selection_expression",
    "measure_psychometric_card",
]


if __name__ == "__main__":
    main()
