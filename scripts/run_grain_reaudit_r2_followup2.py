#!/usr/bin/env python3
"""Re-audit r2 follow-up packet 2 (FU2-a..e); declared simulations only."""

from __future__ import annotations

import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import chi2

import scripts.run_grain_r11_measurement_modes_joint as J
from scripts.run_grain_r29_joint_decomposition import anova

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/GRAIN_REAUDIT_R2_FOLLOWUP2_PROTOCOL_V1.md"
PROTOCOL_SHA = "7538b48520b70190ec6698c4e50fd604a503b0c4b61ffe05781a466397a41d21"
V3 = ROOT / "results/grain_gptpro_foundation_v3"
OUT = V3 / "F17/reaudit_r2_followup2"
SEED = 20260926


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fu2_a():
    F = ("P", "C", "T", "M", "F", "S")
    L = {"P": 120, "C": 10, "T": 6, "M": 5, "F": 3, "S": 2}
    true = {("P",): 1.0, ("C",): 0.5, ("P", "C"): 0.3, ("T",): 0.3, ("P", "T"): 0.1, ("M",): 0.2, ("P", "M"): 0.1,
            ("F",): 0.05, ("P", "F"): 0.05, ("P", "S"): 0.4, ("C", "M"): 0.05}
    rng = np.random.default_rng(SEED)
    shape = tuple(L[f] for f in F)
    y = np.zeros((*shape, 2))
    for eff, var in true.items():
        y += np.sqrt(var) * rng.standard_normal([L[f] if f in eff else 1 for f in F])[..., None]
    y += rng.standard_normal(y.shape)
    est = anova(y, F)
    boots = [anova(y[rng.integers(0, L["P"], L["P"])], F) for _ in range(200)]
    rows = []
    for eff, var in {**true, ("resid",): 1.0}.items():
        e = est[eff]
        if "P" in eff or eff == ("resid",):
            lo, hi = np.percentile([b[eff] for b in boots], [2.5, 97.5])
            method = "person_bootstrap"
        else:
            df = int(np.prod([L[f] - 1 for f in eff]))
            lo, hi = (max(e, 1e-12) * df / chi2.ppf(0.975, df), max(e, 1e-12) * df / chi2.ppf(0.025, df))
            method = f"chi2_df{df}"
        rows.append({"effect": "x".join(eff), "true": var, "estimate": float(e), "ci": [float(lo), float(hi)],
                     "covers": bool(lo <= var <= hi), "method": method})
    ncell = L["T"] * L["M"] * L["F"] * L["S"] * 2
    icc_b = est[("P", "C")] / (est[("P", "C")] + est[("resid",)] / ncell)
    return {"components": rows, "covered": f"{sum(r['covers'] for r in rows)}/{len(rows)}",
            "ICC_B_person_condition_profile": float(icc_b),
            "cross_paraphrase_share": float(est[("P", "F")] / (est[("P", "F")] + est[("P",)]))}


def fu2_b():
    sets = {"A": (0.3, 0.7, 0.8, 1.5), "B": (0.3, 0.7, 0.8, 0.5), "C": (0.3, 0.7, 1.5, 1.5)}
    out = {}
    for name, (e0, e1, s1, p1) in sets.items():
        for cons, sel_nuis in (("selection_sensitive", False), ("selection_free", True)):
            row = {}
            for o in (0, 0.25, 0.5, 0.75, 1.0):
                e, s, p = e0 + e1 * o, s1 * o, p1 * o * o
                target = e * e + (0 if sel_nuis else s * s)
                noise = (s * s if sel_nuis else 0) + p * p / 8 + 0.1 * o / 3 + 1.0 / 8
                row[f"{o:.2f}"] = {"G": round(target / (target + noise), 4), "reader_var_component": round(0.1 * o / 3, 4)}
            out[f"{name}|{cons}"] = row
    return out


def fu2_c():
    m = pd.read_csv(V3 / "F09/rival_person_simulation_r1/rival_person_metrics.csv")
    d = m[(m.design == "D1") & (m.world == "W1")]
    return {r.coordinate: {"population_mean_effect_lpm": round(r.mean_estimate, 4), "person_retest_reliability": round(r.retest_r, 4),
                           "true_mean_logit": round(r.true_mean, 4)} for r in d.itertuples()}


def fu2_d():
    rng = np.random.default_rng(SEED + 4)
    cells = np.array(list(itertools.product([0, 1], repeat=5)), float)
    names = ("M1", "M2", "M3", "M4", "M5")

    def regs(x):
        B, e, r, k, g = x.T
        return {"M1": 1 - B, "M2": e, "M3": r, "M4": B * k, "M5": g}

    res = {}
    for design in ("D1", "D4"):
        conf = np.zeros((5, 5), int)
        for truth in range(5):
            for _ in range(80):
                x = np.repeat(cells, 3, axis=0)
                if design == "D4":
                    tie = rng.random(len(x)) < 0.9
                    x[tie, 4] = x[tie, 1]
                rg = regs(x)
                eta = rng.normal(-1, 0.5) + 1.5 * rg[names[truth]]
                y = (rng.random(len(x)) < expit(eta)).astype(float)
                cell_id = np.repeat(np.arange(32), 3)
                avail = names if design == "D1" else names[:4]
                mse = {}
                for mname in avail:
                    X = np.column_stack([np.ones(len(x)), rg[mname]])
                    err = 0.0
                    for c in range(32):
                        te = cell_id == c
                        w = np.linalg.lstsq(X[~te], y[~te], rcond=None)[0]
                        err += ((y[te] - X[te] @ w) ** 2).sum()
                    mse[mname] = err
                conf[truth, names.index(min(mse, key=mse.get))] += 1
        res[design] = {"confusion_rows_truth_cols_chosen": conf.tolist(), "labels": list(names),
                       "accuracy": float(np.trace(conf) / conf.sum()),
                       "M5_convention_persons_chosen_as": dict(zip(names, conf[4].tolist(), strict=True))}
    return res


def fu2_e():
    res = {}
    rng0 = np.random.default_rng(SEED + 9)
    n = 300
    theta = rng0.standard_normal(n)
    h = 0.7 * theta + 0.71 * rng0.standard_normal(n)
    beta = rng0.normal(1, 0.5, n)
    for lam in (0.0, 0.001, 0.002, 0.005, 0.01, 0.02):
        J.LAM = lam
        for mode in ("TRIG", "ADAPT"):
            mrng = np.random.default_rng(np.random.SeedSequence([SEED, int(lam * 1000), len(mode)]))
            out = [J.run(mode, theta[u], h[u], beta[u], mrng) for u in range(n)]
            est = np.array([o[0] for o in out])
            pr = np.array([o[1] for o in out])
            dist = np.array([o[2] for o in out])
            err = est - theta
            res[f"lambda={lam}|{mode}"] = {"probes_per_person": float(pr.mean()), "rmse": float(np.sqrt(np.mean(err ** 2))),
                                            "utility": float(np.mean(-(err ** 2) - lam * pr - 0.10 * dist))}
    J.LAM = 0.02
    return res


def main():
    if digest(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("Protocol changed")
    OUT.mkdir(parents=True, exist_ok=True)
    out = {"protocol_sha256": PROTOCOL_SHA}
    for name, fn in (("FU2_a_joint_paraphrase_facets", fu2_a), ("FU2_b_R13_per_level_G", fu2_b),
                     ("FU2_c_R12_person_dual_report", fu2_c), ("FU2_d_R18_rival_model_comparison", fu2_d),
                     ("FU2_e_R11_cost_sweep", fu2_e)):
        out[name] = fn()
        print(name, json.dumps(out[name])[:900], flush=True)
    out.update({"code_sha256": digest(Path(__file__)), "neural_calls": 0, "created": time.strftime("%Y-%m-%d %H:%M:%S %Z")})
    (OUT / "followup2.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
