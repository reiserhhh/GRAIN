#!/usr/bin/env python3
"""Alternate numerical checks for GRAIN-C1-01 outputs (independent recomputation)."""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import hidden_evidence_map as H
from grain_research.gptpro_foundation_v3.f04_recovery import (
    fit_source_lmm,
    fit_source_moments,
    score_response,
)

out = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "results/grain_construction_v1/C1/hidden_map_r1")
contract = json.loads((out / "source_application_contract.json").read_text())
if contract.get("pilot_writers"):
    H.PILOT_WRITERS, H.PILOT_TRAIN_WRITERS = contract["pilot_writers"], contract["pilot_train_writers"]
seed, world = contract["seeds"][0], "W2_conditional"
view, truth = H.build_world(seed, world)
checks = {}

# 1. isolation: view digests equal the contract, and estimator output ignores truth
dig = {f"{w}/{s}": H.view_digest(H.build_world(s, w)[0]) for s in contract["seeds"] for w in H.WORLDS}
checks["view_digest_match"] = all(dig[k] == v for k, v in contract["estimator_view_sha256"].items())
r = H.fit_arm("pilot_pcr", view)
for key in ("a", "b", "f", "s", "eps"):
    truth.app_latent[key][:] = 0
r2 = H.fit_arm("pilot_pcr", view)
checks["estimator_ignores_truth_mutation"] = bool(np.array_equal(r.direction, r2.direction) and r.unit == r2.unit)
view, truth = H.build_world(seed, world)

# 2. vectorized kernel vs F04 originals on actual evidence
phi = r.phi(view.source_x)
rows = [{"author_id": f"s{a}", "role": "source", "z": float(z), "y": float(y), "t": int(i % H.EVENTS),
         "event_id": f"s{a}/{i % H.EVENTS}"} for i, (a, z, y) in enumerate(zip(view.source_author, view.source_z, phi))]
ref = fit_source_moments(rows)
mine = H.fit_moments(phi, view.source_z, view.source_author)
checks["moments_max_abs_diff"] = float(max(np.max(np.abs(np.array(ref["mu"]) - mine["mu"])),
                                           np.max(np.abs(np.array(ref["Sigma"]) - mine["Sigma"])),
                                           abs(ref["sigma2"] - mine["sigma2"])))
app_phi = r.phi(view.app_x).reshape(H.N_APP, -1)
zz = view.app_z[view.app_author == 0]
beta, _, _ = H.blup(mine, H.design(zz), app_phi)
fit = {"mode": "linear", "mu": ref["mu"], "Sigma": ref["Sigma"], "sigma2": ref["sigma2"]}
diffs = []
for u in range(5):
    pre = [{"z": float(zz[t]), "y": float(app_phi[u, t]), "event_id": f"a{u}/{t}"} for t in range(H.EVENTS)]
    diffs.append(np.max(np.abs(score_response(pre, fit)["coefficients"] - beta[u])))
checks["blup_vs_score_response_max_abs_diff"] = float(max(diffs))

# 3. REML kernel vs moments kernel on the same evidence
reml = fit_source_lmm(rows)
breml = np.array([score_response([{"z": float(zz[t]), "y": float(app_phi[u, t]), "event_id": f"a{u}/{t}"}
                                  for t in range(H.EVENTS)], reml)["coefficients"] for u in range(H.N_APP)])
truth_delta, truth_level = 2 * truth.app_latent["b"], truth.app_latent["a"]
checks["reml"] = {"method": reml["method"], "fallback": reml.get("fallback"),
                  "corr_delta_reml_vs_moments": float(np.corrcoef(2 * breml[:, 1], 2 * beta[:, 1])[0, 1]),
                  "corr_level_reml_vs_moments": float(np.corrcoef(breml[:, 0], beta[:, 0])[0, 1]),
                  "r_delta_truth_reml": float(np.corrcoef(2 * breml[:, 1], truth_delta)[0, 1]),
                  "r_delta_truth_moments": float(np.corrcoef(2 * beta[:, 1], truth_delta)[0, 1]),
                  "r_level_truth_reml": float(np.corrcoef(breml[:, 0], truth_level)[0, 1]),
                  "r_level_truth_moments": float(np.corrcoef(beta[:, 0], truth_level)[0, 1])}

# 4. recovery csv vs independent recomputation from cold_scores.csv
cold = pd.read_csv(out / "cold_scores.csv")
rec = pd.read_csv(out / "recovery_by_mechanism.csv")
worst = 0.0
for s in contract["seeds"]:
    for w in ("W1_shared_level", "W2_conditional"):
        _, tr = H.build_world(s, w)
        for arm in ("pilot_ridge", "pilot_pcr", "keyed"):
            for tgt in H.WORLDS[w]["targets"]:
                c = cold[(cold.world == w) & (cold.seed == s) & (cold.arm == arm) & (cold.target == tgt)]
                tv = {"level": tr.app_latent["a"], "delta": 2 * tr.app_latent["b"]}[tgt]
                rr = np.corrcoef(c.score.to_numpy(), tv)[0, 1]
                cov = np.mean(np.abs(c.score.to_numpy() - tv) <= 1.959964 * np.sqrt(c.interval_var.to_numpy()))
                row = rec[(rec.world == w) & (rec.seed == s) & (rec.arm == arm) & (rec.target == tgt)
                          & (rec.material == "same_family")].iloc[0]
                worst = max(worst, abs(rr - row.r), abs(cov - row.coverage))
checks["recovery_recompute_max_abs_diff"] = float(worst)

# 5. analytic vs simulated intervention effects
iv = pd.read_csv(out / "intervention_checks.csv")
g = iv[(iv.statistic == "gain_T") & iv.analytic.notna()]
checks["gain_T_vs_analytic_max_abs_diff"] = float(np.max(np.abs(g.value - g.analytic)))
f = iv[iv.intervention.isin(["fingerprint", "style"]) & iv.analytic_rms_change_ols.notna()]
checks["fingerprint_style_rms_vs_analytic_max_abs_diff"] = float(np.max(np.abs(f.rms_change_ols - f.analytic_rms_change_ols)))
checks["pass"] = bool(checks["view_digest_match"] and checks["estimator_ignores_truth_mutation"]
                      and checks["moments_max_abs_diff"] < 1e-9 and checks["blup_vs_score_response_max_abs_diff"] < 1e-9
                      and checks["recovery_recompute_max_abs_diff"] < 1e-9
                      and checks["gain_T_vs_analytic_max_abs_diff"] < 1e-8
                      and checks["fingerprint_style_rms_vs_analytic_max_abs_diff"] < 1e-8)
(out / "verification.json").write_text(json.dumps(checks, indent=1))
print(json.dumps(checks, indent=1))
