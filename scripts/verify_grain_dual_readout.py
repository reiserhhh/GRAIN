#!/usr/bin/env python3
"""Same-agent alternative verification for GRAIN-C1-02 (r4)."""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import dual_readout as DR
from grain_research.construction import hidden_evidence_map as H
from grain_research.gptpro_foundation_v3.f04_recovery import (
    fit_source_moments,
    score_response,
)

out = ROOT / "results/grain_construction_v1/C1/hidden_map_r4"
c = json.loads((out / "source_application_contract.json").read_text())
chk = {"verification_kind": "same-agent alternative computation (not independent third-party)"}

# 1 digests (base view + augmented pilot_z), primary and boundary
ok = True
for key, d in c["view_digests"].items():
    w, s = key.split("/")
    v, a, _ = DR.build_world_augmented(int(s), w)
    ok &= H.view_digest(v) == d["base_view"] and hashlib.sha256(np.ascontiguousarray(a.pilot_z).tobytes()).hexdigest() == d["augmented_pilot_z_sha256"]
for key, d in c["boundary_view_digests"].items():
    v, a, _ = DR.build_world_correlated(int(key.split("/")[1]))
    ok &= H.view_digest(v) == d["base_view"]
chk["view_digests_match"] = bool(ok)

# 2 isolation and declared pilot_z dependency
seed = c["seeds"][0]
v, a, t = DR.build_world_augmented(seed, "W2_conditional")
w1, w2 = DR.fit_dual("lcmv_dual", v).direction.copy(), DR.fit_dual("pilot_dual", a).direction.copy()
for k in ("a", "b", "f", "s"):
    t.app_latent[k][:] = 0
    t.source_latent[k][:] = 0
t.pilot_latent["z"][:] = 0
chk["outputs_unchanged_under_truth_mutation"] = bool(np.array_equal(DR.fit_dual("lcmv_dual", v).direction, w1)
                                                     and np.array_equal(DR.fit_dual("pilot_dual", a).direction, w2))
perm = DR.AugmentedEstimatorView(**{f: getattr(a, f) for f in H.ESTIMATOR_FIELDS},
                                 pilot_z=np.random.default_rng(5).permutation(a.pilot_z))
chk["pilot_dual_depends_on_pilot_z"] = bool(not np.allclose(DR.fit_dual("pilot_dual", perm).direction, w2))

# 3 alternative LCMV route: constrained numerical optimisation in range coordinates
v, a, t = DR.build_world_augmented(seed, "W2_conditional")
between, _ = H._slope_moments(v)
rr = DR.rank_rule(between)
vh = DR.forward_loading(v.pilot_x[v.pilot_writer < H.PILOT_TRAIN_WRITERS], v.pilot_label[v.pilot_writer < H.PILOT_TRAIN_WRITERS])
av = rr["U"].T @ vh
res = minimize(lambda y: y @ (rr["L"] * y), av / (av @ av), jac=lambda y: 2 * rr["L"] * y,
               constraints=[{"type": "eq", "fun": lambda y: y @ av - 1, "jac": lambda y: av}],
               method="SLSQP", options={"ftol": 1e-14, "maxiter": 500})
w_opt = rr["U"] @ res.x
chk["lcmv_closed_form_vs_slsqp_max_abs_diff"] = float(np.max(np.abs(w_opt - DR.lcmv_single(rr["U"], rr["L"], vh))))

# 4 kernel vs F04 originals on lcmv_dual evidence
r = DR.fit_dual("lcmv_dual", v)
phi = r.phi(v.source_x)
rows = [{"author_id": f"s{au}", "role": "source", "z": float(z), "y": float(y), "t": int(i % H.EVENTS),
         "event_id": f"s{au}/{i % H.EVENTS}"} for i, (au, z, y) in enumerate(zip(v.source_author, v.source_z, phi))]
ref = fit_source_moments(rows)
mine = H.fit_moments(phi, v.source_z, v.source_author)
chk["moments_vs_f04_max_abs_diff"] = float(max(np.max(np.abs(np.array(ref["Sigma"]) - mine["Sigma"])),
                                               np.max(np.abs(np.array(ref["mu"]) - mine["mu"]))))
ap = r.phi(v.app_x).reshape(H.N_APP, -1)
zz = v.app_z[v.app_author == 0]
beta, _, _ = H.blup(mine, H.design(zz), ap)
fit = {"mode": "linear", "mu": ref["mu"], "Sigma": ref["Sigma"], "sigma2": ref["sigma2"]}
chk["blup_vs_f04_max_abs_diff"] = float(max(np.max(np.abs(score_response(
    [{"z": float(zz[k]), "y": float(ap[u, k]), "event_id": f"a{u}/{k}"} for k in range(H.EVENTS)], fit)["coefficients"] - beta[u]))
    for u in range(5)))

# 5 recovery recomputed from cold scores
cold, rec = pd.read_csv(out / "cold_scores.csv"), pd.read_csv(out / "recovery_by_mechanism.csv")
worst = 0.0
for s in c["seeds"]:
    for w in ("W1_shared_level", "W2_conditional"):
        _, _, tr = DR.build_world_augmented(s, w)
        for arm in c["arms"]:
            for tg in H.WORLDS[w]["targets"]:
                cc = cold[(cold.world == w) & (cold.seed == s) & (cold.arm == arm) & (cold.target == tg)]
                if cc.empty:
                    continue
                tv = {"level": tr.app_latent["a"], "delta": 2 * tr.app_latent["b"]}[tg]
                row = rec[(rec.world == w) & (rec.seed == s) & (rec.arm == arm) & (rec.target == tg) & (rec.material == "same_family")].iloc[0]
                cov = np.mean(np.abs(cc.score.to_numpy() - tv) <= 1.959964 * np.sqrt(cc.interval_var.to_numpy()))
                worst = max(worst, abs(np.corrcoef(cc.score, tv)[0, 1] - row.r), abs(cov - row.coverage))
chk["recovery_recompute_max_abs_diff"] = float(worst)

# 6 analytic vs simulated intervention effects
iv = pd.read_csv(out / "intervention_checks.csv")
g = iv[iv.statistic == "gain_T"]
f = iv[iv.intervention.isin(["fingerprint", "style"])]
chk["gain_vs_analytic_max_abs_diff"] = float(np.max(np.abs(g.value - g.analytic)))
chk["fingerprint_style_vs_analytic_max_abs_diff"] = float(np.max(np.abs(f.rms_change_ols - f.analytic_rms_change_ols)))

# 7 boundary vs algebra: population lcmv style slope -rho*sb/sr = -.3 (finite-sample estimate reported)
b = pd.read_csv(out / "correlated_boundary/recovery_by_mechanism.csv")
bl = b[(b.arm == "lcmv_dual") & (b.target == "delta") & (b.material == "same_family")]
chk["boundary_lcmv_style_slope_mean"] = float(bl.mech_style_response_per_sd.mean())
chk["boundary_algebra_prediction"] = -DR.RHO_BOUNDARY * 0.5 / 1.0
chk["pass"] = bool(chk["view_digests_match"] and chk["outputs_unchanged_under_truth_mutation"]
                   and chk["pilot_dual_depends_on_pilot_z"] and chk["lcmv_closed_form_vs_slsqp_max_abs_diff"] < 1e-6
                   and chk["moments_vs_f04_max_abs_diff"] < 1e-9 and chk["blup_vs_f04_max_abs_diff"] < 1e-9
                   and chk["recovery_recompute_max_abs_diff"] < 1e-9 and chk["gain_vs_analytic_max_abs_diff"] < 1e-8
                   and chk["fingerprint_style_vs_analytic_max_abs_diff"] < 1e-8)
(out / "verification.json").write_text(json.dumps(chk, indent=1))
print(json.dumps(chk, indent=1))
