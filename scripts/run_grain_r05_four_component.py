#!/usr/bin/env python
"""R05 paired four-component identification (docs/GRAIN_R05_PAIRED_FOUR_COMPONENT_PROTOCOL_V1.md).
CPU only, 0 neural calls. Estimators consume observed source arrays only; truth is evaluation-only."""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research import r05_four_component as m  # noqa: E402

PROTOCOL = ROOT / "docs/GRAIN_R05_PAIRED_FOUR_COMPONENT_PROTOCOL_V1.md"
PROTOCOL_SHA = "46139a8b2062d574923c15e48263024da43b5729d89ee7ba67f9f4a0e62cf221"
OUT = ROOT / "results/grain_gptpro_foundation_v3/F03/paired_four_component_identification_r1"
SEEDS = (1, 2, 3)
N_SRC, N_APP, REPS, SD_ERR, N_BOOT = 300, 150, 3, 0.8, 2000
MU = np.array([0.2, 0.0, -0.1, 0.1, 0.05])
SIG1 = np.diag([0.36, 0.25, 0.49, 0.16])
SIG2 = SIG1.copy(); SIG2[0, 1] = SIG2[1, 0] = -0.18; SIG2[0, 2] = SIG2[2, 0] = -0.168
L3 = m.L_NOMINAL.copy(); L3[3] = [0, 1, 1, 0.3]
L4 = m.L_NOMINAL.copy(); L4[2] = [0, 1.75, 1, 1]
WORLDS = {"W1_independent": (SIG1, m.L_NOMINAL), "W2_correlated": (SIG2, m.L_NOMINAL), "W3_leaky_clamp": (SIG1, L3), "W4_C_restriction_violated": (SIG1, L4)}
IU = np.triu_indices(4)
SIG_NAMES = [f"{m.COMPONENTS[i]}|{m.COMPONENTS[j]}" for i, j in zip(*IU)]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify():
    if sha(PROTOCOL) != PROTOCOL_SHA:
        raise RuntimeError("protocol changed")
    for rel, h in re.findall(r"\| `([^`]+)` \| `([0-9a-f]{64})` \|", PROTOCOL.read_text()):
        if sha(ROOT / rel) != h:
            raise RuntimeError(f"binding mismatch {rel}")


def analytic_witnesses():
    out = {"designs": {}}
    sub = {"ABC": [0, 1, 2], "ABCD": [0, 1, 2, 3], "ABCDE": [0, 1, 2, 3, 4], "ABDE": list(m.BASIS)}
    for k, rows in sub.items():
        out["designs"][k] = m.design_summary(m.L_NOMINAL[rows])
    v = m.NULL_WITNESS
    for k in ("ABC", "ABCD", "ABCDE"):
        L = m.L_NOMINAL[sub[k]]
        out["designs"][k]["null_witness_v"] = v.tolist()
        out["designs"][k]["L_times_v"] = (L @ v).tolist()
        out["designs"][k]["cov_change_under_Sigma_plus_0.1vv'"] = float(np.max(np.abs(m.implied_cov(L, SIG1 + 0.1 * np.outer(v, v)) - m.implied_cov(L, SIG1))))
    # W3 leaky clamp: explicit latent reparameterization
    T_decl = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.3], [0, 0, 0, 0.7]], float)
    T3, err3 = m.basis_transform(m.L_NOMINAL, L3)
    S3 = T_decl @ SIG1 @ T_decl.T
    out["W3_leaky_clamp"] = {"declared_T (I'=I, S'=S+.3R, R'=.7R, Q'=Q)": T_decl.tolist(), "solved_T": T3.tolist(), "solved_equals_declared": float(np.max(np.abs(T3 - T_decl))),
                             "L_nominal_T_minus_L_true_all_arms": float(np.max(np.abs(m.L_NOMINAL @ T_decl - L3))),
                             "implied_cov_nominal(TSigmaT')_minus_true": float(np.max(np.abs(m.implied_cov(m.L_NOMINAL, S3) - m.implied_cov(L3, SIG1)))),
                             "restriction_residuals_population": m.restriction_residuals(m.implied_cov(L3, SIG1)).tolist(),
                             "named_Sigma_true": SIG1.tolist(), "named_Sigma_estimand_under_nominal_L": S3.tolist()}
    # seminar claim: changed independent D style loading (declared delta .5)
    LD = m.L_NOMINAL.copy(); LD[3] = [0, 1, 1.5, 0]
    TD, errD = m.basis_transform(m.L_NOMINAL, LD)
    out["changed_D_style_loading_delta_.5"] = {"true_D": LD[3].tolist(), "solved_T": TD.tolist(), "L_nominal_T_minus_L_true_all_arms": errD,
                                               "implied_cov_misfit": float(np.max(np.abs(m.implied_cov(m.L_NOMINAL, TD @ SIG1 @ TD.T) - m.implied_cov(LD, SIG1)))),
                                               "restriction_residuals_population": m.restriction_residuals(m.implied_cov(LD, SIG1)).tolist(),
                                               "conclusion": "absorbed by an invertible latent-coordinate change; C=B still holds; no observable misfit"}
    # unknown-L gauge: arbitrary invertible G
    G = np.random.default_rng([21008]).normal(size=(4, 4)) + 2 * np.eye(4)
    out["unknown_L_gauge"] = {"G": G.tolist(), "det_G": float(np.linalg.det(G)),
                              "implied_cov_diff": float(np.max(np.abs(m.implied_cov(m.L_NOMINAL @ np.linalg.inv(G), G @ SIG1 @ G.T) - m.implied_cov(m.L_NOMINAL, SIG1))))}
    # W4: C violation cannot be absorbed (rows B and C of L_nominal T are always equal)
    T4, err4 = m.basis_transform(m.L_NOMINAL, L4)
    out["W4_C_restriction_violated"] = {"best_basis_T_residual_on_C_row": err4, "restriction_residuals_population": m.restriction_residuals(m.implied_cov(L4, SIG1)).tolist()}
    out["W2_psd"] = {"eigenvalues": np.linalg.eigvalsh(SIG2).tolist(), "varA_minus_varB_population": float(m.implied_cov(m.L_NOMINAL, SIG2)[0, 0] - m.implied_cov(m.L_NOMINAL, SIG2)[1, 1])}
    return out


def ci(x):
    lo, hi = np.percentile(x, [2.5, 97.5], axis=0)
    return lo, hi


def main():
    t0 = time.time(); started = datetime.now(timezone.utc).isoformat()
    verify()
    if OUT.exists():
        raise FileExistsError("paired_four_component_identification_r1 exists; preserve earlier runs")
    wit = analytic_witnesses()
    OUT.mkdir(parents=True)
    (OUT / "analytic_witnesses.json").write_text(json.dumps(wit, indent=1) + "\n")
    est_rows, res_rows, app_rows, per = [], [], [], {}
    n_obs = 0
    boot_store = {}
    for wi, (wname, (Sig, Lt)) in enumerate(WORLDS.items()):
        C_pop = m.implied_cov(Lt, Sig)
        S_star, _ = m.sigma_hat(C_pop, m.L_NOMINAL)          # what the nominal estimator converges to
        for s in SEEDS:
            data, truth = m.simulate(Sig, Lt, MU, SD_ERR, N_SRC, N_APP, REPS, [21005, s])
            n_obs += data["Y_src"].size + data["Y_app"].size
            Ys = data["Y_src"]
            st = m.summary_stats(Ys)
            ident = float(np.max(np.abs(st["C"] - m.between_cov_crossrep(Ys))))
            eigC, vecC = np.linalg.eigh(st["C"])
            bs, idx = m.bootstrap(Ys, N_BOOT, [21006, wi, s])
            boot_store[f"{wname}|{s}|sigma_vec"] = bs["sigma_vec"]; boot_store[f"{wname}|{s}|restriction"] = bs["restriction"]
            lo, hi = ci(bs["sigma_vec"])
            for k, nm in enumerate(SIG_NAMES):
                tv, sv = Sig[IU][k], S_star[IU][k]
                est_rows.append({"world": wname, "seed": s, "element": nm, "estimate": st["sigma_vec"][k], "ci_lo": lo[k], "ci_hi": hi[k], "true_named": tv, "nominal_estimand": sv,
                                 "covers_true_named": bool(lo[k] <= tv <= hi[k]), "covers_nominal_estimand": bool(lo[k] <= sv <= hi[k])})
            rpop = m.restriction_residuals(C_pop)
            rlo, rhi = ci(bs["restriction"])
            V = np.cov(bs["restriction"], rowvar=False)
            W = float(st["restriction"] @ np.linalg.solve(V, st["restriction"]))
            for k, nm in enumerate(m.RESTRICTION_NAMES):
                res_rows.append({"world": wname, "seed": s, "moment": nm, "estimate": st["restriction"][k], "ci_lo": rlo[k], "ci_hi": rhi[k], "population": rpop[k],
                                 "ci_excludes_0": bool(rlo[k] > 0 or rhi[k] < 0)})
            contr = {}
            for key in ("varA_minus_varB", "varB_minus_varC", "paired_var_A_minus_B", "var_between_B_minus_C"):
                clo, chi_ = ci(bs[key])
                contr[key] = {"estimate": st[key], "ci": [float(clo), float(chi_)]}
            pop_contr = m.naive_and_paired(C_pop)
            # A/B/C-only design: estimable functions and null witness on data
            Labc = m.L_NOMINAL[:3]; Cabc = st["C"][:3, :3]
            S_abc, _ = m.sigma_hat(Cabc, Labc)
            v = m.NULL_WITNESS
            abc = {"estimable": {"var_choice(A-B paired)": st["paired_var_A_minus_B"], "var_I+S+R (B)": float(Cabc[1, 1]), "cov_choice_I+S+R (C_AB-C_BB)": float(Cabc[0, 1] - Cabc[1, 1])},
                   "fitted_cov_change_Sigma_hat_plus_0.1vv'": float(np.max(np.abs(m.implied_cov(Labc, S_abc + 0.1 * np.outer(v, v)) - m.implied_cov(Labc, S_abc)))),
                   "min_norm_Sigma_hat_diag": np.diag(S_abc).tolist()}
            # application scores (source-only centring; truth evaluation-only)
            sc = m.scores(data["Y_app"], st["means"])
            th = truth["theta_app"]
            K = m.CONTRASTS @ Lt
            X = np.column_stack([np.ones(len(th)), th])
            B_emp = np.linalg.lstsq(X, sc, rcond=None)[0][1:].T
            exp_noise = (m.CONTRASTS ** 2) @ st["noise"] / REPS
            aidx = np.random.default_rng([21007, wi, s]).integers(0, N_APP, (N_BOOT, N_APP))
            comp = {}
            for k, nm in enumerate(m.COMPONENTS):
                err = sc[:, k] - th[:, k]
                mse_b = (err[aidx] ** 2).mean(1)
                comp[nm] = {"bias": float(err.mean()), "mse": float((err ** 2).mean()), "mse_ci": [float(x) for x in np.percentile(mse_b, [2.5, 97.5])],
                            "expected_noise_mse_from_source": float(exp_noise[k]), "corr_score_truth": float(np.corrcoef(sc[:, k], th[:, k])[0, 1])}
            for j in range(len(th)):
                app_rows.append({"world": wname, "seed": s, "person": j, **{f"score_{c}": sc[j, k] for k, c in enumerate(m.COMPONENTS)}, **{f"true_{c}": th[j, k] for k, c in enumerate(m.COMPONENTS)}})
            per[f"{wname}|{s}"] = {"covariance_noise_identity_max_err": ident, "C_hat": st["C"].tolist(), "C_pop": C_pop.tolist(), "noise_hat": st["noise"].tolist(),
                                   "eigen_C_hat": eigC.tolist(), "eigvec_min_C_hat": vecC[:, 0].tolist(), "eigen_Sigma_hat": np.linalg.eigvalsh(st["Sigma"]).tolist(),
                                   "Sigma_hat": st["Sigma"].tolist(), "fit_residual_fro": st["resid_fro"], "restriction_wald_bootstrapV": W, "restriction_wald_p_chi2_5_approx": float(chi2.sf(W, 5)),
                                   "contrasts": contr, "contrasts_population": pop_contr, "ABC_only": abc, "score_mixing_theoretical_K": K.tolist(), "score_mixing_empirical": B_emp.tolist(),
                                   "score_summary": comp, "bootstrap_first_index_row_sha": hashlib.sha256(idx[0].tobytes()).hexdigest()}
            print(wname, s, round(time.time() - t0, 1), flush=True)
    pd.DataFrame(est_rows).to_csv(OUT / "source_component_estimates.csv", index=False)
    pd.DataFrame(res_rows).to_csv(OUT / "restriction_residuals.csv", index=False)
    pd.DataFrame(app_rows).to_csv(OUT / "application_scores.csv", index=False)
    np.savez_compressed(OUT / "bootstrap_draws.npz", **{k.replace("|", "__"): v for k, v in boot_store.items()})
    (OUT / "per_world_seed.json").write_text(json.dumps(per, indent=1) + "\n")
    E = pd.DataFrame(est_rows)
    res = {"protocol_sha256": PROTOCOL_SHA, "neural_calls": 0, "scalar_observations": int(n_obs), "worlds": list(WORLDS), "seeds": list(SEEDS),
           "coverage_descriptive": E.groupby("world")[["covers_true_named", "covers_nominal_estimand"]].mean().to_dict(),
           "identity_max_err": float(max(v["covariance_noise_identity_max_err"] for v in per.values())),
           "labels": {"coverage": "3 seeds x 10 elements: descriptive only, not nominal-coverage validation",
                      "components": "simulated components under declared loadings; not real-person personality accuracy",
                      "D_E": "idealized exclusion/anchor channels, not validated human tasks"},
           "seconds": time.time() - t0, "created": datetime.now(timezone.utc).isoformat()}
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    (OUT / "runtime.json").write_text(json.dumps({"started": started, "seconds": res["seconds"], "neural_calls": 0, "threads": 1, "code_sha256": sha(Path(__file__)),
                                                  "module_sha256": sha(ROOT / "grain_research/r05_four_component.py"), "protocol_sha256": PROTOCOL_SHA, "scalar_observations": int(n_obs)}, indent=1) + "\n")
    print("done", round(res["seconds"], 1), "obs", n_obs)


if __name__ == "__main__":
    main()
