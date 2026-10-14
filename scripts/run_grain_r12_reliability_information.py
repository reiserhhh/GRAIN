#!/usr/bin/env python
"""R12 packet 28 runner (docs/GRAIN_R12_RELIABILITY_INFORMATION_PROTOCOL_V1.md). Declared scalar world only; 0 neural
calls; one paired 3-seed simulation over the OLD design grid + one equal-Szz diagnostic family; FU2-c W1/D1 exact replay
with person-bootstrap CIs. Readings are simulated readings, not language tokens."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research import r12_reliability_information as m  # noqa: E402
import scripts.run_grain_r18_rival_simulation as R18  # noqa: E402  (frozen; functions only, main() never called)

PROTOCOL = ROOT / "docs/GRAIN_R12_RELIABILITY_INFORMATION_PROTOCOL_V1.md"
PROTOCOL_SHA = "f353fb04b8503e8d8105e82642b75e1f126f417c3a40af2348261f11e8182d8b"
V3 = ROOT / "results/grain_gptpro_foundation_v3"
OUT = V3 / "F04/reliability_information_r1"
OLD = V3 / "F04/excitation_reliability_r2/excitation_reliability.json"
F09 = V3 / "F09/rival_person_simulation_r1/rival_person_metrics.csv"
SEEDS, N, NMAX, BOOT = (0, 1, 2), 400, 64, 2000
TARGET = 80.0 / 9.0
DESIGNS = [("n8_K4", 8, 4, False), ("n16_K4", 16, 4, False), ("n32_K4", 32, 4, False), ("n64_K4", 64, 4, False),
           ("n32_K1", 32, 1, False), ("n32_K2", 32, 2, False), ("n32_K8", 32, 8, False),
           ("eq_n32_K2", 32, 2, True), ("eq_n32_K4", 32, 4, True), ("eq_n32_K8", 32, 8, True)]
FAMILIES = {"tokens": (["n8_K4", "n16_K4", "n32_K4", "n64_K4"], np.log2([8, 16, 32, 64])),
            "diversity": (["n32_K2", "n32_K4", "n32_K8"], np.log2([2, 4, 8])),
            "equal_szz": (["eq_n32_K2", "eq_n32_K4", "eq_n32_K8"], np.log2([2, 4, 8]))}
CONTRASTS = [("n16_K4", "n8_K4"), ("n32_K4", "n16_K4"), ("n64_K4", "n32_K4"),
             ("n32_K4", "n32_K2"), ("n32_K8", "n32_K4"), ("n32_K8", "n32_K2"),
             ("eq_n32_K4", "eq_n32_K2"), ("eq_n32_K8", "eq_n32_K4"), ("eq_n32_K8", "eq_n32_K2"),
             ("eq_n32_K2", "n32_K2"), ("eq_n32_K8", "n32_K8")]
STATS = ("half_r", "whole_SB", "whole_truth", "mse_full")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify():
    if sha(PROTOCOL) != PROTOCOL_SHA:
        raise RuntimeError("protocol changed")
    for rel, hh in re.findall(r"\| `([^`]+)` \| `([0-9a-f]{64})` \|", PROTOCOL.read_text()):
        if sha(ROOT / rel) != hh:
            raise RuntimeError(f"binding mismatch {rel}")


def stats(bA, bB, bF, b):
    """Vectorised along the last (author) axis."""
    hr = m.corr_rows(bA, bB)
    return {"half_r": hr, "whole_SB": m.spearman_brown(hr), "whole_truth": m.corr_rows(bF, b) ** 2,
            "mse_full": ((bF - b) ** 2).mean(-1), "mse_half": 0.5 * (((bA - b) ** 2).mean(-1) + ((bB - b) ** 2).mean(-1))}


def ci(x):
    lo, hi = np.percentile(x, [2.5, 97.5])
    return [float(lo), float(hi)]


def design_table():
    rows, zs = [], {}
    for did, n, k, eq in DESIGNS:
        A, B = m.halves(n)
        try:
            z = m.design_z(n, k)
        except m.Unidentified as exc:
            zz = np.ones(n)
            rows.append({"design_id": did, "readings": n, "K": k, "equal_szz": eq, "scored": False, "reason": str(exc), "szz_half_A": 0.0, "szz_half_B": 0.0,
                         "szz_full": 0.0, "info_rank_half": m.information(zz[A])["rank"], "info_min_pos_eig_half": m.information(zz[A])["min_positive_eig"]})
            continue
        if eq:
            z = m.equal_szz(z, TARGET)
        zs[did] = z
        sA, sB, sF = m.szz(z[A]), m.szz(z[B]), m.szz(z)
        vA, vB = m.slope_var(sA), m.slope_var(sB)
        hr = float(m.half_reliability(vA, vB)); inf = m.information(z[A])
        rows.append({"design_id": did, "readings": n, "K": k, "equal_szz": eq, "scored": True, "reason": "", "z_min": float(z.min()), "z_max": float(z.max()),
                     "szz_half_A": sA, "szz_half_B": sB, "szz_full": sF, "info_rank_half": inf["rank"], "info_min_pos_eig_half": inf["min_positive_eig"],
                     "v_half_A": vA, "v_half_B": vB, "half_r_theory": hr, "whole_SB_theory": float(m.spearman_brown(hr)), "whole_theory": float(m.full_reliability(sF)),
                     "mse_full_theory": m.slope_var(sF), "mse_half_theory": 0.5 * (vA + vB), "posterior_var_b_theory_separate": m.posterior_var_b(z)})
    return rows, zs


def fu2c_replay(outdir):
    """Exact replay of F09 W1/D1 only with the frozen functions; person-bootstrap CIs for LPM mean effect and retest r."""
    old = pd.read_csv(F09); old = old[(old.world == "W1") & (old.design == "D1")].set_index("coordinate")
    rng = np.random.default_rng(np.random.SeedSequence([R18.SEED, 0, 0]))
    theta, alpha, gamma, trait = R18.persons(rng, "W1")
    s1 = R18.simulate_session(rng, theta, alpha, gamma, "D1"); s2 = R18.simulate_session(rng, theta, alpha, gamma, "D1")
    e1, v1, names = R18.estimate(s1, "D1"); e2, _, _ = R18.estimate(s2, "D1")
    brng = np.random.default_rng(np.random.SeedSequence([R18.SEED, 0, 0, 99]))
    cols = ("true_person_sd", "mean_estimate", "true_mean", "recovery_r", "retest_r", "est_person_sd", "corr_with_proficiency_gamma", "r2_on_trait", "partial_retest_given_trait")
    worst, ciworst, rep = 0.0, 0.0, {}
    truths = []
    for j, name in enumerate(names):
        truth = gamma if name == "M5_convention" else theta[:, R18.MECH.index(name)]
        truths.append(truth)
        a_, b_ = e1[:, j], e2[:, j]
        r = {"true_person_sd": float(np.std(truth)), "mean_estimate": float(a_.mean()), "true_mean": float(truth.mean()), "recovery_r": R18.corr(a_, truth),
             "retest_r": R18.corr(a_, b_), "est_person_sd": float(np.sqrt(max(0.0, a_.var(ddof=1) - v1[:, j].mean()))), "corr_with_proficiency_gamma": R18.corr(a_, gamma),
             "r2_on_trait": R18.corr(a_, trait) ** 2, "partial_retest_given_trait": R18.partial(a_, b_, trait)}
        cio = R18.boot_ci(lambda i, a=a_, b=b_: R18.corr(a[i], b[i]), brng)
        worst = max(worst, max(abs(r[c] - float(old.loc[name, c])) for c in cols))
        ciworst = max(ciworst, float(np.max(np.abs(np.array(cio) - np.array(json.loads(old.loc[name, "retest_ci"]))))))
        rep[name] = r
    outdir.mkdir()
    np.savez(outdir / "person_arrays.npz", session1_lpm=e1, session2_lpm=e2, session1_var=v1, truth_logit=np.column_stack(truths), coordinates=np.array(names), trait=trait)
    ok = worst <= 1e-9 and ciworst <= 1e-9
    rows = []
    if ok:
        idx = np.random.default_rng(np.random.SeedSequence([20260927, 28, 901])).integers(0, R18.N, (BOOT, R18.N))
        for j, name in enumerate(names):
            A1, A2 = e1[idx, j], e2[idx, j]
            mean_draws = A1.mean(1); r_draws = m.corr_rows(A1, A2)
            rows.append({"coordinate": name, "unit": "LPM (linear-probability) person estimates", "mean_effect_session1_lpm": rep[name]["mean_estimate"],
                         "mean_effect_ci": ci(mean_draws), "retest_r": rep[name]["retest_r"], "retest_r_ci_new2000": ci(r_draws),
                         "retest_r_ci_old1000_reproduced": json.loads(old.loc[name, "retest_ci"]), "true_mean_logit_separate_unit": rep[name]["true_mean"],
                         "true_person_sd_logit_separate_unit": rep[name]["true_person_sd"]})
        pd.DataFrame(rows).to_csv(outdir / "fu2c_uncertainty.csv", index=False)
    return {"status": "RECOVERED" if ok else "OPEN_REPLAY_MISMATCH", "replay_max_abs": worst, "old_retest_ci_max_abs": ciworst, "persons": R18.N, "sessions": 2,
            "rows": rows, "note": "LPM estimates vs logit truth never compared numerically; template-level dual reports are not person-level evidence"}


def main():
    t0 = time.time(); started = datetime.now(timezone.utc).isoformat()
    verify()
    if OUT.exists():
        raise FileExistsError("reliability_information_r1 exists; preserve earlier runs")
    OUT.mkdir(parents=True)
    table, zs = design_table()
    scored = [r["design_id"] for r in table if r["scored"]]
    tab = {r["design_id"]: r for r in table}
    gates = {"k1_refused": not tab["n32_K1"]["scored"] and tab["n32_K1"]["info_rank_half"] == 1,
             "equal_szz_max_dev": max(abs(tab[d]["szz_half_A"] - TARGET) + abs(tab[d]["szz_half_B"] - TARGET) for d in FAMILIES["equal_szz"][0])}
    old = json.loads(OLD.read_text())
    oldv = {f"n{r['readings']}_K4": r["slope_spearman_brown"] for r in old["R12_reliability_vs_tokens"]}
    oldv.update({f"n32_K{r['conditions']}": r["slope_spearman_brown"] for r in old["R12_reliability_vs_diversity_fixed_32"] if r["conditions"] > 1})
    gates["analytic_vs_old_curve_max_abs"] = max(abs(tab[d]["whole_theory"] - v) for d, v in oldv.items())
    # ---- paired simulation with common random numbers
    arrays, ehash = {}, []
    A_all, B_all = np.zeros((3, N)), np.zeros((3, N))
    est = {d: {"bA": np.zeros((3, N)), "bB": np.zeros((3, N)), "bF": np.zeros((3, N))} for d in scored}
    for s in SEEDS:
        rng = np.random.default_rng(np.random.SeedSequence([20260927, 28, s]))
        a = rng.normal(0.0, np.sqrt(m.VAR_A), N); b = rng.normal(m.MEAN_B, m.SD_B, N); E = rng.normal(0.0, m.SIGMA, (N, NMAX))
        A_all[s], B_all[s] = a, b; ehash.append(hashlib.sha256(E.tobytes()).hexdigest())
        for d in scored:
            z = zs[d]; n = len(z); hA, hB = m.halves(n)
            Y = a[:, None] + b[:, None] * z[None, :] + E[:, :n]
            est[d]["bA"][s] = m.ols_slopes(Y[:, hA], z[hA]); est[d]["bB"][s] = m.ols_slopes(Y[:, hB], z[hB]); est[d]["bF"][s] = m.ols_slopes(Y, z)
    gates["crn_shared_noise"] = True
    for d in scored:
        for k_ in ("bA", "bB", "bF"):
            arrays[f"{d}__{k_}"] = est[d][k_]
    np.savez(OUT / "author_arrays.npz", a=A_all, b=B_all, design_ids=np.array(scored), **arrays)
    # ---- point estimates
    per_seed, pooled = {}, {}
    for d in scored:
        for s in SEEDS:
            st = stats(est[d]["bA"][s], est[d]["bB"][s], est[d]["bF"][s], B_all[s])
            per_seed[f"{d}|seed{s}"] = {f"{k_}_empirical": float(v) for k_, v in st.items()}
        st = stats(est[d]["bA"].ravel(), est[d]["bB"].ravel(), est[d]["bF"].ravel(), B_all.ravel())
        pooled[d] = {f"{k_}_empirical": float(v) for k_, v in st.items()}
    # ---- paired cluster bootstrap (same draws for every design)
    idx = m.stratified_indices(N, 3, BOOT, np.random.default_rng(np.random.SeedSequence([20260927, 28, 900])))
    bflat = B_all.ravel()[idx]
    draws, seed_draws = {}, {}
    for d in scored:
        bA, bB, bF = (est[d][k_].ravel()[idx] for k_ in ("bA", "bB", "bF"))
        draws[d] = stats(bA, bB, bF, bflat)
        seed_draws[d] = [stats(bA[:, s * N:(s + 1) * N], bB[:, s * N:(s + 1) * N], bF[:, s * N:(s + 1) * N], bflat[:, s * N:(s + 1) * N]) for s in SEEDS]
    ci_pooled = {d: {k_: ci(draws[d][k_]) for k_ in STATS} for d in scored}
    ci_seed = {f"{d}|seed{s}": {k_: ci(seed_draws[d][s][k_]) for k_ in STATS} for d in scored for s in SEEDS}
    contrasts = {}
    for d1, d2 in CONTRASTS:
        key = f"{d1}-{d2}"
        contrasts[key] = {}
        for k_ in STATS:
            x = draws[d1][k_] - draws[d2][k_]
            pt = pooled[d1][f"{k_}_empirical"] - pooled[d2][f"{k_}_empirical"]
            th = {"half_r": "half_r_theory", "whole_SB": "whole_theory", "whole_truth": "whole_theory", "mse_full": "mse_full_theory"}[k_]
            lo, hi = ci(x)
            contrasts[key][k_] = {"point": pt, "ci": [lo, hi], "ci_contains_0": bool(lo <= 0 <= hi), "theory": tab[d1][th] - tab[d2][th],
                                  "per_seed": [{"point": per_seed[f"{d1}|seed{s}"][f"{k_}_empirical"] - per_seed[f"{d2}|seed{s}"][f"{k_}_empirical"],
                                                "ci": ci(seed_draws[d1][s][k_] - seed_draws[d2][s][k_])} for s in SEEDS]}
    # ---- curvature on log2 grids (K1 excluded)
    curv = {}
    for fam, (ds, x) in FAMILIES.items():
        curv[fam] = {"grid_log2": x.tolist(), "spacing": 1.0}
        for k_, th in (("whole_SB", "whole_theory"), ("half_r", "half_r_theory")):
            f = np.array([pooled[d][f"{k_}_empirical"] for d in ds]); fd = np.stack([draws[d][k_] for d in ds], 1)
            D, C = m.divided_differences(x, f); Dd, Cd = m.divided_differences(x, fd)
            Dt, Ct = m.divided_differences(x, np.array([tab[d][th] for d in ds]))
            curv[fam][k_] = {"first_dd": D.tolist(), "first_dd_ci": [ci(Dd[:, i]) for i in range(Dd.shape[1])], "first_dd_theory": Dt.tolist(),
                             "second_dd": C.tolist(), "second_dd_ci": [ci(Cd[:, i]) for i in range(Cd.shape[1])], "second_dd_theory": Ct.tolist(),
                             "theory_inside_ci": [bool(ci(Cd[:, i])[0] <= Ct[i] <= ci(Cd[:, i])[1]) for i in range(len(Ct))],
                             "per_seed_second_dd": [m.divided_differences(x, np.array([per_seed[f"{d}|seed{s}"][f"{k_}_empirical"] for d in ds]))[1].tolist() for s in SEEDS]}
    fu = fu2c_replay(OUT / "fu2c_replay")
    gates["fu2c_replay_max_abs"] = fu["replay_max_abs"]; gates["fu2c_old_retest_ci_max_abs"] = fu["old_retest_ci_max_abs"]
    pd.DataFrame(table).to_csv(OUT / "design_table.csv", index=False)
    res = {"protocol_sha256": PROTOCOL_SHA, "neural_calls": 0, "world": "y=a+b z+e; a~N(0,1), b~N(.5,.5^2), e~N(0,.5^2); readings are simulated, not language tokens",
           "seeds": list(SEEDS), "authors_per_seed": N, "bootstrap_draws": BOOT, "noise_sha256_per_seed": ehash, "equal_szz_target_half": TARGET,
           "design_table": table, "old_curve_sb": oldv, "per_seed": per_seed, "pooled": pooled, "ci_pooled_stratified": ci_pooled, "ci_per_seed": ci_seed,
           "contrasts": contrasts, "curvature": curv, "fu2c": fu, "gates": gates,
           "labels": {"whole_SB": "Spearman-Brown of empirical half r", "whole_truth": "empirical corr(full slope, true b)^2 (simulation only)",
                      "whole_theory": "analytic Var(b)/(Var(b)+sigma2/Szz_full)", "posterior": "Bayes posterior variance, separate from OLS repeat reliability",
                      "cis": "conditional simulated-population intervals; authors resampled within seed; paired across designs"}}
    res["seconds"] = time.time() - t0; res["created"] = datetime.now(timezone.utc).isoformat()
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    (OUT / "runtime.json").write_text(json.dumps({"started": started, "seconds": res["seconds"], "neural_calls": 0, "threads": 1, "code_sha256": sha(Path(__file__)),
                                                  "module_sha256": sha(ROOT / "grain_research/r12_reliability_information.py"), "protocol_sha256": PROTOCOL_SHA}, indent=1) + "\n")
    print(json.dumps(gates, indent=1, default=float)); print("fu2c", fu["status"]); print("done", round(res["seconds"], 1))


if __name__ == "__main__":
    main()
