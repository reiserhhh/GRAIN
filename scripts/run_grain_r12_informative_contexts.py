#!/usr/bin/env python
"""R12 packet 31 runner (docs/GRAIN_R12_INFORMATIVE_CONTEXTS_PROTOCOL_V1.md). Declared world, known loadings, 0 neural calls.
Generator and audit are the only places touching theta or generating variances; the source fitter sees Y and phi only."""
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
from grain_research import r12_informative_contexts as m  # noqa: E402

PROTOCOL = ROOT / "docs/GRAIN_R12_INFORMATIVE_CONTEXTS_PROTOCOL_V1.md"
PROTOCOL_SHA = "b452802a5809301207e60aa0fbcd23ee3035f1cb9f0d4e63b1bd345bcf7674f7"
OUT = ROOT / "results/grain_gptpro_foundation_v3/F04/informative_contexts_r1"
DESIGNS = ("D1", "D2", "D3", "D2_relabel")
REGIMES = {"tau0": 0.0, "tau.25": 0.25}
SIGMA2, SEEDS, NAPP, NSRC, BOOT = 1.0, (0, 1, 2), 400, 200, 2000
SIG_THETA = np.array([1.0, 0.25, 0.25])
Q = {"A": np.array([0, 1.0, 0]), "B": np.array([0, 0, 1.0]), "AB": np.array([0, 1.0, 1.0]) / np.sqrt(2)}
CONTRASTS = [("D1", "D2", ("A",)), ("D2", "D3", ("A", "B", "AB")), ("D2", "D2_relabel", ("A", "B", "AB"))]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify():
    if sha(PROTOCOL) != PROTOCOL_SHA:
        raise RuntimeError("protocol changed")
    for rel, hh in re.findall(r"\| `([^`]+)` \| `([0-9a-f]{64})` \|", PROTOCOL.read_text()):
        if sha(ROOT / rel) != hh:
            raise RuntimeError(f"binding mismatch {rel}")


def draws(seed, stream, U):
    rng = np.random.default_rng(np.random.SeedSequence([20260927, 31, seed, stream]))
    return {"theta": rng.standard_normal((U, 3)) * np.sqrt(SIG_THETA), "d0": rng.standard_normal((U, 8)), "e0": rng.standard_normal((U, 32))}


def generate(B, ctx, tau2):
    """Generator (truth used here only)."""
    Y = np.zeros((B["theta"].shape[0], 32))
    for k in ctx:
        Y[:, k["slots"]] = (B["theta"] @ k["phi"])[:, None] + np.sqrt(tau2) * B["d0"][:, [k["dslot"]]] + np.sqrt(SIGMA2) * B["e0"][:, k["slots"]]
    return Y


def main():
    t0 = time.time(); started = datetime.now(timezone.utc).isoformat()
    verify()
    if OUT.exists():
        raise FileExistsError("informative_contexts_r1 exists")
    OUT.mkdir(parents=True)
    ana, wit_rows, cal_rows, met_rows, rel_rows, con_rows = [], [], [], [], [], []
    checks = {"dense_vs_formula_max": 0.0, "relabel_score_max_abs": 0.0, "tau0_D2_D3_risk_max": 0.0, "unidentified_never_scored": True, "independent_risk_recompute_max": 0.0}
    # ---- analytic table
    for rg, t2 in REGIMES.items():
        for d in DESIGNS:
            c = m.design(d); I = m.info(c, SIGMA2, t2)
            checks["dense_vs_formula_max"] = max(checks["dense_vs_formula_max"], float(np.abs(I - m.info_dense(c, SIGMA2, t2)).max()))
            post = np.linalg.inv(np.diag(1 / SIG_THETA) + I)
            row = {"design": d, "regime": rg, "tau2": t2, "K_actual": len(m.actual_groups(c)), "K_nominal": len(c), "loading_energy": float(sum(k["n"] * (k["phi"][1] ** 2 + k["phi"][2] ** 2) for k in c))}
            for qn, q in Q.items():
                r, ok, wit, rank, eig = m.gls_risk(I, q)
                row[f"risk_{qn}"] = r; row[f"identified_{qn}"] = ok; row["rank"] = rank; row["eigenvalues"] = ";".join(f"{x:.6g}" for x in eig)
                row[f"prior_driven_posterior_var_{qn}"] = float(q @ post @ q)
                if not ok:
                    wit_rows.append({"design": d, "regime": rg, "target": qn, "null_witness": ";".join(f"{x:.6g}" for x in wit), "q_dot_witness": float(q @ wit), "I_times_witness_norm": float(np.linalg.norm(I @ wit))})
            if d == "D2_relabel":
                In = m.info(c, SIGMA2, t2, by="label")
                for qn, q in Q.items():
                    row[f"naive_wrong_risk_{qn}"] = m.gls_risk(In, q)[0]
            ana.append(row)
        r2 = {qn: m.gls_risk(m.info(m.design("D2"), SIGMA2, 0.0), q)[0] for qn, q in Q.items()}
        r3 = {qn: m.gls_risk(m.info(m.design("D3"), SIGMA2, 0.0), q)[0] for qn, q in Q.items()}
        checks["tau0_D2_D3_risk_max"] = max(abs(r2[k] - r3[k]) for k in Q)
    AN = pd.DataFrame(ana)
    store = {}
    for seed in SEEDS:
        BA, BS = draws(seed, 1, NAPP), draws(seed, 2, NSRC)
        idx = np.random.default_rng(np.random.SeedSequence([20260927, 31, seed, 9])).integers(0, NAPP, (BOOT, NAPP))
        truth = {qn: BA["theta"] @ q for qn, q in Q.items()}      # audit only
        for rg, t2 in REGIMES.items():
            src = m.source_fit(generate(BS, m.design("D3"), t2), m.design("D3"))   # frozen before scoring
            cal_rows.append({"seed": seed, "regime": rg, "tau2_true": t2, "sigma2_true": SIGMA2, **src})
            sq_err = {}
            scores_d = {}
            for d in DESIGNS:
                c = m.design(d); Y = generate(BA, c, t2); store[f"s{seed}_{rg}_{d}"] = Y.astype(np.float32)
                th_hat, I_est = m.gls_scores(Y, c, src["sigma2"], src["tau2_used"])
                _, I_or = m.gls_scores(Y, c, SIGMA2, t2)
                scores_d[d] = th_hat
                if d == "D2_relabel":
                    In = m.info(c, src["sigma2"], src["tau2_used"], by="label")
                for qn, q in Q.items():
                    r_or, ok, _, _, _ = m.gls_risk(I_or, q)
                    if not ok:
                        met_rows.append({"seed": seed, "regime": rg, "design": d, "target": qn, "status": "UNIDENTIFIED", "score": None})
                        continue
                    s = th_hat @ q; err = s - truth[qn]
                    se_est = np.sqrt(m.gls_risk(I_est, q)[0]); se_or = np.sqrt(r_or)
                    r_ind = float(q @ np.linalg.pinv(m.info_dense(c, SIGMA2, t2), rcond=1e-10) @ q)   # independent path: dense block inverse
                    checks["independent_risk_recompute_max"] = max(checks["independent_risk_recompute_max"], abs(r_or - r_ind))
                    row = {"seed": seed, "regime": rg, "design": d, "target": qn, "status": "SCORED", "analytic_risk_oracle": r_or, "empirical_mse": float((err ** 2).mean()),
                           "bias": float(err.mean()), "coverage_source_se": float((np.abs(err) <= 1.96 * se_est).mean()), "coverage_oracle_se": float((np.abs(err) <= 1.96 * se_or).mean()),
                           "se_source": float(se_est), "se_oracle": float(se_or), "sq_corr_score_truth": float(np.corrcoef(s, truth[qn])[0, 1] ** 2)}
                    if d == "D2_relabel":
                        se_n = np.sqrt(m.gls_risk(In, q)[0]); row["naive_wrong_se"] = float(se_n); row["naive_wrong_coverage"] = float((np.abs(err) <= 1.96 * se_n).mean())
                    met_rows.append(row); sq_err[(d, qn)] = err ** 2
            for qn in Q:
                if ("D2", qn) in sq_err:
                    diff = float(np.abs(scores_d["D2"] @ Q[qn] - scores_d["D2_relabel"] @ Q[qn]).max())
                    checks["relabel_score_max_abs"] = max(checks["relabel_score_max_abs"], diff)
                    rel_rows.append({"seed": seed, "regime": rg, "target": qn, "max_abs_score_diff_D2_vs_relabel": diff,
                                     "analytic_risk_D2": float(AN[(AN.design == "D2") & (AN.regime == rg)][f"risk_{qn}"].iloc[0]),
                                     "analytic_risk_relabel_correct": float(AN[(AN.design == "D2_relabel") & (AN.regime == rg)][f"risk_{qn}"].iloc[0]),
                                     "naive_wrong_risk_relabel": float(AN[(AN.design == "D2_relabel") & (AN.regime == rg)][f"naive_wrong_risk_{qn}"].iloc[0])})
            for d1, d0, tg in CONTRASTS:
                for qn in tg:
                    a1, a0 = sq_err[(d1, qn)], sq_err[(d0, qn)]
                    bs = a1[idx].mean(1) - a0[idx].mean(1); lo, hi = np.percentile(bs, [2.5, 97.5])
                    an1 = AN[(AN.design == d1) & (AN.regime == rg)][f"risk_{qn}"].iloc[0]; an0 = AN[(AN.design == d0) & (AN.regime == rg)][f"risk_{qn}"].iloc[0]
                    con_rows.append({"seed": seed, "regime": rg, "contrast": f"{d1}-{d0}", "target": qn, "point": float(a1.mean() - a0.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
                                     "ci_contains_0": bool(lo <= 0 <= hi), "analytic": float(an1 - an0), "conditioning": "application-author bootstrap; source noise fixed"})
    ME = pd.DataFrame(met_rows)
    checks["unidentified_never_scored"] = bool(ME[ME.status == "UNIDENTIFIED"].score.isna().all())
    AN.to_csv(OUT / "analytic_designs.csv", index=False); pd.DataFrame(wit_rows).to_csv(OUT / "rank_null_witnesses.csv", index=False)
    pd.DataFrame(cal_rows).to_csv(OUT / "source_calibration.csv", index=False); ME.to_csv(OUT / "target_metrics.csv", index=False)
    pd.DataFrame(rel_rows).to_csv(OUT / "relabel_checks.csv", index=False); pd.DataFrame(con_rows).to_csv(OUT / "paired_contrasts.csv", index=False)
    np.savez_compressed(OUT / "observation_archive.npz", **store)
    res = {"protocol_sha256": PROTOCOL_SHA, "neural_calls": 0, "seeds": list(SEEDS), "app_authors": NAPP, "source_authors": NSRC, "bootstrap": BOOT, "checks": checks,
           "seconds": time.time() - t0, "created": datetime.now(timezone.utc).isoformat(),
           "note": "balanced homoscedastic designs: GLS point estimates equal OLS on context means; source noise affects SE/coverage only"}
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    (OUT / "runtime.json").write_text(json.dumps({"started": started, "seconds": res["seconds"], "neural_calls": 0, "threads": 1, "code_sha256": sha(Path(__file__)),
                                                  "module_sha256": sha(ROOT / "grain_research/r12_informative_contexts.py"), "protocol_sha256": PROTOCOL_SHA}, indent=1) + "\n")
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
