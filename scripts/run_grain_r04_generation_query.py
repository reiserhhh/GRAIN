#!/usr/bin/env python
"""R04 packet 32 runner (docs/GRAIN_R04_GENERATION_QUERY_PROTOCOL_V1.md). Declared static world; 0 neural calls."""
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
from grain_research import r04_generation_query as m  # noqa: E402

PROTOCOL = ROOT / "docs/GRAIN_R04_GENERATION_QUERY_PROTOCOL_V1.md"
PROTOCOL_SHA = "9d2898360cf0b95f606c5727fcbe1644854dcb3a35b904fee2e81cae4889b9e8"
OUT = ROOT / "results/grain_gptpro_foundation_v3/F01/generation_query_r1"
SEEDS, NAPP, NSRC, NANC, BOOT = (0, 1, 2), 400, 200, 100, 2000


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify():
    if sha(PROTOCOL) != PROTOCOL_SHA:
        raise RuntimeError("protocol changed")
    for rel, hh in re.findall(r"\| `([^`]+)` \| `([0-9a-f]{64})` \|", PROTOCOL.read_text()):
        if sha(ROOT / rel) != hh:
            raise RuntimeError(f"binding mismatch {rel}")


def draws(seed, role, U, bilingual=False):
    rng = np.random.default_rng(np.random.SeedSequence([20260927, 32, seed, role]))
    B = {"th0": rng.standard_normal(U), "z": rng.standard_normal(U), "Ex": rng.standard_normal((U, m.T, 3)), "Eq": rng.standard_normal((U, m.T, 2)), "sel": rng.standard_normal((U, 2))}
    if bilingual:
        B["Ex1"] = rng.standard_normal((U, m.T, 3)); B["Eq1"] = rng.standard_normal((U, m.T, 2))
    return B


def expected_readouts(theta, K, lang, g, a, wA, lam, dvec, pair):
    """Independent noise-free readout function with EXPLICIT parameters (for witnesses / recompute)."""
    e = theta * m.W_TH + g * m.W_G + a * wA + dvec
    x = K * (m.M[lang] @ e)
    return np.array([m.SENS[q][0] @ x + lam[j] * a for j, q in enumerate(m.PAIRS[pair])])


def main():
    t0 = time.time(); started = datetime.now(timezone.utc).isoformat()
    verify()
    if OUT.exists():
        raise FileExistsError("generation_query_r1 exists")
    OUT.mkdir(parents=True)
    checks = {}
    # ---------------- exact witnesses (analytic, verified numerically)
    lam_d = [m.SENS[q][1] for q in m.PAIRS["distinct"]]
    b = {k: np.array([m.SENS[q][0] @ m.M[0] @ w for q in m.PAIRS["distinct"]]) for k, w in (("th", m.W_TH), ("G", m.W_G), ("A", m.W_A))}
    mtrue = expected_readouts(0.5, 1.0, 0, 0.0, 1.0, m.W_A, lam_d, m.D[0], "distinct")
    wit1 = None
    for gp, ap in ((1.0, 1.0), (1.0, 0.0), (0.0, 0.0)):
        A_ = np.stack([b["th"], gp * b["G"] + ap * b["A"]], 1); rhs = mtrue - np.array(lam_d) * ap
        u, Kp = np.linalg.solve(A_, rhs)
        if Kp > 0:
            mr = expected_readouts(u / Kp, Kp, 0, gp, ap, m.W_A, lam_d, m.D[0], "distinct")
            wit1 = {"true": {"theta": 0.5, "K": 1.0, "g": 0, "a": 1}, "rival": {"theta": float(u / Kp), "K": float(Kp), "g": gp, "a": ap}, "max_abs_readout_diff": float(np.abs(mr - mtrue).max())}
            break
    checks["witness_conf_max_abs"] = wit1["max_abs_readout_diff"]
    # W2 reader-side vs generation-side (constant K): delta w in null(r_q1) with r_q2' M dw = lam_q2 / K
    r1, r2 = m.SENS["q1"][0], m.SENS["q2"][0]
    n = np.cross(r1, np.cross(r2, r1)); dw = n * (m.SENS["q2"][1] / (r2 @ n))
    diffs_const, diffs_var = [], []
    for K in (1.0,):
        for g_ in (0.0, 1.0):
            for a_ in (0.0, 1.0):
                A1 = expected_readouts(0.3, K, 0, g_, a_, m.W_A, lam_d, m.D[0], "distinct")
                A2 = expected_readouts(0.3, K, 0, g_, a_, m.W_A + dw / K, [0.0, 0.0], m.D[0], "distinct")
                diffs_const.append(np.abs(A1 - A2).max())
    for K in (0.7, 1.4):
        A1 = expected_readouts(0.3, K, 0, 0.0, 1.0, m.W_A, lam_d, m.D[0], "distinct"); A2 = expected_readouts(0.3, K, 0, 0.0, 1.0, m.W_A + dw / 1.0, [0.0, 0.0], m.D[0], "distinct")
        diffs_var.append(np.abs(A1 - A2).max())
    checks["witness_readerside_max_abs"] = float(max(diffs_const)); checks["readerside_rival_breaks_when_K_varies"] = float(max(diffs_var))
    # W3 language offset vs group mean without anchors: theta' = theta - .4, d1' = d1 + .4 w_th
    diffs3 = [np.abs(expected_readouts(th, 1.2, 1, g_, a_, m.W_A, lam_d, m.D[1], "distinct") - expected_readouts(th - m.MU_L1, 1.2, 1, g_, a_, m.W_A, lam_d, m.D[1] + m.MU_L1 * m.W_TH, "distinct")).max()
              for th in (-1.0, 0.4, 2.0) for g_ in (0.0, 1.0) for a_ in (0.0, 1.0)]
    checks["witness_language_max_abs"] = float(max(diffs3))
    # W4 ranks / singular values
    Rd = np.stack([m.SENS[q][0] for q in m.PAIRS["distinct"]]); Rr = np.stack([m.SENS[q][0] for q in m.PAIRS["replica"]])
    Jc = np.array([[1.0 * b["th"][j], (0.5 * b["th"][j] + g_ * b["G"][j] + a_ * b["A"][j])] for g_ in (0, 1) for a_ in (0, 1) for j in range(2)])
    rank_rows = [{"object": "sensor map R (distinct)", "rank": int(np.linalg.matrix_rank(Rd)), "singular_values": ";".join(f"{s:.4g}" for s in np.linalg.svd(Rd, compute_uv=False))},
                 {"object": "sensor map R (replica)", "rank": int(np.linalg.matrix_rank(Rr)), "singular_values": ";".join(f"{s:.4g}" for s in np.linalg.svd(Rr, compute_uv=False))},
                 {"object": "per-author Jacobian d(means)/d(theta, logK), crossed recorded G x A, distinct", "rank": int(np.linalg.matrix_rank(Jc)), "singular_values": ";".join(f"{s:.4g}" for s in np.linalg.svd(Jc, compute_uv=False))},
                 {"object": "CONF: unknowns (theta, logK, g, a) vs 2 distinct-sensor mean equations", "rank": 2, "singular_values": "4 unknowns > 2 equations; exact rival W1"}]
    # ---------------- simulation
    met, con, rsd, qcov, conf_rows = [], [], [], [], []
    recompute = 0.0
    for seed in SEEDS:
        BA, BS, BN = draws(seed, 1, NAPP), draws(seed, 2, NSRC), draws(seed, 3, NANC, bilingual=True)
        LA = np.r_[np.zeros(NAPP // 2, int), np.ones(NAPP // 2, int)]; thA = BA["th0"] + m.MU_L1 * LA; KA = np.exp(m.SD_LOGK * BA["z"])
        LS = np.r_[np.zeros(NSRC // 2, int), np.ones(NSRC // 2, int)]; thS = BS["th0"] + m.MU_L1 * LS; KS = np.exp(m.SD_LOGK * BS["z"])
        thN, KN = BN["th0"], np.exp(m.SD_LOGK * BN["z"])
        gA, aA = m.crossed(NAPP); gS, aS = m.crossed(NSRC); gN, aN = m.crossed(NANC)
        idx = np.random.default_rng(np.random.SeedSequence([20260927, 32, seed, 9])).integers(0, NAPP, (BOOT, NAPP))
        sq = {}
        for pair in ("distinct", "replica"):
            ys = m.author_ols(m.generate(thS, KS, LS, gS, aS, pair, BS["Ex"], BS["Eq"]), gS, aS)
            src0 = {k: v[LS == 0] for k, v in ys.items()}; src1 = {k: v[LS == 1] for k, v in ys.items()}
            a0 = m.author_ols(m.generate(thN, KN, np.zeros(NANC, int), gN, aN, pair, BN["Ex"], BN["Eq"]), gN, aN)
            a1 = m.author_ols(m.generate(thN, KN, np.ones(NANC, int), gN, aN, pair, BN["Ex1"], BN["Eq1"]), gN, aN)
            cals = {"anchor": m.calibrate(src0, anchor0=a0, anchor1=a1), "oracle": m.oracle_cal(pair)}
            if pair == "distinct":
                cals["noanchor"] = m.calibrate(src0, src1_mono=src1)
            for j, rs in enumerate(m.reader_side(src0)):
                q = m.PAIRS[pair][j]; r, lam = m.SENS[q]
                rsd.append({"seed": seed, "pair": pair, "query": q, **rs, "lambda_generating": lam, "slope_generating": float((r @ m.W_A) / (r @ m.W_G))})
            st = m.author_ols(m.generate(thA, KA, LA, gA, aA, pair, BA["Ex"], BA["Eq"]), gA, aA)
            rho, _ = m.ratio(st)
            for cname, cal in cals.items():
                th_hat, se = m.score(st, LA, cal); err = th_hat - thA; arm = f"CROSS_{pair.upper()}_{cname.upper()}"
                sq[arm] = err ** 2
                for grp, s in (("all", slice(None)), ("L0", LA == 0), ("L1", LA == 1)):
                    met.append({"seed": seed, "arm": arm, "group": grp, "status": "SCORED", "bias": float(err[s].mean()), "mse": float((err[s] ** 2).mean()), "median_abs_err": float(np.median(np.abs(err[s]))),
                                "coverage95": float((np.abs(err[s]) <= 1.96 * se[s]).mean()), "sq_corr_thetahat_theta": float(np.corrcoef(th_hat[s], thA[s])[0, 1] ** 2), "calibration": cal["method_L1"]})
                if cname == "anchor":
                    thq = (rho - np.where(LA[:, None] == 1, cal["off"][1], cal["off"][0])) / np.where(LA[:, None] == 1, cal["scale"][1], cal["scale"][0])
                    qcov.append({"seed": seed, "pair": pair, "corr_query_scores": float(np.corrcoef(thq[:, 0], thq[:, 1])[0, 1]), "corr_query_raw_intercepts": float(np.corrcoef(st["alpha"][:, 0], st["alpha"][:, 1])[0, 1])})
                    if seed == 0 and pair == "distinct":     # independent recompute (no module estimator)
                        for u in range(5):
                            X = np.column_stack([np.ones(m.T), gA[u], aA[u]]); yy = m.generate(thA[u:u + 1], KA[u:u + 1], LA[u:u + 1], gA[u:u + 1], aA[u:u + 1], pair, BA["Ex"][u:u + 1], BA["Eq"][u:u + 1])[0]
                            co = np.linalg.lstsq(X, yy, rcond=None)[0]; lg = LA[u]
                            th_i = np.mean([(co[0, q] / co[1, q] - cal["off"][lg][q]) / cal["scale"][lg][q] for q in range(2)])
                            recompute = max(recompute, abs(th_i - th_hat[u]))
        # CONF (distinct sensors; G/A unrecorded, self-selected): theta REFUSED; naive standardized readout as non-estimator reference
        gC = np.tile((thA + BA["sel"][:, 0] > 0).astype(float)[:, None], (1, m.T)); aC = np.tile((-thA + BA["sel"][:, 1] > 0).astype(float)[:, None], (1, m.T))
        gCs = np.tile((thS + BS["sel"][:, 0] > 0).astype(float)[:, None], (1, m.T)); aCs = np.tile((-thS + BS["sel"][:, 1] > 0).astype(float)[:, None], (1, m.T))
        yC = m.generate(thA, KA, LA, gC, aC, "distinct", BA["Ex"], BA["Eq"]).mean(1); ySC = m.generate(thS, KS, LS, gCs, aCs, "distinct", BS["Ex"], BS["Eq"]).mean(1)
        mu, sd = ySC[LS == 0].mean(0), ySC[LS == 0].std(0, ddof=1); naive = ((yC - mu) / sd).mean(1)
        conf_rows.append({"seed": seed, "status_theta": "UNIDENTIFIED (refused; exact rival W1)", "naive_corr_theta": float(np.corrcoef(naive, thA)[0, 1]), "naive_corr_K": float(np.corrcoef(naive, KA)[0, 1]),
                          "naive_corr_g": float(np.corrcoef(naive, gC[:, 0])[0, 1]), "naive_corr_a": float(np.corrcoef(naive, aC[:, 0])[0, 1]), "naive_mse_vs_theta_not_an_estimator": float(((naive - thA) ** 2).mean())})
        for c_name, a1n, a0n, sub in (("distinct-replica", "CROSS_DISTINCT_ANCHOR", "CROSS_REPLICA_ANCHOR", None), ("anchor-noanchor", "CROSS_DISTINCT_ANCHOR", "CROSS_DISTINCT_NOANCHOR", None),
                                      ("anchor-noanchor_L1only", "CROSS_DISTINCT_ANCHOR", "CROSS_DISTINCT_NOANCHOR", LA == 1), ("distinct-oracle", "CROSS_DISTINCT_ANCHOR", "CROSS_DISTINCT_ORACLE", None)):
            d = sq[a1n] - sq[a0n]
            if sub is not None:
                ii = np.flatnonzero(sub)
                idx_s = np.random.default_rng(np.random.SeedSequence([20260927, 32, seed, 11])).integers(0, len(ii), (BOOT, len(ii))); bs = d[ii][idx_s].mean(1); pt = float(d[ii].mean())
            else:
                bs = d[idx].mean(1); pt = float(d.mean())
            lo, hi = np.percentile(bs, [2.5, 97.5])
            con.append({"seed": seed, "contrast": c_name, "quantity": "squared error of theta-hat", "point": pt, "ci_lo": float(lo), "ci_hi": float(hi), "ci_contains_0": bool(lo <= 0 <= hi),
                        "conditioning": "application-author bootstrap; source calibration fixed"})
    checks["conf_theta_refused"] = True
    checks["independent_recompute_max_abs"] = float(recompute)
    pd.DataFrame(met).to_csv(OUT / "target_metrics.csv", index=False); pd.DataFrame(con).to_csv(OUT / "paired_contrasts.csv", index=False)
    pd.DataFrame(rsd).to_csv(OUT / "reader_vs_generation.csv", index=False); pd.DataFrame(qcov).to_csv(OUT / "query_covariance.csv", index=False)
    pd.DataFrame(conf_rows).to_csv(OUT / "confounded_design.csv", index=False); pd.DataFrame(rank_rows).to_csv(OUT / "identification_rank.csv", index=False)
    (OUT / "witnesses.json").write_text(json.dumps({"W1_conf_theta_vs_K": wit1, "W2_readerside_constant_K": {"delta_w": dw.tolist(), "max_abs_diff_constant_K": checks["witness_readerside_max_abs"], "max_abs_diff_when_K_varies": checks["readerside_rival_breaks_when_K_varies"]},
                                                    "W3_language_offset_vs_group_mean": {"rival": "theta' = theta - .4 for L1, d1' = d1 + .4 w_th", "max_abs_diff": checks["witness_language_max_abs"]}}, indent=1) + "\n")
    pd.DataFrame([{"role": role, "seed": s, "stream": f"SeedSequence([20260927,32,{s},{k}])", "authors": n} for s in SEEDS for role, k, n in (("application", 1, NAPP), ("source_monolingual", 2, NSRC), ("source_bilingual_anchor", 3, NANC))]).to_csv(OUT / "role_id_manifest.csv", index=False)
    res = {"protocol_sha256": PROTOCOL_SHA, "neural_calls": 0, "checks": checks, "seconds": time.time() - t0, "created": datetime.now(timezone.utc).isoformat(),
           "observations": {"application_events": len(SEEDS) * NAPP * m.T * 2, "source_events": len(SEEDS) * (NSRC + 2 * NANC) * m.T * 2}}
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    (OUT / "runtime.json").write_text(json.dumps({"started": started, "seconds": res["seconds"], "neural_calls": 0, "threads": 1, "code_sha256": sha(Path(__file__)),
                                                  "module_sha256": sha(ROOT / "grain_research/r04_generation_query.py"), "protocol_sha256": PROTOCOL_SHA}, indent=1) + "\n")
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
