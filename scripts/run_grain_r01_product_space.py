#!/usr/bin/env python
"""R01 product-space identification of the frozen z->action estimators (docs/GRAIN_R01_PRODUCT_SPACE_IDENTIFICATION_PROTOCOL_V1.md).
Cached arrays only; 0 neural calls; no fitting beyond reconstructing the frozen closed forms."""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "results/grain_gptpro_foundation_v3"
PROTOCOL = ROOT / "docs/GRAIN_R01_PRODUCT_SPACE_IDENTIFICATION_PROTOCOL_V1.md"
PROTOCOL_SHA = "806a52e5460fe89ce7f3aef420a4894e0ccec8604aaab470256b44daa616e3f5"
NA, AJ = V3 / "P05/native_action_r1", V3 / "P05/application_j_diagnostic_r1"
OP = ROOT / "results/grain_vector_mechanisms/conditional_operator_v1/r2"
IDS = ROOT / "docs/GRAIN_P05_NATIVE_ACTION_PACKET_IDS_V1.json"
OUT = V3 / "P05/product_space_identification_r1"
PINS = {NA / "private_prepared.npz": "b9ce2b1c8de9365adf45a1cced3cd351dc83d24f230b663104d4aead92940338",
        NA / "fit.json": "03e164e64b4c125b86ae20f8a2b5d2e42dde07a19e4f0105b8967454ff10f0ad",
        AJ / "components_by_change.csv": "e3d0db8a0ac5737d08ac82a9946667edf0b9ec470ee2476e860837dcc6bccaf7",
        AJ / "decomposition_vs_donor.csv": "5d6e987e9ed9c6e99360f4fbc5a9fdbeba35af645a80d07393391a0676721510",
        OP / "action_arrays.npz": "778a5acdd88649e31eb8b5bd0013ddab0e50bacf4cc3e0d0bbe2325b8f9c0a9f",
        OP / "pair_order.json": "a942fd048c42ac89a241ee27929f29f37d7599e0c9c3fb114d3d935eeabaf17a",
        IDS: "4f3d38693cf8dd2024fc5a8be5a9ee2c7b4f80479bcf230d04e10a1ebe557b87"}
TREES = {NA / "private_jacobians": "35e1b6d4d305a8e0aacf0f47827e42bf99921515d0fcf295c19c2e8ea649d751",
         AJ / "private_app_jacobians": "8601b303be50ad1b0279ab4d7aad89c831c25cd6401d703a27955bc36adb60cc"}
QUERIES, OFF = ("extraversion", "neuroticism"), {"extraversion": slice(0, 7552), "neuroticism": slice(7552, 15232)}
ZMAX_SRC = {"4": 2.9412287465250984, "24": 3.1436678321842675}
RANK_TOL, ALPHA, SEED, BOOT = 1e-10, 1.0, 20260926, 2000


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tree_sha(d):
    fs = sorted(p for p in Path(d).glob("*.npy") if not p.name.startswith("._"))
    return hashlib.sha256("".join(f"{p.name}:{sha(p)}\n" for p in fs).encode()).hexdigest()


def pinv_psd(K, tol=RANK_TOL):
    w, V = np.linalg.eigh(K)
    keep = w > w.max() * tol
    return (V[:, keep] / w[keep]) @ V[:, keep].T, w, int(keep.sum()), int(((w > 0) & ~keep).sum())


def span_share(M, x):
    u, s_, _ = np.linalg.svd(M.T, full_matrices=False)
    Q = u[:, s_ > s_.max() * RANK_TOL]
    return float(((Q.T @ x) ** 2).sum() / (x @ x)) if x @ x > 0 else float("nan")


def main():
    t0 = time.time()
    if sha(PROTOCOL) != PROTOCOL_SHA:
        raise RuntimeError("protocol changed")
    for p, h in PINS.items():
        if sha(p) != h:
            raise RuntimeError(f"pin mismatch {p}")
    for d, h in TREES.items():
        if tree_sha(d) != h:
            raise RuntimeError(f"tree mismatch {d}")
    if OUT.exists():
        raise FileExistsError("product_space_identification_r1 exists; preserve earlier runs")
    OUT.mkdir(parents=True)
    arr = np.load(OP / "action_arrays.npz"); prep = np.load(NA / "private_prepared.npz")
    order = json.loads((OP / "pair_order.json").read_text()); ids = json.loads(IDS.read_text())
    src = arr["source"]; si = np.flatnonzero(src); app = ids["application_pairs"]; ai = [order.index(p) for p in app]
    fit = json.loads((NA / "fit.json").read_text())
    comp = pd.read_csv(AJ / "components_by_change.csv"); dec = pd.read_csv(AJ / "decomposition_vs_donor.csv")
    Lref = comp.set_index(["pair", "query", "change"])["L"]; Tref = dec.set_index(["pair", "query", "change"])["T"]
    rows, loo_rows, spec, betas, checks = [], [], {}, {}, {"J_recon_max_abs": {}, "E_recon_max_abs": {}, "J_recon_tol": {}, "E_recon_tol": {}}
    for q in QUERIES:
        sl = OFF[q]
        Js = np.stack([np.load(NA / "private_jacobians" / f"{i:03d}_{q}.npy").ravel() for i in si])
        Ja = np.stack([np.load(AJ / "private_app_jacobians" / f"{i:03d}_{q}.npy").ravel() for i in ai])
        s = float(Js.shape[1] / np.mean((Js ** 2).sum(1)))
        assert abs(s - fit["cells"][f"J24|{q}"]["metric_scale_s"]) <= 1e-9 * s
        Us, Ua = np.sqrt(s) * Js, np.sqrt(s) * Ja
        Y = arr["dh"][si, sl].astype(float)
        t = (Us * Y).sum(1)
        UU = Us @ Us.T; UUa = Us @ Ua.T          # (36,36), (36,72)
        jspan = np.array([span_share(Us, Ua[j]) for j in range(len(ai))])
        for inp in ("4", "24"):
            Z = prep[f"z{inp}"][si].astype(float); Za = prep[f"z{inp}"][ai].astype(float)
            ZZ = Z @ Z.T; ZZa = Z @ Za.T
            K = ZZ * UU
            Kp, w, rank, dropped_pos = pinv_psd(K)
            Kr_inv = np.linalg.inv(K + ALPHA * np.eye(len(K)))
            c = Kr_inv @ t
            spec[f"{inp}|{q}"] = {"K_rank": rank, "K_dim": len(K), "dropped_positive_modes": dropped_pos, "eig_max": float(w.max()), "eig_min": float(w.min()),
                                  "eig_min_kept": float(w[w > w.max() * RANK_TOL].min()), "condition_kept": float(w.max() / w[w > w.max() * RANK_TOL].min()),
                                  "rank_ZZ": int(np.linalg.matrix_rank(ZZ)), "rank_UU": int(np.linalg.matrix_rank(UU)), "metric_scale_s": s}
            # source LOO reference
            for i in range(len(K)):
                keep = np.arange(len(K)) != i
                Kpi, *_ = pinv_psd(K[np.ix_(keep, keep)])
                gi = K[keep, i]
                loo_rows.append({"input": inp, "query": q, "source_pair": order[si[i]], "loo_exposure": float(gi @ Kpi @ gi / K[i, i])})
            # E reconstruction pieces
            A_E = np.linalg.inv(Z.T @ Z + ALPHA * np.eye(Z.shape[1]))
            # whitening sensitivity (source-derived)
            ev, Ve = np.linalg.eigh(Z.T @ Z / len(Z)); kw = ev > ev.max() * RANK_TOL
            Wh = (Ve[:, kw] / np.sqrt(ev[kw])).T
            Zw, Zaw = Z @ Wh.T, Za @ Wh.T
            Kw = (Zw @ Zw.T) * UU; Kwp, *_ = pinv_psd(Kw)
            for k in range(len(K)):
                keep = np.arange(len(K)) != k
                Kpi, *_ = pinv_psd(Kw[np.ix_(keep, keep)]); gi = Kw[keep, k]
                loo_rows[-len(K) + k]["loo_exposure_whitened"] = float(gi @ Kpi @ gi / Kw[k, k])
            JdY = Ja @ Y.T   # (72,36): J_app . d_i
            Ljhat, Lehat = [], []
            for j, p in enumerate(app):
                zs, Us_ = Za[j], Ua[j]
                g = ZZa[:, j] * UUa[:, j]
                qn = float((zs @ zs) * (Us_ @ Us_))
                gKg = float(g @ Kp @ g)
                expo = gKg / qn if qn > 0 else float("nan")
                Lj = float(g @ c / np.sqrt(s)); Ljhat.append(Lj)
                wv = Z @ (A_E @ zs); Le = float(wv @ JdY[j]); Lehat.append(Le)
                # crossed controls
                Gcz = ZZa[:, j][:, None] * UU                   # app z with source J_k (columns k)
                qcz = (zs @ zs) * np.diag(UU)
                cz = np.einsum("ik,ij,jk->k", Gcz, Kp, Gcz) / qcz
                Gcj = ZZ * UUa[:, j][:, None]                   # source z_k with app J
                qcj = np.diag(ZZ) * (Us_ @ Us_)
                cj = np.einsum("ik,ij,jk->k", Gcj, Kp, Gcj) / qcj
                # witness (explicit, checked directly)
                beta = Kp @ g
                dW = np.outer(zs, Us_) - Z.T @ (beta[:, None] * Us)
                nW = float(np.linalg.norm(dW))
                src_act = np.einsum("ip,pd,id->i", Z, dW, Us)
                src_rel = float(np.max(np.abs(src_act) / (np.linalg.norm(Z, axis=1) * np.linalg.norm(Us, axis=1) * nW))) if nW > 0 else float("nan")
                app_act = float(zs @ dW @ Us_)
                app_rel = abs(app_act - nW ** 2) / max(nW ** 2, 1e-300)
                betas[f"{inp}|{q}|{p}"] = beta
                # frozen J map effect per unit Frobenius norm on this row
                Wf_norm = float(np.sqrt(c @ K @ c))
                gw = (Zw @ Zaw[j]) * UUa[:, j]; qw = float((Zaw[j] @ Zaw[j]) * (Us_ @ Us_))
                ch = "J" + inp
                rows.append({"input": inp, "query": q, "pair": p, "cell": ids["cells"][p], "exposure": expo, "unsupported_sq": qn - gKg, "q": qn,
                             "J_span_share": float(jspan[j]), "Z_span_share": span_share(Z, zs),
                             "CZ_mean_exposure_appz_srcJ": float(cz.mean()), "CJ_mean_exposure_srcz_appJ": float(cj.mean()),
                             "exposure_whitened": float(gw @ Kwp @ gw / qw) if qw > 0 else float("nan"),
                             "witness_norm": nW, "witness_src_rel_max": src_rel, "witness_app_rel_err": app_rel,
                             "witness_unit_native_effect": nW / np.sqrt(s), "frozenJ_native_effect_per_unit_norm": abs(Lj) / Wf_norm if Wf_norm > 0 else float("nan"),
                             "Lhat_J": Lj, "L_ref_J": float(Lref[(p, q, ch)]), "Lhat_E": Le, "L_ref_E": float(Lref[(p, q, "E" + inp)]),
                             "E_sum_w": float(wv.sum()), "E_l1_w": float(np.abs(wv).sum()), "E_l2_w": float(np.linalg.norm(wv)),
                             "E_leverage": float(zs @ A_E @ zs), "zmax": float(np.abs(zs).max()), "beyond_source_support": bool(np.abs(zs).max() > ZMAX_SRC[inp]),
                             "Jd_mean": float(JdY[j].mean()), "Jd_sd": float(JdY[j].std()), "T_E": float(Tref[(p, q, "E" + inp)]), "T_J": float(Tref[(p, q, ch)])})
            Ljhat, Lehat = np.array(Ljhat), np.array(Lehat)
            Lj_ref = np.array([Lref[(p, q, "J" + inp)] for p in app]); Le_ref = np.array([Lref[(p, q, "E" + inp)] for p in app])
            checks["J_recon_max_abs"][f"{inp}|{q}"] = float(np.abs(Ljhat - Lj_ref).max()); checks["J_recon_tol"][f"{inp}|{q}"] = float(1e-8 * np.sqrt(np.mean(Lj_ref ** 2)) + 1e-15)
            checks["E_recon_max_abs"][f"{inp}|{q}"] = float(np.abs(Lehat - Le_ref).max()); checks["E_recon_tol"][f"{inp}|{q}"] = float(1e-8 * np.sqrt(np.mean(Le_ref ** 2)) + 1e-15)
    R = pd.DataFrame(rows); LOO = pd.DataFrame(loo_rows)
    ok_J = all(checks["J_recon_max_abs"][k] <= checks["J_recon_tol"][k] for k in checks["J_recon_tol"])
    ok_E = all(checks["E_recon_max_abs"][k] <= checks["E_recon_tol"][k] for k in checks["E_recon_tol"])
    checks["reconstruction_pass"] = bool(ok_J and ok_E)
    checks["witness_src_rel_max"] = float(R.witness_src_rel_max.max()); checks["witness_app_rel_err_max"] = float(R.witness_app_rel_err.max())
    checks["witness_pass"] = bool(checks["witness_src_rel_max"] <= 1e-8 and checks["witness_app_rel_err_max"] <= 1e-8)
    rng = np.random.default_rng(SEED)
    summ = {}
    for (inp, q), g in R.groupby(["input", "query"]):
        cm = g.groupby("cell")[["exposure", "exposure_whitened", "J_span_share", "Z_span_share", "CZ_mean_exposure_appz_srcJ", "CJ_mean_exposure_srcz_appJ"]].mean()
        draws = rng.integers(0, len(cm), (BOOT, len(cm)))
        lo = LOO[(LOO.input == inp) & (LOO["query"] == q)]

        def ci(col):
            x = cm[col].to_numpy()
            return {"cell_mean": float(x.mean()), "ci": [float(v) for v in np.percentile(x[draws].mean(1), [2.5, 97.5])], "cell_median": float(np.median(x))}
        sp = lambda a, b: float(spearmanr(g[a], g[b]).statistic)
        summ[f"{inp}|{q}"] = {"application_exposure": ci("exposure"), "application_exposure_whitened": ci("exposure_whitened"),
                              "J_span_share": ci("J_span_share"), "Z_span_share": ci("Z_span_share"),
                              "crossed_appz_srcJ": ci("CZ_mean_exposure_appz_srcJ"), "crossed_srcz_appJ": ci("CJ_mean_exposure_srcz_appJ"),
                              "source_loo_exposure": {"mean": float(lo.loo_exposure.mean()), "median": float(lo.loo_exposure.median()),
                                                      "q10_q90": [float(v) for v in np.percentile(lo.loo_exposure, [10, 90])]},
                              "source_loo_exposure_whitened": {"mean": float(lo.loo_exposure_whitened.mean()), "median": float(lo.loo_exposure_whitened.median())},
                              "unit_witness_native_effect_median": float(g.witness_unit_native_effect.median()),
                              "frozenJ_native_effect_per_unit_norm_median": float(g.frozenJ_native_effect_per_unit_norm.median()),
                              "E_weights": {"sum_w_median": float(g.E_sum_w.median()), "l1_median": float(g.E_l1_w.median()), "leverage_median": float(g.E_leverage.median()),
                                            "beyond_support_pairs": int(g.beyond_source_support.sum())},
                              "spearman": {"E_l1_vs_T": sp("E_l1_w", "T_E"), "E_l1_vs_absT": float(spearmanr(g.E_l1_w, g.T_E.abs()).statistic),
                                           "E_leverage_vs_absT": float(spearmanr(g.E_leverage, g.T_E.abs()).statistic),
                                           "exposure_vs_absT_J": float(spearmanr(g.exposure, g.T_J.abs()).statistic)},
                              "absT_E_beyond_vs_within_support": [float(g[g.beyond_source_support].T_E.abs().mean()) if g.beyond_source_support.any() else None,
                                                                  float(g[~g.beyond_source_support].T_E.abs().mean())]}
    R.to_csv(OUT / "exposure_rows.csv", index=False); LOO.to_csv(OUT / "source_loo_rows.csv", index=False)
    np.savez(OUT / "witness_compact_factors.npz", **{k.replace("|", "__"): v for k, v in betas.items()})
    res = {"protocol_sha256": PROTOCOL_SHA, "neural_calls": 0, "fits": 0, "checks": checks, "spectra": spec, "summary": summ, "seconds": time.time() - t0,
           "created": datetime.now(timezone.utc).isoformat(), "note": "witness factors: dW = z* U*' - Z' diag(beta) U_source; reconstruct from cached arrays"}
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    (OUT / "runtime.json").write_text(json.dumps({"seconds": res["seconds"], "neural_calls": 0, "code_sha256": sha(Path(__file__)), "protocol_sha256": PROTOCOL_SHA,
                                                  "inputs": {str(p.relative_to(ROOT)): h for p, h in PINS.items()} | {str(d.relative_to(ROOT)): h for d, h in TREES.items()}}, indent=1) + "\n")
    print(json.dumps({"checks": checks, "spectra": spec, "seconds": res["seconds"]}, indent=1, default=float))
    if not checks["reconstruction_pass"]:
        raise SystemExit("RECONSTRUCTION FAILED (outputs kept)")


if __name__ == "__main__":
    main()
