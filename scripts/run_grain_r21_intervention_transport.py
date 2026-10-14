#!/usr/bin/env python
"""R21/R23 level vs intervention-signature transport (docs/GRAIN_R21_INTERVENTION_TRANSPORT_PROTOCOL_V1.md).

Cached numeric reader scores only; zero neural calls.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/GRAIN_R21_INTERVENTION_TRANSPORT_PROTOCOL_V1.md"
PROTOCOL_SHA = "e47025d32a6052e119aab5f80ed7ca119be1680eab498116a4d7c7199aa4c78f"
CF = ROOT / "results/grain_gptpro_foundation_v3/R14/counterfactual_reader_r1"
OUT = ROOT / "results/grain_gptpro_foundation_v3/F12/intervention_transport_r1"
CONDS = ("orig", "neu", "pos", "neg", "oth")
ARMS = ("pos", "neg", "oth")
SEED, DRAWS, NPERM = 20260926, 2000, 200
RANK_TOL = 1e-12


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_verified() -> tuple[pd.DataFrame, dict]:
    rows = [json.loads(x) for x in (CF / "results.jsonl").read_text().splitlines()]
    acc = {}
    for x in (CF / "accounting.jsonl").read_text().splitlines():
        e = json.loads(x)
        if e["status"] == "OK":
            acc[e["item"]] = e
    chk = json.loads((CF / "input_checkpoint.json").read_text())
    chk_rows = {r["key"]: r for r in chk["rows"]}
    problems = []
    if len(rows) != 40 or chk["status"] != "PASS":
        problems.append("unit count or input checkpoint status")
    out = []
    for r in rows:
        k = r["key"]
        if k not in chk_rows or chk_rows[k]["corpus"] != r["corpus"] or chk_rows[k]["set"] != r["set"]:
            problems.append(f"key/corpus/set mismatch {k}")
        rec = {"key": k, "corpus": r["corpus"], "set": r["set"], "text_sha256": chk_rows.get(k, {}).get("text_sha256")}
        for c in CONDS:
            q, n = r.get(f"qwen_{c}"), r.get(f"nli_{c}")
            if q is None or n is None or not np.isfinite(q) or not np.isfinite(n):
                problems.append(f"missing {k} {c}")
            if acc.get(f"qwen|{k}|{c}", {}).get("score") != q or acc.get(f"nli|{k}|{c}", {}).get("score") != n:
                problems.append(f"accounting mismatch {k} {c}")
            rec[f"Q_{c}"], rec[f"N_{c}"] = q, n
        out.append(rec)
    df = pd.DataFrame(out)
    if df.key.duplicated().any():
        problems.append("duplicate unit keys")
    manifest = {"results_sha256": sha(CF / "results.jsonl"), "accounting_sha256": sha(CF / "accounting.jsonl"),
                "input_checkpoint_sha256": sha(CF / "input_checkpoint.json"), "units": len(df),
                "per_corpus": df.corpus.value_counts().to_dict(), "problems": problems,
                "status": "PASS" if not problems else "STOP_INPUT", "units_native": {"Q": "Qwen digit expectation [0,1]", "N": "NLI p_ent - p_contra [-1,1]"}}
    return df, manifest


def fit_map(src: pd.DataFrame) -> dict:
    n = np.concatenate([src.N_orig.to_numpy(), src.N_neu.to_numpy()])
    q = np.concatenate([src.Q_orig.to_numpy(), src.Q_neu.to_numpy()])
    var = float(np.var(n))
    if var < RANK_TOL:
        return {"a": float(q.mean()), "b": 0.0, "slope_identified": False, "source_N_var": var, "n_points": len(n)}
    b = float(np.cov(n, q, bias=True)[0, 1] / var)
    return {"a": float(q.mean() - b * n.mean()), "b": b, "slope_identified": True, "source_N_var": var, "n_points": len(n)}


def level_stats(app: pd.DataFrame, m: dict, which=("orig", "neu")) -> dict:
    Q = np.concatenate([app[f"Q_{c}"].to_numpy() for c in which])
    M = np.concatenate([m["a"] + m["b"] * app[f"N_{c}"].to_numpy() for c in which])
    d = M - Q
    return {"bias": float(d.mean()), "rmse": float(np.sqrt((d ** 2).mean())),
            "pearson": float(np.corrcoef(M, Q)[0, 1]) if np.std(M) > 0 else float("nan"),
            "spearman": float(stats.spearmanr(M, Q).statistic) if np.std(M) > 0 else float("nan"),
            "var_ratio": float(np.var(M) / np.var(Q)), "mapped_out_of_01": int(((M < 0) | (M > 1)).sum())}


def signatures(app: pd.DataFrame, m: dict) -> tuple[np.ndarray, np.ndarray]:
    dQ = np.column_stack([app[f"Q_{c}"] - app.Q_neu for c in ARMS])
    dM = np.column_stack([m["b"] * (app[f"N_{c}"] - app.N_neu) for c in ARMS])
    return dQ, dM


def sig_stats(dQ, dM) -> dict:
    mis = dM - dQ
    out = {}
    for j, c in enumerate(ARMS):
        out[f"{c}_mean_dQ"] = float(dQ[:, j].mean())
        out[f"{c}_mean_dmap"] = float(dM[:, j].mean())
        out[f"{c}_mean_mismatch"] = float(mis[:, j].mean())
        out[f"{c}_mismatch_rmse"] = float(np.sqrt((mis[:, j] ** 2).mean()))
        out[f"{c}_sign_agreement"] = float((np.sign(dM[:, j]) == np.sign(dQ[:, j])).mean())
    out["population_signature_discrepancy_norm"] = float(np.linalg.norm(mis.mean(axis=0)))
    out["author_signature_discrepancy_mean_norm"] = float(np.linalg.norm(mis, axis=1).mean())
    own_q = dQ[:, 2] - dQ[:, 0]
    own_m = dM[:, 2] - dM[:, 0]
    out["ownership_Q"] = float(own_q.mean())
    out["ownership_map"] = float(own_m.mean())
    out["ownership_mismatch"] = float((own_m - own_q).mean())
    out["S_matched"] = float((np.linalg.norm(mis, axis=1) ** 2).mean())
    cq, cm = dQ - dQ.mean(axis=0), dM - dM.mean(axis=0)
    out["centered_config_corr"] = float(np.corrcoef(cq.ravel(), cm.ravel())[0, 1]) if cm.std() > 0 else float("nan")
    return out


def run_direction(df: pd.DataFrame, src_corpus: str, app_corpus: str, rng: np.random.Generator) -> dict:
    src = df[df.corpus == src_corpus].reset_index(drop=True)
    app = df[df.corpus == app_corpus].reset_index(drop=True)
    assert not set(src.key) & set(app.key)
    m = fit_map(src)
    lv = {w: level_stats(app, m, ws) for w, ws in (("orig", ("orig",)), ("neu", ("neu",)), ("both", ("orig", "neu")))}
    dQ, dM = signatures(app, m)
    sg = sig_stats(dQ, dM)
    # algebraic checks
    mapped = {c: m["a"] + m["b"] * app[f"N_{c}"].to_numpy() for c in CONDS}
    alg = float(max(np.abs((mapped[c] - mapped["neu"]) - dM[:, j]).max() for j, c in enumerate(ARMS)))
    # per-unit rows
    units = []
    for i, r in app.iterrows():
        row = {"direction": f"{src_corpus}->{app_corpus}", "key": r.key}
        for c in CONDS:
            row[f"Q_{c}"], row[f"N_{c}"], row[f"M_{c}"] = r[f"Q_{c}"], r[f"N_{c}"], mapped[c][i]
        for j, c in enumerate(ARMS):
            row[f"dQ_{c}"], row[f"dmap_{c}"] = dQ[i, j], dM[i, j]
        units.append(row)
    n_app = len(app)
    # primary bootstrap (conditional on fitted map)
    boot = {"level_both_rmse": [], "level_both_bias": [], "level_both_pearson": []}
    for k in sg:
        boot[k] = []
    for _ in range(DRAWS):
        ix = rng.integers(0, n_app, n_app)
        b_app = app.iloc[ix].reset_index(drop=True)
        l = level_stats(b_app, m)
        boot["level_both_rmse"].append(l["rmse"]); boot["level_both_bias"].append(l["bias"]); boot["level_both_pearson"].append(l["pearson"])
        s = sig_stats(dQ[ix], dM[ix])
        for k in sg:
            boot[k].append(s[k])
    ci = {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in boot.items()}
    # fit-uncertainty companion
    n_src = len(src)
    comp = {"level_both_rmse": [], "population_signature_discrepancy_norm": [], "b": []}
    for k in ("pos_mean_mismatch", "neg_mean_mismatch", "oth_mean_mismatch", "ownership_mismatch"):
        comp[k] = []
    singular = 0
    for _ in range(DRAWS):
        sx = rng.integers(0, n_src, n_src)
        ax = rng.integers(0, n_app, n_app)
        mm = fit_map(src.iloc[sx].reset_index(drop=True))
        singular += int(not mm["slope_identified"])
        b_app = app.iloc[ax].reset_index(drop=True)
        q2, m2 = signatures(b_app, mm)
        s = sig_stats(q2, m2)
        comp["level_both_rmse"].append(level_stats(b_app, mm)["rmse"])
        comp["population_signature_discrepancy_norm"].append(s["population_signature_discrepancy_norm"])
        comp["b"].append(mm["b"])
        for k in ("pos_mean_mismatch", "neg_mean_mismatch", "oth_mean_mismatch", "ownership_mismatch"):
            comp[k].append(s[k])
    comp_ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in comp.items()}
    # null 1: source-author block permutation before fitting
    n1 = {"level_both_rmse": [], "level_both_pearson": [], "population_signature_discrepancy_norm": [], "b": []}
    for _ in range(NPERM):
        perm = rng.permutation(n_src)
        s_perm = src.copy()
        s_perm[["N_orig", "N_neu"]] = src[["N_orig", "N_neu"]].to_numpy()[perm]
        mm = fit_map(s_perm)
        q2, m2 = signatures(app, mm)
        n1["level_both_rmse"].append(level_stats(app, mm)["rmse"])
        n1["level_both_pearson"].append(level_stats(app, mm)["pearson"])
        n1["population_signature_discrepancy_norm"].append(sig_stats(q2, m2)["population_signature_discrepancy_norm"])
        n1["b"].append(mm["b"])
    # null 2: application whole-signature shuffles after the correct fit
    n2 = {"S": [], "centered_config_corr": [], "arm_means_preserved_max_dev": 0.0}
    for _ in range(NPERM):
        perm = rng.permutation(n_app)
        dMp = dM[perm]
        n2["arm_means_preserved_max_dev"] = max(n2["arm_means_preserved_max_dev"], float(np.abs(dMp.mean(0) - dM.mean(0)).max()))
        s = sig_stats(dQ, dMp)
        n2["S"].append(s["S_matched"])
        n2["centered_config_corr"].append(s["centered_config_corr"])
    def summ(v):
        v = np.asarray(v, float)
        return {"q05": float(np.nanpercentile(v, 5)), "median": float(np.nanmedian(v)), "q95": float(np.nanpercentile(v, 95))}
    null1 = {k: summ(v) for k, v in n1.items()}
    null2 = {"S": summ(n2["S"]), "centered_config_corr": summ(n2["centered_config_corr"]),
             "share_perm_S_le_matched": float(np.mean(np.asarray(n2["S"]) <= sg["S_matched"])),
             "share_perm_corr_ge_matched": float(np.mean(np.asarray(n2["centered_config_corr"]) >= sg["centered_config_corr"])),
             "arm_means_preserved_max_dev": n2["arm_means_preserved_max_dev"]}
    null1["share_null_rmse_le_actual"] = float(np.mean(np.asarray(n1["level_both_rmse"]) <= lv["both"]["rmse"]))
    null1["share_null_sigdisc_le_actual"] = float(np.mean(np.asarray(n1["population_signature_discrepancy_norm"]) <= sg["population_signature_discrepancy_norm"]))
    return {"direction": f"{src_corpus}->{app_corpus}", "map": m, "level": lv, "signature": sg, "primary_ci": ci,
            "fit_uncertainty_ci": comp_ci, "fit_uncertainty_singular_draws": singular, "null1_source_block_permutation": null1,
            "null2_application_signature_shuffle": null2, "algebra_max_abs_dev_dmap": alg,
            "roles_disjoint": not set(src.key) & set(app.key), "n_source": n_src, "n_application": n_app}, units


def main() -> int:
    t0 = time.time()
    if sha(PROTOCOL) != PROTOCOL_SHA:
        raise RuntimeError("protocol sha mismatch")
    OUT.mkdir(parents=True, exist_ok=True)
    df, man = load_verified()
    man["protocol_sha256"] = PROTOCOL_SHA
    (OUT / "input_manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    if man["status"] != "PASS":
        print(json.dumps(man, indent=1))
        return 2
    rng = np.random.default_rng(SEED)
    results, units = [], []
    for s, a in (("pandora", "essays"), ("essays", "pandora")):
        r, u = run_direction(df, s, a, rng)
        results.append(r)
        units += u
    pd.DataFrame(units).to_csv(OUT / "per_unit_levels_signatures.csv", index=False)
    (OUT / "source_maps.json").write_text(json.dumps({r["direction"]: r["map"] for r in results}, indent=1) + "\n")
    (OUT / "results.json").write_text(json.dumps(results, indent=1) + "\n")
    checks = {r["direction"]: {"dmap_equals_b_dN_max_dev": r["algebra_max_abs_dev_dmap"], "null2_arm_means_preserved_max_dev": r["null2_application_signature_shuffle"]["arm_means_preserved_max_dev"],
                               "roles_disjoint": r["roles_disjoint"], "slope_identified": r["map"]["slope_identified"]} for r in results}
    ok = all(v["dmap_equals_b_dN_max_dev"] <= 1e-12 and v["null2_arm_means_preserved_max_dev"] <= 1e-12 and v["roles_disjoint"] for v in checks.values())
    (OUT / "algebra_checks.json").write_text(json.dumps({"checks": checks, "status": "PASS" if ok else "FAIL"}, indent=1) + "\n")
    (OUT / "runtime.json").write_text(json.dumps({"seconds": time.time() - t0, "neural_calls": 0,
                                                  "code_sha256": sha(Path(__file__))}, indent=1) + "\n")
    print(json.dumps({"algebra": ok, "seconds": time.time() - t0}))
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
