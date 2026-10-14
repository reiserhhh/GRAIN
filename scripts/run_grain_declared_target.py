#!/usr/bin/env python3
"""GRAIN-C3-01 runner: alignment gate, keyed venue-class declared-target bundle, held checks."""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import declared_target as DT

PROTOCOL = ROOT / "docs/GRAIN_C3_01_DECLARED_TARGET_PROTOCOL_V1.json"
OUT = ROOT / "results/grain_construction_v1/C3/declared_target_r1"
CR = ROOT / "results/grain_vector_mechanisms/all_plan_closure_v1/conditional_reference_r2"
ADDR = ROOT / "results/grain_gptpro_foundation_v3/F16/event_addresses_r1/private_pandora_event_addresses.csv"
ROLES = ROOT / "results/grain_gptpro_foundation_v3/F14/pandora_venue_oom_r1/author_roles.json"
E5 = ROOT / "results/grain_vector_mechanisms/predecessor_integration_v1/material/private_arrays.npz"
HIST, FUT, K, Z_FW, B = range(0, 25), range(25, 36), 30, 3.14, 2000


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_csv(path, rows):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def corr(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and a.std() > 0 and b.std() > 0 else float("nan")


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    t0, c0 = time.time(), time.process_time()
    proto = json.loads(PROTOCOL.read_text())
    (OUT / "protocol_before_run.json").write_text(json.dumps({**proto, "protocol_sha256": sha(PROTOCOL)}, indent=1))
    units = np.load(CR / "pandora/private_units.npz")
    scores, authors_arr = units["scores"], units["authors"]
    addr = pd.read_csv(ADDR, usecols=["author_index", "unit", "condition", "timestamp_unix_seconds"]).sort_values(["author_index", "unit"]).reset_index(drop=True)
    if not (np.array_equal(addr.author_index.to_numpy(), authors_arr) and np.array_equal(addr.condition.to_numpy(), units["conditions"])):
        raise ValueError("units/address alignment failed")
    # alignment gate for C3-02
    D = np.load(CR / "private_directions.npz")["direction"]
    com = np.load(E5)["comments"].astype(np.float64)
    nrm = com / np.linalg.norm(com, axis=2, keepdims=True)
    gate_diff = float(np.max(np.abs((nrm @ D).reshape(-1, K) - scores)))
    gate = {"check": "normalize(e5 comment[a,u]) . keyed directions == units.scores[(a,u)]", "max_abs_diff": gate_diff,
            "pass": gate_diff <= 1e-6, "files": {"e5": {"path": str(E5.relative_to(ROOT)), "sha256": sha(E5)},
                                                  "units": sha(CR / "pandora/private_units.npz"), "directions": sha(CR / "private_directions.npz")},
            "consequence": "raw e5 event vectors are row-aligned with event addresses; C3-02 anonymous evidence route executable"}
    (OUT / "alignment_gate.json").write_text(json.dumps(gate, indent=1))
    del com, nrm
    roles = json.loads(ROLES.read_text())
    n_auth = authors_arr.max() + 1
    y_all = scores.reshape(n_auth, 36, K)
    venue = addr.condition.to_numpy().reshape(n_auth, 36)
    src_tr, src_va, held = (np.array(roles[k]) for k in ("source_train", "source_validation", "held"))
    if set(src_tr) & set(held) or set(src_va) & set(held):
        raise ValueError("role overlap")
    cnt = Counter(venue[src_tr][:, list(HIST)].ravel().tolist())
    top = [v for v, _ in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[:20]]
    rule = DT.context_rule(top)
    cmat = np.where(np.isin(venue, top), 1.0, -1.0)
    h = list(HIST)
    fits, info = DT.fit_moments([y_all[a][h] for a in src_tr], [cmat[a][h] for a in src_tr])
    ref_scores = [DT.score_author(fits, y_all[a][h], cmat[a][h]) for a in src_tr]
    reference = {t: {"mean": np.mean([r[t]["mean"] for r in ref_scores], 0).tolist(),
                     "sd": np.std([r[t]["mean"] for r in ref_scores], 0, ddof=1).tolist()} for t in ("level", "contrast")}
    spec = {"evidence": "pandora IPIP-keyed semantic projection of e5-large-v2 comment vectors", "facets": K,
            "directions_sha256": sha(CR / "private_directions.npz"), "windows": {"history": [0, 24], "future": [25, 35]},
            "targets": ["choice_composition", "level", "contrast"], "context": "author-chosen venue class (TOP20 source_train history venues)"}
    bundle = DT.build_bundle(spec, rule, fits, info, reference, {"source_train_authors": len(src_tr), "protocol_sha256": sha(PROTOCOL)})
    bsha = DT.save_bundle(bundle, OUT / "source_fitted_bundle.json")
    brief = {"brief_id": "C3_01_keyed_venue_class", "spec": spec, "context_rule_sha256": rule["top_sha256"],
             "top20_source_event_share": float(np.mean(np.isin(venue[src_tr][:, h], top))),
             "held_history_top_share": float(np.mean(np.isin(venue[held][:, h], top))),
             "estimator": "per-facet random-coefficient moments + Gaussian BLUP (author-specific design)",
             "fit_info": info, "bundle_sha256": bsha,
             "interpretation_scope": bundle["interpretation"], "eligible_authors": "held authors with >=1 history event"}
    (OUT / "construction_brief.json").write_text(json.dumps(brief, indent=1))

    # cold scoring of held authors (history only) + CLI replay
    recs = [{"author_key": f"held_{a}", "events": [{"event_id": f"{a}_{u}", "venue": str(venue[a][u]),
                                                  "evidence": y_all[a][u].tolist()} for u in h]} for a in held]
    recs.append({"author_key": "no_events_author", "events": []})
    warm = DT.cold_score(bundle, recs)
    inp, outp = OUT / "private_cold_input.jsonl", OUT / "private_cold_cli_output.jsonl"
    inp.write_text("".join(json.dumps(r) + "\n" for r in recs))
    subprocess.run([sys.executable, str(ROOT / "scripts/score_grain_declared_target_cold.py"), "--bundle", str(OUT / "source_fitted_bundle.json"),
                    "--sha256", bsha, "--input", str(inp), "--output", str(outp)], check=True)
    cli = [json.loads(x) for x in outp.read_text().splitlines()]
    replay_diff = max(float(np.max(np.abs(np.array(w[t]["mean"]) - np.array(c_[t]["mean"])))) for w, c_ in zip(warm[:-1], cli[:-1]) for t in ("level", "contrast"))
    rows = []
    for a, w in zip(held, warm[:-1]):
        for j in range(K):
            rows.append({"author_index": int(a), "facet": j, "support": w["support"], "level": w["level"]["mean"][j],
                         "level_var": w["level"]["var"][j], "contrast": w["contrast"]["mean"][j], "contrast_var": w["contrast"]["var"][j],
                         "m_plus": w["m_plus"]["mean"][j], "m_minus": w["m_minus"]["mean"][j],
                         "choice_composition": w["choice_composition"]["mean"]})
    write_csv(OUT / "private_cold_scores.csv", rows)

    # evidence trace: exact linear decomposition for two held authors (event ids only)
    trace = []
    for a in held[:2]:
        c = cmat[a][h]
        x = np.column_stack([np.ones(len(c)), c])
        for j in (0,):
            fit = fits[j]
            sig, s2, mu = np.asarray(fit["Sigma"]), fit["sigma2"], np.asarray(fit["mu"])
            cov = np.linalg.inv(np.linalg.inv(sig) + x.T @ x / s2)
            gain = cov @ x.T / s2
            prior_term = cov @ np.linalg.solve(sig, mu)
            contrib = DT.TARGETS["contrast"] @ gain * y_all[a][h][:, j]
            total = float(DT.TARGETS["contrast"] @ prior_term + contrib.sum())
            trace.append({"author": f"held_{a}", "facet": j, "prior_term": float(DT.TARGETS["contrast"] @ prior_term),
                          "event_contributions": [{"event_id": f"{a}_{u}", "class": int(c[i]), "weight": float((DT.TARGETS['contrast'] @ gain)[i]),
                                                   "contribution": float(contrib[i])} for i, u in enumerate(h)],
                          "sum_check_abs_diff": abs(total - warm[list(held).index(a)]["contrast"]["mean"][j])})
    (OUT / "private_evidence_trace.json").write_text(json.dumps(trace, indent=1))
    (OUT / "evidence_trace.json").write_text(json.dumps({"rule": "score = prior term + sum_i weight_i * evidence_i (exact linear BLUP)",
        "authors_traced": len(trace), "max_sum_check_abs_diff": max(t["sum_check_abs_diff"] for t in trace),
        "weights_sign_rule": "contrast weights are positive for TOP-class events and negative for other-class events",
        "detail": "private_evidence_trace.json (event ids only, no text)"}, indent=1))

    # empirical checks (evaluator): held primary, source_validation diagnostic
    rng = np.random.default_rng(20261025)
    checks, prior_var = [], {t: [float(DT.TARGETS[t] @ np.asarray(f["Sigma"]) @ DT.TARGETS[t]) for f in fits] for t in ("level", "contrast")}
    glob_mean = {s: np.array([y_all[src_tr][:, h][cmat[src_tr][:, h] == s].reshape(-1, K).mean(0)]).ravel() for s in (1.0, -1.0)}
    fut = list(FUT)
    for role, ids in (("held", held), ("source_validation", src_va)):
        sc = [DT.score_author(fits, y_all[a][h], cmat[a][h]) for a in ids]
        ev = [h[0::2], h[1::2]]
        halves = [[DT.score_author(fits, y_all[a][hh], cmat[a][hh]) for a in ids] for hh in ev]
        n = len(ids)
        idx = rng.integers(0, n, (B, n))
        for j in range(K):
            se_b, se_g, se_r, cells, cover = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
            pd_, od_, ok2 = np.zeros(n), np.zeros(n), np.zeros(n, bool)
            for i, a in enumerate(ids):
                for s, name in ((1.0, "m_plus"), (-1.0, "m_minus")):
                    fm = cmat[a][fut] == s
                    if not fm.any():
                        continue
                    obs = y_all[a][fut][fm, j].mean()
                    pred = sc[i][name]["mean"][j]
                    hm = cmat[a][h] == s
                    raw = y_all[a][h][hm, j].mean() if hm.any() else glob_mean[s][j]
                    se_b[i] += (pred - obs) ** 2
                    se_g[i] += (glob_mean[s][j] - obs) ** 2
                    se_r[i] += (raw - obs) ** 2
                    pv = sc[i][name]["var"][j] + fits[j]["sigma2"] / fm.sum()
                    cover[i] += abs(pred - obs) <= 1.959964 * np.sqrt(pv)
                    cells[i] += 1
                fp, fmn = cmat[a][fut] > 0, cmat[a][fut] < 0
                if fp.any() and fmn.any():
                    ok2[i] = True
                    od_[i] = y_all[a][fut][fp, j].mean() - y_all[a][fut][fmn, j].mean()
                    pd_[i] = sc[i]["contrast"]["mean"][j]

            def imp(ix):
                return (se_g[ix].sum() - se_b[ix].sum()) / cells[ix].sum()

            def imp_raw(ix):
                return (se_r[ix].sum() - se_b[ix].sum()) / cells[ix].sum()

            lv = [np.array([halves[k][i]["level"]["mean"][j] for i in range(n)]) for k in (0, 1)]
            cv = [np.array([halves[k][i]["contrast"]["mean"][j] for i in range(n)]) for k in (0, 1)]
            both = np.array([halves[0][i]["support"] == "observed" and halves[1][i]["support"] == "observed" for i in range(n)])
            stats = {
                "E1_improvement_vs_global": (imp(np.arange(n)), np.std([imp(ix) for ix in idx], ddof=1)),
                "E1_improvement_vs_raw_history": (imp_raw(np.arange(n)), np.std([imp_raw(ix) for ix in idx], ddof=1)),
                "E2_contrast_future_corr": (corr(pd_[ok2], od_[ok2]), np.nanstd([corr(pd_[ix][ok2[ix]], od_[ix][ok2[ix]]) for ix in idx], ddof=1)),
                "E3_level_split_half_corr": (corr(lv[0], lv[1]), np.nanstd([corr(lv[0][ix], lv[1][ix]) for ix in idx], ddof=1)),
                "E3_contrast_split_half_corr": (corr(cv[0][both], cv[1][both]), np.nanstd([corr(cv[0][ix][both[ix]], cv[1][ix][both[ix]]) for ix in idx], ddof=1)),
            }
            for name, (val, se) in stats.items():
                checks.append({"role": role, "facet": j, "statistic": name, "value": val, "boot_se": float(se),
                               "fw_lower": float(val - Z_FW * se), "lower95": float(val - 1.959964 * se)})
            mvl = 1 - np.mean([s_["level"]["var"][j] for s_ in sc]) / prior_var["level"][j]
            mvc = 1 - np.mean([s_["contrast"]["var"][j] for s_ in sc]) / prior_var["contrast"][j]
            checks += [{"role": role, "facet": j, "statistic": "model_reliability_level", "value": float(mvl)},
                       {"role": role, "facet": j, "statistic": "model_reliability_contrast", "value": float(mvc)},
                       {"role": role, "facet": j, "statistic": "E4_predictive_coverage", "value": float(cover.sum() / cells.sum()), "cells": int(cells.sum())},
                       {"role": role, "facet": j, "statistic": "sd_level_scores", "value": float(np.std([s_["level"]["mean"][j] for s_ in sc])), "prior_sd": float(np.sqrt(prior_var["level"][j]))},
                       {"role": role, "facet": j, "statistic": "sd_contrast_scores", "value": float(np.std([s_["contrast"]["mean"][j] for s_ in sc])), "prior_sd": float(np.sqrt(prior_var["contrast"][j]))},
                       {"role": role, "facet": j, "statistic": "n_contrast_future_authors", "value": int(ok2.sum())},
                       {"role": role, "facet": j, "statistic": "n_contrast_split_half_authors", "value": int(both.sum())}]
        pih = np.array([(np.sum(cmat[a][h] > 0) + 1) / (len(h) + 2) for a in ids])
        pif = np.array([np.mean(cmat[a][fut] > 0) for a in ids])
        checks.append({"role": role, "facet": -1, "statistic": "E5_choice_composition_history_future_corr", "value": corr(pih, pif),
                       "boot_se": float(np.std([corr(pih[ix], pif[ix]) for ix in idx], ddof=1))})
        checks.append({"role": role, "facet": -1, "statistic": "share_model_borrowed_contrast",
                       "value": float(np.mean([s_["support"] != "observed" for s_ in sc]))})
    write_csv(OUT / "empirical_checks.csv", checks)

    # frozen acceptance
    ck = pd.DataFrame(checks)
    hd = ck[ck.role == "held"]
    from scipy.stats import binom
    facet_rows = []
    for j in range(K):
        g = hd[hd.facet == j].set_index("statistic")
        lvl = bool(g.loc["E1_improvement_vs_global", "fw_lower"] > 0 and g.loc["E3_level_split_half_corr", "fw_lower"] > 0)
        con = bool(g.loc["E2_contrast_future_corr", "fw_lower"] > 0 and g.loc["E3_contrast_split_half_corr", "fw_lower"] > 0)
        nc_l = bool(g.loc["sd_level_scores", "value"] >= .1 * g.loc["sd_level_scores", "prior_sd"])
        nc_c = bool(g.loc["sd_contrast_scores", "value"] >= .1 * g.loc["sd_contrast_scores", "prior_sd"])
        cells = int(g.loc["E4_predictive_coverage", "cells"])
        cov_lo = binom.ppf(.025, cells, .95) / cells
        facet_rows.append({"facet": j, "level_supported": lvl and nc_l, "contrast_supported": con and nc_c,
                           "calibrated": bool(g.loc["E4_predictive_coverage", "value"] >= cov_lo),
                           "coverage": float(g.loc["E4_predictive_coverage", "value"]), "coverage_lower_bound": float(cov_lo)})
    fr = pd.DataFrame(facet_rows)
    acc = {"cold_replay_max_abs_diff": replay_diff, "cold_replay_ok": replay_diff <= 1e-12,
           "no_events_author_support": cli[-1]["support"],
           "facets_level_supported": fr.facet[fr.level_supported].tolist(), "facets_contrast_supported": fr.facet[fr.contrast_supported].tolist(),
           "facets_calibrated": int(fr.calibrated.sum()), "alignment_gate_pass": gate["pass"]}
    acc["positive_construction"] = bool(acc["cold_replay_ok"] and (len(acc["facets_level_supported"]) + len(acc["facets_contrast_supported"]) > 0))
    acc["target_types_passing"] = [t for t, key in (("level", "facets_level_supported"), ("contrast", "facets_contrast_supported")) if acc[key]]
    write_csv(OUT / "acceptance_by_facet.csv", facet_rows)
    (OUT / "acceptance.json").write_text(json.dumps(acc, indent=1, default=float))
    run = {"total_wall_s": time.time() - t0, "total_cpu_s": time.process_time() - c0, "python": sys.version, "numpy": np.__version__,
           "platform": platform.platform(), "code_sha256": {p: sha(ROOT / p) for p in (
               "grain_research/construction/declared_target.py", "scripts/run_grain_declared_target.py", "scripts/score_grain_declared_target_cold.py")}}
    (OUT / "run_log.json").write_text(json.dumps(run, indent=1))
    print(json.dumps({**acc, "cpu_s": run["total_cpu_s"]}, indent=1, default=float))


if __name__ == "__main__":
    main()
