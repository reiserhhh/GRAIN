#!/usr/bin/env python3
"""GRAIN-C4-01 runner: anonymous cold bundle (Part A) + source-only lexical reader transport (Part B).

Text bodies are read locally in memory only; nothing textual is printed or written.
"""

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
from scipy.stats import binom
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import anonymous_bundle as AB
from grain_research.construction import declared_target as DT

PROTOCOL = ROOT / "docs/GRAIN_C4_01_TRANSPORT_PROTOCOL_V1.json"
OUT = ROOT / "results/grain_construction_v1/C4/transport_r1"
CR = ROOT / "results/grain_vector_mechanisms/all_plan_closure_v1/conditional_reference_r2"
ADDR = ROOT / "results/grain_gptpro_foundation_v3/F16/event_addresses_r1/private_pandora_event_addresses.csv"
ROLES = ROOT / "results/grain_gptpro_foundation_v3/F14/pandora_venue_oom_r1/author_roles.json"
MAT = ROOT / "results/grain_vector_mechanisms/predecessor_integration_v1/material"
DIRS = ROOT / "results/grain_construction_v1/C3/anonymous_evidence_r1/private_learned_directions.npz"
C3 = ROOT / "results/grain_construction_v1/C3"
H, F, B = list(range(25)), list(range(25, 36)), 2000
FAMS = ("author_stability", "response_stability", "principal_components")


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


def scores(fits, ev, cmat, ids, units):
    return [DT.score_author(fits, ev[a][units], cmat[a][units]) for a in ids]


SUPP = "--uncertainty-layer" in sys.argv


def main():
    global OUT, PROTOCOL
    if SUPP:
        OUT = ROOT / "results/grain_construction_v1/C4/transport_r1_uncertainty_layer"
        PROTOCOL = ROOT / "docs/GRAIN_C4_01_TRANSPORT_PROTOCOL_V1_1_UNCERTAINTY_LAYER.json"
    OUT.mkdir(parents=True, exist_ok=False)
    t0, c0 = time.time(), time.process_time()
    proto = json.loads(PROTOCOL.read_text())
    (OUT / "protocol_before_run.json").write_text(json.dumps({**proto, "protocol_sha256": sha(PROTOCOL)}, indent=1))
    roles = json.loads(ROLES.read_text())
    src_tr, src_va, held = (np.array(roles[k]) for k in ("source_train", "source_validation", "held"))
    addr = pd.read_csv(ADDR, usecols=["author_index", "unit", "start", "end", "condition", "body_sha256"]).sort_values(["author_index", "unit"]).reset_index(drop=True)
    n_auth = int(addr.author_index.max()) + 1
    venue = addr.condition.to_numpy().reshape(n_auth, 36)
    # ---------------- provenance gate (in memory; no text output)
    docs = json.loads((MAT / "private_material.json").read_text())["documents"]
    texts = [docs[r.author_index][r.start:r.end] for r in addr.itertuples()]
    match = sum(hashlib.sha256(t.encode()).hexdigest() == h for t, h in zip(texts, addr.body_sha256))
    del docs
    gate = {"events": len(texts), "body_sha256_matches": int(match), "pass": match == len(texts),
            "material_sha256": sha(MAT / "private_material.json"), "text_exported": False}
    (OUT / "provenance_gate.json").write_text(json.dumps(gate, indent=1))
    if not gate["pass"]:
        raise ValueError("provenance gate failed")
    # ---------------- Part A: anonymous cold bundle
    dz = np.load(DIRS)
    D = np.column_stack([dz[f] for f in FAMS])
    fam_labels = [f"{f}_{j}" for f in FAMS for j in range(5)]
    com = np.load(MAT / "private_arrays.npz")["comments"].astype(np.float64)
    nrm = AB.normalize(com)
    del com
    raw = nrm @ D                                        # n_auth x 36 x 15
    cnt = Counter(venue[src_tr][:, H].ravel().tolist())
    top = [v for v, _ in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[:20]]
    rule = DT.context_rule(top)
    cmat = np.where(np.isin(venue, top), 1.0, -1.0)
    table, pooled = AB.venue_offsets(raw[src_tr][:, H].reshape(-1, 15), venue[src_tr][:, H].ravel())
    ev_mode = {m: np.stack([AB.apply_mode(raw[a], venue[a], table, pooled, m)[0] for a in range(n_auth)]) for m in AB.MODES}
    fits, ref = {}, {}
    for m in AB.MODES:
        fits[m], _ = DT.fit_moments([ev_mode[m][a][H] for a in src_tr], [cmat[a][H] for a in src_tr])
        sc = scores(fits[m], ev_mode[m], cmat, src_tr, H)
        ref[m] = {t: {"mean": np.mean([s[t]["mean"] for s in sc], 0).tolist(), "sd": np.std([s[t]["mean"] for s in sc], 0, ddof=1).tolist()} for t in ("level", "contrast")}
    mm = json.loads((MAT / "material_manifest.json").read_text())
    encoder = {"id": "intfloat/e5-large-v2@f169b11e22de13617baa190a028a32f3493550b6", "prefix": "query: ", "pooling": "mean",
               "manifest_model_field": mm.get("model")}
    bundle = AB.build_bundle(D, fam_labels, encoder, rule, table, pooled, fits, ref,
                             {"directions_file_sha256": sha(DIRS), "c3_02_protocol": "ab306519", "source_train_authors": len(src_tr)})
    bsha = AB.save_bundle(bundle, OUT / "anonymous_bundle.json")
    recs = [{"author_key": f"held_{a}", "events": [{"event_id": f"{a}_{u}", "venue": str(venue[a][u]), "encoder_id": encoder["id"],
                                                  "vector": nrm[a][u]} for u in H]} for a in held]
    warm = AB.cold_score(bundle, recs)
    direct_diff = 0.0
    for a, w in zip(held, warm):
        for m in AB.MODES:
            s = DT.score_author(fits[m], ev_mode[m][a][H], cmat[a][H])
            direct_diff = max(direct_diff, float(np.max(np.abs(np.array(s["level"]["mean"]) - np.array(w[m]["level"]["mean"])))))
    c3_diff = 0.0
    for m, root in (("composition_inclusive", "anonymous_evidence_r1"), ("within_venue", "anonymous_evidence_r1_within_venue")):
        ck = pd.read_csv(C3 / root / "empirical_checks.csv")
        halves = [[AB.cold_score(bundle, [{"author_key": "x", "events": [{"event_id": f"{a}_{u}", "venue": str(venue[a][u]), "encoder_id": encoder["id"], "vector": nrm[a][u]} for u in hh]}])[0][m] for a in held] for hh in (H[0::2], H[1::2])]
        for k, lab in enumerate(fam_labels):
            fam, j = lab.rsplit("_", 1)
            r = corr([h["level"]["mean"][k] for h in halves[0]], [h["level"]["mean"][k] for h in halves[1]])
            saved = ck[(ck.family == fam) & (ck.direction == int(j)) & (ck.role == "held") & (ck.statistic == "E3_level_split_half_corr")].value.iloc[0]
            c3_diff = max(c3_diff, abs(r - saved))
    sub = recs[:20]
    inp, outp = OUT / "private_cli_input.jsonl", OUT / "private_cli_output.jsonl"
    with inp.open("w") as fh:
        for r in sub + [{"author_key": "no_events_author", "events": []}]:
            fh.write(json.dumps({"author_key": r["author_key"], "events": [{**e, "vector": np.asarray(e["vector"]).tolist()} for e in r["events"]]}) + "\n")
    subprocess.run([sys.executable, str(ROOT / "scripts/score_grain_anonymous_bundle_cold.py"), "--bundle", str(OUT / "anonymous_bundle.json"),
                    "--sha256", bsha, "--input", str(inp), "--output", str(outp)], check=True)
    cli = [json.loads(x) for x in outp.read_text().splitlines()]
    cli_diff = max(float(np.max(np.abs(np.array(c[m][t]["mean"]) - np.array(w[m][t]["mean"])))) for c, w in zip(cli[:20], warm[:20]) for m in AB.MODES for t in ("level", "contrast"))
    replay = {"bundle_sha256": bsha, "in_process_vs_direct_max_abs_diff": direct_diff, "c3_02_split_half_reproduction_max_abs_diff": c3_diff,
              "cli_vs_in_process_max_abs_diff": cli_diff, "no_events_support": cli[-1]["composition_inclusive"]["support"],
              "unseen_venue_share_held_history_mean": float(np.mean([w["within_venue"]["unseen_venue_share"] for w in warm])),
              "exact": bool(direct_diff <= 1e-12 and c3_diff <= 1e-12 and cli_diff <= 1e-12)}
    (OUT / "bundle_replay.json").write_text(json.dumps(replay, indent=1))
    # ---------------- Part B: lexical reader transport
    rows_idx = np.arange(n_auth * 36).reshape(n_auth, 36)
    src_rows = rows_idx[src_tr][:, H].ravel()
    vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=5, max_features=20000, sublinear_tf=True, norm="l2")
    vec.fit([texts[i] for i in src_rows])
    X = vec.transform(texts)
    vocab_size = len(vec.vocabulary_)
    del texts, vec
    keyed = np.load(CR / "pandora/private_units.npz")["scores"].reshape(n_auth, 36, 30)
    Y = np.concatenate([raw.reshape(-1, 15), keyed.reshape(-1, 30)], axis=1)
    ridge = Ridge(alpha=1.0).fit(X[src_rows], Y[src_rows])
    Yhat = ridge.predict(X).reshape(n_auth, 36, 45)
    sets = {("anon", "composition_inclusive"): (slice(0, 15), "composition_inclusive"),
            ("anon", "within_venue"): (slice(0, 15), "within_venue"),
            ("keyed", "composition_inclusive"): (slice(15, 45), "composition_inclusive")}
    rng = np.random.default_rng(20261031)
    va_rows_auth = [(a, H) for a in src_va]
    map_rows, held_rows, dec_rows = [], [], []
    for (unit, mode), (sl, _) in sets.items():
        k = sl.stop - sl.start
        e_raw, m_raw = np.concatenate([raw, keyed], 2)[:, :, sl], Yhat[:, :, sl]
        if mode == "within_venue":
            e_tab, e_pool = table, pooled
            E = np.stack([AB.apply_mode(e_raw[a], venue[a], e_tab, e_pool, mode)[0] for a in range(n_auth)])
            M = np.stack([AB.apply_mode(m_raw[a], venue[a], e_tab, e_pool, mode)[0] for a in range(n_auth)])
            l_tab, l_pool = AB.venue_offsets(m_raw[src_tr][:, H].reshape(-1, k), venue[src_tr][:, H].ravel())
            MR = np.stack([AB.apply_mode(m_raw[a], venue[a], l_tab, l_pool, mode)[0] for a in range(n_auth)])
        else:
            E, M, MR = e_raw, m_raw, m_raw
        # source_validation calibration per coordinate (author-cluster bootstrap of OLS y = a + b yhat)
        yv = E[src_va][:, H].reshape(len(src_va), -1, k)
        xv = M[src_va][:, H].reshape(len(src_va), -1, k)
        sx, sy, sxx, sxy, nn = xv.sum(1), yv.sum(1), (xv ** 2).sum(1), (xv * yv).sum(1), yv.shape[1]

        def ols(ix):
            Sx, Sy, Sxx, Sxy, N = sx[ix].sum(0), sy[ix].sum(0), sxx[ix].sum(0), sxy[ix].sum(0), nn * len(ix)
            b = (Sxy - Sx * Sy / N) / (Sxx - Sx ** 2 / N)
            return Sy / N - b * Sx / N, b

        a0, b0 = ols(np.arange(len(src_va)))
        boots = [ols(rng.integers(0, len(src_va), len(src_va))) for _ in range(1000)]
        ab_ = np.array([np.r_[x[0], x[1]] for x in boots])
        a_lo, a_hi = np.percentile(ab_[:, :k], [2.5, 97.5], axis=0)
        b_lo, b_hi = np.percentile(ab_[:, k:], [2.5, 97.5], axis=0)
        sdy = yv.reshape(-1, k).std(0)
        r2 = 1 - ((yv - xv) ** 2).reshape(-1, k).sum(0) / ((yv - yv.reshape(-1, k).mean(0)) ** 2).reshape(-1, k).sum(0)
        status = np.where((b_lo >= .8) & (b_hi <= 1.25) & (a_lo >= -.1 * sdy) & (a_hi <= .1 * sdy), "DIRECT",
                          np.where((b_lo > 0) | (b_hi < 0), "CALIBRATED", "REFUSE"))
        MA = a0 + b0 * M
        fits_e = fits[mode] if unit == "anon" else DT.fit_moments([E[a][H] for a in src_tr], [cmat[a][H] for a in src_tr])[0]
        fits_r = DT.fit_moments([MR[a][H] for a in src_tr], [cmat[a][H] for a in src_tr])[0]
        arms = {"e5_native": (E, fits_e), "lex_frozen": (M, fits_e), "lex_affine": (MA, fits_e), "lex_refit": (MR, fits_r)}
        tau2 = {}
        if SUPP:
            for base in ("lex_frozen", "lex_refit"):
                ev_b, f_b = arms[base]
                sc_v = scores(f_b, ev_b, cmat, src_va, H)
                t2 = np.zeros(k)
                for j in range(k):
                    acc_ = []
                    for i, a in enumerate(src_va):
                        for s, nm in ((1.0, "m_plus"), (-1.0, "m_minus")):
                            fm = cmat[a][F] == s
                            if fm.any():
                                res = E[a][F][fm, j].mean() - sc_v[i][nm]["mean"][j]
                                acc_.append(res ** 2 - sc_v[i][nm]["var"][j] - fits_e[j]["sigma2"] / fm.sum())
                    t2[j] = max(0.0, float(np.mean(acc_)))
                tau2[base] = t2
                arms[base + "_readerdiff"] = (ev_b, f_b)
        for j in range(k):
            map_rows.append({"unit": unit, "mode": mode, "coordinate": j, "label": fam_labels[j] if unit == "anon" else f"keyed_{j}",
                             "validation_r2": float(r2[j]), "a": float(a0[j]), "a_lo": float(a_lo[j]), "a_hi": float(a_hi[j]),
                             "b": float(b0[j]), "b_lo": float(b_lo[j]), "b_hi": float(b_hi[j]), "sd_y": float(sdy[j]), "status": str(status[j])})
        glob_e = {s: E[src_tr][:, H][cmat[src_tr][:, H] == s].reshape(-1, k).mean(0) for s in (1.0, -1.0)}
        n = len(held)
        idx = rng.integers(0, n, (B, n))
        sc_full = {arm: scores(f_, ev_, cmat, held, H) for arm, (ev_, f_) in arms.items()}
        sc_half = {arm: [scores(f_, ev_, cmat, held, hh) for hh in (H[0::2], H[1::2])] for arm, (ev_, f_) in arms.items()}
        for arm, (ev_, f_) in arms.items():
            glob_arm = {s: ev_[src_tr][:, H][cmat[src_tr][:, H] == s].reshape(-1, k).mean(0) for s in (1.0, -1.0)}
            for j in range(k):
                lev = np.array([s["level"]["mean"][j] for s in sc_full[arm]])
                lev_e = np.array([s["level"]["mean"][j] for s in sc_full["e5_native"]])
                con = np.array([s["contrast"]["mean"][j] for s in sc_full[arm]])
                con_e = np.array([s["contrast"]["mean"][j] for s in sc_full["e5_native"]])
                h0 = np.array([s["level"]["mean"][j] for s in sc_half[arm][0]])
                h1 = np.array([s["level"]["mean"][j] for s in sc_half[arm][1]])
                e0 = np.array([s["level"]["mean"][j] for s in sc_half["e5_native"][0]])
                e1 = np.array([s["level"]["mean"][j] for s in sc_half["e5_native"][1]])
                sb_a, sg_a, sb_b, sg_b, sb_c, sg_c = (np.zeros(n) for _ in range(6))
                cov, cells = np.zeros(n), np.zeros(n)
                for i, a in enumerate(held):
                    for s, nm in ((1.0, "m_plus"), (-1.0, "m_minus")):
                        fm = cmat[a][F] == s
                        if not fm.any():
                            continue
                        obs_e, obs_m = E[a][F][fm, j].mean(), ev_[a][F][fm, j].mean()
                        p_arm, p_e = sc_full[arm][i][nm]["mean"][j], sc_full["e5_native"][i][nm]["mean"][j]
                        sb_a[i] += (p_arm - obs_e) ** 2
                        sg_a[i] += (glob_e[s][j] - obs_e) ** 2
                        sb_b[i] += (p_e - obs_m) ** 2
                        sg_b[i] += (glob_arm[s][j] - obs_m) ** 2
                        sb_c[i] += (p_arm - obs_m) ** 2
                        sg_c[i] += (glob_arm[s][j] - obs_m) ** 2
                        if arm.endswith("_readerdiff"):
                            pv = sc_full[arm][i][nm]["var"][j] + fits_e[j]["sigma2"] / fm.sum() + tau2[arm[:-11]][j]
                        else:
                            pv = sc_full[arm][i][nm]["var"][j] + f_[j]["sigma2"] / fm.sum()
                        cov[i] += abs(p_arm - obs_e) <= 1.959964 * np.sqrt(pv)
                        cells[i] += 1
                rel = lambda ix, sb, sg: (sg[ix].sum() - sb[ix].sum()) / sg[ix].sum()  # noqa: E731
                rel_a_b = np.array([rel(ix, sb_a, sg_a) for ix in idx])
                sb_full = lambda r: 2 * r / (1 + r)  # noqa: E731
                agree_b = np.array([corr(lev[ix], lev_e[ix]) for ix in idx])
                rl_b = np.array([sb_full(corr(h0[ix], h1[ix])) for ix in idx])
                re_b = np.array([sb_full(corr(e0[ix], e1[ix])) for ix in idx])
                dis_b = agree_b / np.sqrt(np.clip(rl_b * re_b, 1e-9, None))
                cells_tot = int(cells.sum())
                held_rows.append({"unit": unit, "mode": mode, "coordinate": j, "arm": arm,
                                  "agreement_level_vs_e5": corr(lev, lev_e), "agreement_contrast_vs_e5": corr(con, con_e),
                                  "split_half_level": corr(h0, h1), "split_half_level_e5": corr(e0, e1),
                                  "disattenuated_level_agreement": float(corr(lev, lev_e) / np.sqrt(max(sb_full(corr(h0, h1)) * sb_full(corr(e0, e1)), 1e-9))),
                                  "disattenuated_lo": float(np.nanpercentile(dis_b, 2.5)), "disattenuated_hi": float(np.nanpercentile(dis_b, 97.5)),
                                  "forecast_a_rel_improvement": rel(np.arange(n), sb_a, sg_a), "forecast_a_se": float(np.std(rel_a_b, ddof=1)),
                                  "forecast_b_rel_improvement": rel(np.arange(n), sb_b, sg_b), "forecast_c_rel_improvement": rel(np.arange(n), sb_c, sg_c),
                                  "coverage_a": float(cov.sum() / cells_tot), "cells": cells_tot,
                                  "coverage_lower_bound": float(binom.ppf(.025, cells_tot, .95) / cells_tot)})
        hr = pd.DataFrame([r for r in held_rows if r["unit"] == unit and r["mode"] == mode])
        z = 2.94 if unit == "anon" else 3.14
        for j in range(k):
            st = str(status[j])
            def useful(arm):
                r = hr[(hr.coordinate == j) & (hr.arm == arm)].iloc[0]
                return bool(r.forecast_a_rel_improvement - z * r.forecast_a_se > 0 and r.coverage_a >= r.coverage_lower_bound), r
            designated = {"DIRECT": "lex_frozen", "CALIBRATED": "lex_affine"}.get(st)
            u_des, r_des = useful(designated) if designated else (False, None)
            u_ref, r_ref = useful("lex_refit")
            u_fro, _ = useful("lex_frozen")
            u_rd = useful("lex_refit_readerdiff")[0] if SUPP else None
            u_frd = useful("lex_frozen_readerdiff")[0] if SUPP else None
            diag = ("transported" if u_des else
                    "downstream_needs_source_recalibration" if u_ref else
                    "reader_map_layer_fails" if map_rows[[i for i, m in enumerate(map_rows) if m["unit"] == unit and m["mode"] == mode and m["coordinate"] == j][0]]["validation_r2"] < .2 else
                    "not_transported")
            dec_rows.append({"unit": unit, "mode": mode, "coordinate": j, "status": st, "designated_arm": designated,
                             "useful_transported_scoring": u_des, "lex_frozen_useful": u_fro, "lex_refit_useful": u_ref, "diagnosis": diag,
                             **({"lex_refit_readerdiff_useful": u_rd, "lex_frozen_readerdiff_useful": u_frd,
                                 "tau2_refit": float(tau2["lex_refit"][j]), "tau2_frozen": float(tau2["lex_frozen"][j])} if SUPP else {})})
    write_csv(OUT / "map_validation.csv", map_rows)
    write_csv(OUT / "transport_held.csv", held_rows)
    write_csv(OUT / "transport_decision.csv", dec_rows)
    dd = pd.DataFrame(dec_rows)
    acc = {"part_A_replay_exact": replay["exact"], "provenance_gate": gate["pass"], "vocab_size": vocab_size,
           "status_counts": {f"{u}/{m}": dd[(dd.unit == u) & (dd["mode"] == m)].status.value_counts().to_dict() for (u, m) in sets},
           "useful_transported": {f"{u}/{m}": dd[(dd.unit == u) & (dd["mode"] == m) & dd.useful_transported_scoring].coordinate.tolist() for (u, m) in sets},
           "diagnosis_counts": {f"{u}/{m}": dd[(dd.unit == u) & (dd["mode"] == m)].diagnosis.value_counts().to_dict() for (u, m) in sets}}
    acc["positive_construction"] = bool(acc["part_A_replay_exact"] and dd[(dd.unit == "anon") & dd.useful_transported_scoring].shape[0] > 0)
    if SUPP:
        acc["supplement_readerdiff_useful"] = {f"{u}/{m}": {"lex_refit_readerdiff": dd[(dd.unit == u) & (dd["mode"] == m) & (dd.lex_refit_readerdiff_useful == True)].coordinate.tolist(),
                                                            "lex_frozen_readerdiff": dd[(dd.unit == u) & (dd["mode"] == m) & (dd.lex_frozen_readerdiff_useful == True)].coordinate.tolist()} for (u, m) in sets}
    (OUT / "acceptance.json").write_text(json.dumps(acc, indent=1, default=str))
    run = {"total_wall_s": time.time() - t0, "total_cpu_s": time.process_time() - c0, "python": sys.version, "numpy": np.__version__,
           "platform": platform.platform(), "code_sha256": {p: sha(ROOT / p) for p in (
               "grain_research/construction/anonymous_bundle.py", "grain_research/construction/declared_target.py",
               "scripts/run_grain_c4_transport.py", "scripts/score_grain_anonymous_bundle_cold.py")}}
    (OUT / "run_log.json").write_text(json.dumps(run, indent=1))
    print(json.dumps({**acc, "replay": replay, "cpu_s": run["total_cpu_s"]}, indent=1, default=str))


if __name__ == "__main__":
    main()
