#!/usr/bin/env python3
"""GRAIN-C3-02 runner: anonymous source-learned evidence directions on raw e5 comment vectors."""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import csv
import hashlib
import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import eigh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import declared_target as DT

PROTOCOL = ROOT / "docs/GRAIN_C3_02_ANONYMOUS_EVIDENCE_PROTOCOL_V1.json"
OUT = ROOT / "results/grain_construction_v1/C3/anonymous_evidence_r1"
CR = ROOT / "results/grain_vector_mechanisms/all_plan_closure_v1/conditional_reference_r2"
ADDR = ROOT / "results/grain_gptpro_foundation_v3/F16/event_addresses_r1/private_pandora_event_addresses.csv"
ROLES = ROOT / "results/grain_gptpro_foundation_v3/F14/pandora_venue_oom_r1/author_roles.json"
E5 = ROOT / "results/grain_vector_mechanisms/predecessor_integration_v1/material/private_arrays.npz"
H, F, Z_FW, B, NPC, TOPK = list(range(25)), list(range(25, 36)), 2.94, 2000, 64, 5


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


def gen_top(between, noise, k):
    reg = 1e-3 * float(np.mean(np.linalg.eigvalsh(noise)))
    vals, vecs = eigh((between + between.T) / 2, noise + reg * np.eye(len(noise)))
    return vecs[:, ::-1][:, :k], vals[::-1][:k]


def checks_for(y_all, cmat, fits, ids, rng, label, role):
    """C3-01 E1-E5 checks for one evidence family (K directions)."""
    k = y_all.shape[2]
    src_glob = None
    sc = [DT.score_author(fits, y_all[a][H], cmat[a][H]) for a in ids]
    halves = [[DT.score_author(fits, y_all[a][hh], cmat[a][hh]) for a in ids] for hh in (H[0::2], H[1::2])]
    n = len(ids)
    idx = rng.integers(0, n, (B, n))
    out = []
    return sc, halves, idx, out, src_glob, k, n


def within_venue(ev, venue, src_tr):
    """Subtract source_train history venue means (venues >=5 events; others pooled)."""
    vt = venue[src_tr][:, H].ravel()
    et = ev[src_tr][:, H].reshape(-1, ev.shape[2])
    cnt = Counter(vt.tolist())
    pooled = et.mean(0)
    table = {v: et[vt == v].mean(0) for v, c in cnt.items() if c >= 5}
    sub = np.array([[table.get(v, pooled) for v in row] for row in venue])
    return ev - sub


def main():
    global OUT, PROTOCOL
    WV = "--within-venue" in sys.argv
    if WV:
        OUT = ROOT / "results/grain_construction_v1/C3/anonymous_evidence_r1_within_venue"
        PROTOCOL = ROOT / "docs/GRAIN_C3_02_ANONYMOUS_EVIDENCE_PROTOCOL_V1_1_WITHIN_VENUE.json"
    OUT.mkdir(parents=True, exist_ok=False)
    t0, c0 = time.time(), time.process_time()
    proto = json.loads(PROTOCOL.read_text())
    (OUT / "protocol_before_run.json").write_text(json.dumps({**proto, "protocol_sha256": sha(PROTOCOL)}, indent=1))
    roles = json.loads(ROLES.read_text())
    src_tr, src_va, held = (np.array(roles[k]) for k in ("source_train", "source_validation", "held"))
    com = np.load(E5)["comments"].astype(np.float64)
    nrm = com / np.linalg.norm(com, axis=2, keepdims=True)
    del com
    n_auth = nrm.shape[0]
    addr = pd.read_csv(ADDR, usecols=["author_index", "unit", "condition"]).sort_values(["author_index", "unit"])
    venue = addr.condition.to_numpy().reshape(n_auth, 36)
    cnt = Counter(venue[src_tr][:, H].ravel().tolist())
    top = [v for v, _ in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[:20]]
    rule = DT.context_rule(top)
    cmat = np.where(np.isin(venue, top), 1.0, -1.0)
    # ---------------- source-only learning (source_train history)
    X = nrm[src_tr][:, H].reshape(-1, 1024)
    mean = X.mean(0)
    _, sv, vt = np.linalg.svd(X - mean, full_matrices=False)
    V = vt[:NPC].T                                   # 1024 x 64
    Zs = ((nrm[src_tr][:, H] - mean) @ V)            # A x 25 x 64
    am = Zs.mean(1)
    within = (Zs - am[:, None]).reshape(-1, NPC)
    sw = within.T @ within / (within.shape[0] - len(src_tr))
    sb = np.cov(am, rowvar=False) - sw / len(H)
    fa, la = gen_top(sb, sw, TOPK)
    slopes, inv_zz, rss, df = [], [], np.zeros((NPC, NPC)), 0
    for i, a in enumerate(src_tr):
        c = cmat[a][H]
        if min(np.sum(c > 0), np.sum(c < 0)) < 2:
            continue
        x = np.column_stack([np.ones(len(c)), c])
        xi = np.linalg.inv(x.T @ x)
        coef = xi @ x.T @ Zs[i]
        r = Zs[i] - x @ coef
        rss += r.T @ r
        df += len(c) - 2
        slopes.append(coef[1])
        inv_zz.append(xi[1, 1])
    slopes = np.array(slopes)
    samp = rss / df * float(np.mean(inv_zz))
    fr_, lr = gen_top(np.cov(slopes, rowvar=False) - samp, samp, TOPK)
    families = {"author_stability": (V @ fa, la), "response_stability": (V @ fr_, lr),
                "principal_components": (V[:, :TOPK], sv[:TOPK] ** 2 / (X.shape[0] - 1))}
    dirs = {}
    for name, (d, vals) in families.items():
        d = d / np.linalg.norm(d, axis=0, keepdims=True)
        dirs[name] = d
    np.savez(OUT / "private_learned_directions.npz", **dirs, pca_basis=V, center=mean)
    learn_info = {"source_rows": int(X.shape[0]), "authors": int(len(src_tr)), "pca_components": NPC,
                  "pca_variance_share": float(np.sum(sv[:NPC] ** 2) / np.sum(sv ** 2)),
                  "response_stability_authors": int(len(slopes)),
                  "eigenvalues": {k: [float(x) for x in v[1]] for k, v in families.items()}}
    # keyed baseline evidence for the shared diagnostics
    keyed = np.load(CR / "pandora/private_units.npz")["scores"].reshape(n_auth, 36, 30)
    # ---------------- evaluate each family
    rng = np.random.default_rng(20261026)
    checks, diag = [], []
    venue_tr = venue[src_tr][:, H].ravel()
    vc = Counter(venue_tr.tolist())
    vlab = np.array([v if vc[v] >= 5 else "__pooled__" for v in venue_tr])
    codes, vinv = np.unique(vlab, return_inverse=True)
    auth_tr = np.repeat(np.arange(len(src_tr)), len(H))
    for fam, ev in [(k, np.einsum("aud,dk->auk", nrm, v)) for k, v in dirs.items()] + [("keyed_baseline", keyed)]:
        if WV:
            ev = within_venue(ev, venue, src_tr)
        kk = ev.shape[2]
        fits, info = DT.fit_moments([ev[a][H] for a in src_tr], [cmat[a][H] for a in src_tr])
        prior = {t: [float(DT.TARGETS[t] @ np.asarray(f["Sigma"]) @ DT.TARGETS[t]) for f in fits] for t in ("level", "contrast")}
        glob = {s: ev[src_tr][:, H][cmat[src_tr][:, H] == s].reshape(-1, kk).mean(0) for s in (1.0, -1.0)}
        e_tr = ev[src_tr][:, H].reshape(-1, kk)
        for j in range(kk):
            v = e_tr[:, j]
            vm = np.bincount(vinv, weights=v) / np.bincount(vinv)
            am_ = np.bincount(auth_tr, weights=v) / np.bincount(auth_tr)
            diag.append({"family": fam, "direction": j, "venue_r2": float(1 - np.sum((v - vm[vinv]) ** 2) / np.sum((v - v.mean()) ** 2)),
                         "author_mean_share": float(np.var(am_) / np.var(v))})
        for role, ids in (("held", held), ("source_validation", src_va)):
            sc = [DT.score_author(fits, ev[a][H], cmat[a][H]) for a in ids]
            halves = [[DT.score_author(fits, ev[a][hh], cmat[a][hh]) for a in ids] for hh in (H[0::2], H[1::2])]
            n = len(ids)
            idx = rng.integers(0, n, (B, n))
            for j in range(kk):
                se_b, se_g, cells, cover = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
                pd_, od_, ok2 = np.zeros(n), np.zeros(n), np.zeros(n, bool)
                for i, a in enumerate(ids):
                    for s, nm in ((1.0, "m_plus"), (-1.0, "m_minus")):
                        fm = cmat[a][F] == s
                        if not fm.any():
                            continue
                        obs = ev[a][F][fm, j].mean()
                        pred = sc[i][nm]["mean"][j]
                        se_b[i] += (pred - obs) ** 2
                        se_g[i] += (glob[s][j] - obs) ** 2
                        cover[i] += abs(pred - obs) <= 1.959964 * np.sqrt(sc[i][nm]["var"][j] + fits[j]["sigma2"] / fm.sum())
                        cells[i] += 1
                    fp, fmn = cmat[a][F] > 0, cmat[a][F] < 0
                    if fp.any() and fmn.any():
                        ok2[i] = True
                        od_[i] = ev[a][F][fp, j].mean() - ev[a][F][fmn, j].mean()
                        pd_[i] = sc[i]["contrast"]["mean"][j]
                rel = lambda ix, sb_=se_b, sg_=se_g, cl_=cells: (sg_[ix].sum() - sb_[ix].sum()) / sg_[ix].sum()  # noqa: E731
                lv = [np.array([halves[q][i]["level"]["mean"][j] for i in range(n)]) for q in (0, 1)]
                cv = [np.array([halves[q][i]["contrast"]["mean"][j] for i in range(n)]) for q in (0, 1)]
                both = np.array([halves[0][i]["support"] == "observed" and halves[1][i]["support"] == "observed" for i in range(n)])
                stats = {"E1_relative_improvement_vs_global": (rel(np.arange(n)), np.std([rel(ix) for ix in idx], ddof=1)),
                         "E2_contrast_future_corr": (corr(pd_[ok2], od_[ok2]), np.nanstd([corr(pd_[ix][ok2[ix]], od_[ix][ok2[ix]]) for ix in idx], ddof=1)),
                         "E3_level_split_half_corr": (corr(lv[0], lv[1]), np.nanstd([corr(lv[0][ix], lv[1][ix]) for ix in idx], ddof=1)),
                         "E3_contrast_split_half_corr": (corr(cv[0][both], cv[1][both]), np.nanstd([corr(cv[0][ix][both[ix]], cv[1][ix][both[ix]]) for ix in idx], ddof=1))}
                z = Z_FW if fam != "keyed_baseline" else 3.14
                for nm, (val, se) in stats.items():
                    checks.append({"family": fam, "role": role, "direction": j, "statistic": nm, "value": val, "boot_se": float(se),
                                   "fw_lower": float(val - z * se), "lower95": float(val - 1.959964 * se)})
                checks += [{"family": fam, "role": role, "direction": j, "statistic": "E4_predictive_coverage", "value": float(cover.sum() / cells.sum()), "cells": int(cells.sum())},
                           {"family": fam, "role": role, "direction": j, "statistic": "model_reliability_level", "value": float(1 - np.mean([s_["level"]["var"][j] for s_ in sc]) / prior["level"][j])},
                           {"family": fam, "role": role, "direction": j, "statistic": "model_reliability_contrast", "value": float(1 - np.mean([s_["contrast"]["var"][j] for s_ in sc]) / prior["contrast"][j])},
                           {"family": fam, "role": role, "direction": j, "statistic": "sd_level_scores", "value": float(np.std([s_["level"]["mean"][j] for s_ in sc])), "prior_sd": float(np.sqrt(prior["level"][j]))},
                           {"family": fam, "role": role, "direction": j, "statistic": "sd_contrast_scores", "value": float(np.std([s_["contrast"]["mean"][j] for s_ in sc])), "prior_sd": float(np.sqrt(prior["contrast"][j]))}]
    write_csv(OUT / "empirical_checks.csv", checks)
    write_csv(OUT / "direction_diagnostics.csv", diag)
    ck = pd.DataFrame(checks)
    rows = []
    for (fam, j), g in ck[ck.role == "held"].groupby(["family", "direction"]):
        g = g.set_index("statistic")
        lvl = g.loc["E1_relative_improvement_vs_global", "fw_lower"] > 0 and g.loc["E3_level_split_half_corr", "fw_lower"] > 0 \
            and g.loc["sd_level_scores", "value"] >= .1 * g.loc["sd_level_scores", "prior_sd"]
        con = g.loc["E2_contrast_future_corr", "fw_lower"] > 0 and g.loc["E3_contrast_split_half_corr", "fw_lower"] > 0 \
            and g.loc["sd_contrast_scores", "value"] >= .1 * g.loc["sd_contrast_scores", "prior_sd"]
        d = [x for x in diag if x["family"] == fam and x["direction"] == j][0]
        rows.append({"family": fam, "direction": int(j), "level_supported": bool(lvl), "contrast_supported": bool(con),
                     "venue_r2": d["venue_r2"], "author_mean_share": d["author_mean_share"],
                     "E1_rel": g.loc["E1_relative_improvement_vs_global", "value"], "E2": g.loc["E2_contrast_future_corr", "value"],
                     "E3_level": g.loc["E3_level_split_half_corr", "value"], "E3_contrast": g.loc["E3_contrast_split_half_corr", "value"],
                     "coverage": g.loc["E4_predictive_coverage", "value"]})
    write_csv(OUT / "acceptance_by_direction.csv", rows)
    ad = pd.DataFrame(rows)
    learned = ad[ad.family != "keyed_baseline"]
    acc = {"anonymous_route_positive": bool((learned.level_supported | learned.contrast_supported).any()),
           "contrast_discovery": bool(learned.contrast_supported.any()),
           "supported": {f: {"level": ad[(ad.family == f) & ad.level_supported].direction.tolist(),
                             "contrast": ad[(ad.family == f) & ad.contrast_supported].direction.tolist()} for f in ad.family.unique()},
           "learn_info": learn_info, "z_learned": Z_FW, "z_keyed": 3.14,
           "naming": "none; anonymous response structure only"}
    (OUT / "acceptance.json").write_text(json.dumps(acc, indent=1, default=float))
    run = {"total_wall_s": time.time() - t0, "total_cpu_s": time.process_time() - c0, "python": sys.version, "numpy": np.__version__,
           "platform": platform.platform(), "code_sha256": {p: sha(ROOT / p) for p in ("grain_research/construction/declared_target.py", "scripts/run_grain_anonymous_evidence.py")}}
    (OUT / "run_log.json").write_text(json.dumps(run, indent=1))
    print(json.dumps({k: acc[k] for k in ("anonymous_route_positive", "contrast_discovery", "supported")}, indent=1))
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print("cpu", run["total_cpu_s"])


if __name__ == "__main__":
    main()
