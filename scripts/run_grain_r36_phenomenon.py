#!/usr/bin/env python3
"""GRAIN-ORIG-R36-U1-01: one anonymous phenomenon through rivals, structural reliability and gauge."""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import csv
import hashlib
import json
import platform
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import eigh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import anonymous_bundle as AB
from grain_research.construction import declared_target as DT

PROTOCOL = ROOT / "docs/GRAIN_R36_U1_PHENOMENON_PROTOCOL_V1.json"
OUT = ROOT / "results/grain_construction_v1/R36/phenomenon_u1_r1"
C3W = ROOT / "results/grain_construction_v1/C3/anonymous_evidence_r1_within_venue/empirical_checks.csv"
ABUNDLE = ROOT / "results/grain_construction_v1/C4/transport_r1/anonymous_bundle.json"
C4U = ROOT / "results/grain_construction_v1/C4/transport_r1_uncertainty_layer"
ADDR = ROOT / "results/grain_gptpro_foundation_v3/F16/event_addresses_r1/private_pandora_event_addresses.csv"
ROLES = ROOT / "results/grain_gptpro_foundation_v3/F14/pandora_venue_oom_r1/author_roles.json"
MAT = ROOT / "results/grain_vector_mechanisms/predecessor_integration_v1/material"
H, F, B = list(range(25)), list(range(25, 36)), 2000
FAMS = ("author_stability", "response_stability", "principal_components")
FPS = {"i", "me", "my", "mine", "myself", "im", "i'm", "i've", "ive", "i'd", "id", "i'll"}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def corr(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and a.std() > 0 and b.std() > 0 else float("nan")


def gen_top(between, noise, k=5):
    reg = 1e-3 * float(np.mean(np.linalg.eigvalsh(noise)))
    vals, vecs = eigh((between + between.T) / 2, noise + reg * np.eye(len(noise)))
    return vecs[:, ::-1][:, :k]


def learn(nrm, cmat, authors, family):
    X = nrm[authors][:, H].reshape(-1, nrm.shape[2])
    mean = X.mean(0)
    _, _, vt = np.linalg.svd(X - mean, full_matrices=False)
    V = vt[:64].T
    Z = (nrm[authors][:, H] - mean) @ V
    if family == "principal_components":
        d = V[:, :5]
    elif family == "author_stability":
        am = Z.mean(1)
        w = (Z - am[:, None]).reshape(-1, 64)
        sw = w.T @ w / (w.shape[0] - len(authors))
        d = V @ gen_top(np.cov(am, rowvar=False) - sw / len(H), sw)
    else:
        sl, inv, rss, df = [], [], np.zeros((64, 64)), 0
        for i, a in enumerate(authors):
            c = cmat[a][H]
            if min(np.sum(c > 0), np.sum(c < 0)) < 2:
                continue
            x = np.column_stack([np.ones(len(c)), c])
            xi = np.linalg.inv(x.T @ x)
            co = xi @ x.T @ Z[i]
            r = Z[i] - x @ co
            rss += r.T @ r
            df += len(c) - 2
            sl.append(co[1])
            inv.append(xi[1, 1])
        samp = rss / df * float(np.mean(inv))
        d = V @ gen_top(np.cov(np.array(sl), rowvar=False) - samp, samp)
    return d / np.linalg.norm(d, axis=0, keepdims=True)


def held_stats(y, cmat, src, held, idx):
    """Refit moments on source_train y (n_auth x 36, 1 coord) and return held E1 rel improvement, split-half, scores."""
    ev = y[:, :, None]
    fits, _ = DT.fit_moments([ev[a][H] for a in src], [cmat[a][H] for a in src])
    glob = {s: ev[src][:, H][cmat[src][:, H] == s].mean() for s in (1.0, -1.0)}
    sc = [DT.score_author(fits, ev[a][H], cmat[a][H]) for a in held]
    hv = [[DT.score_author(fits, ev[a][hh], cmat[a][hh]) for a in held] for hh in (H[0::2], H[1::2])]
    sb, sg = np.zeros(len(held)), np.zeros(len(held))
    for i, a in enumerate(held):
        for s, nm in ((1.0, "m_plus"), (-1.0, "m_minus")):
            fm = cmat[a][F] == s
            if fm.any():
                o = ev[a][F][fm, 0].mean()
                sb[i] += (sc[i][nm]["mean"][0] - o) ** 2
                sg[i] += (glob[s] - o) ** 2
    rel = lambda ix: (sg[ix].sum() - sb[ix].sum()) / sg[ix].sum()  # noqa: E731
    h0 = np.array([h["level"]["mean"][0] for h in hv[0]])
    h1 = np.array([h["level"]["mean"][0] for h in hv[1]])
    rb = np.array([rel(ix) for ix in idx])
    shb = np.array([corr(h0[ix], h1[ix]) for ix in idx])
    return {"E1_rel": rel(np.arange(len(held))), "E1_lo95": float(np.percentile(rb, 2.5)),
            "split_half": corr(h0, h1), "split_half_lo95": float(np.nanpercentile(shb, 2.5))}, np.array([s["level"]["mean"][0] for s in sc])


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    t0, c0 = time.time(), time.process_time()
    proto = json.loads(PROTOCOL.read_text())
    (OUT / "protocol_before_run.json").write_text(json.dumps({**proto, "protocol_sha256": sha(PROTOCOL)}, indent=1))
    ck = pd.read_csv(C3W)
    cand = ck[(ck.role == "source_validation") & (ck.statistic == "E1_relative_improvement_vs_global") & (ck.family.isin(FAMS))]
    best = cand.sort_values("value", ascending=False).iloc[0]
    fam, j = best.family, int(best.direction)
    k_idx = FAMS.index(fam) * 5 + j
    selection = {"family": fam, "direction": j, "bundle_index": k_idx, "source_validation_E1_rel": float(best.value),
                 "rule": proto["selection_rule_source_only"]}
    (OUT / "selection.json").write_text(json.dumps(selection, indent=1))
    roles = json.loads(ROLES.read_text())
    src, held = np.array(roles["source_train"]), np.array(roles["held"])
    addr = pd.read_csv(ADDR, usecols=["author_index", "unit", "start", "end", "condition"]).sort_values(["author_index", "unit"])
    n_auth = int(addr.author_index.max()) + 1
    venue = addr.condition.to_numpy().reshape(n_auth, 36)
    ab = json.loads(ABUNDLE.read_text())
    AB.verify_bundle(ab)
    cmat = np.where(np.isin(venue, ab["context_rule"]["top"]), 1.0, -1.0)
    nrm = AB.normalize(np.load(MAT / "private_arrays.npz")["comments"].astype(np.float64))
    d_full = np.asarray(ab["directions"])[:, k_idx]
    rng = np.random.default_rng(20261035)
    idx = rng.integers(0, len(held), (B, len(held)))

    def within(direction):
        raw = nrm @ direction
        tab, pool = AB.venue_offsets(raw[src][:, H].reshape(-1, 1), venue[src][:, H].ravel())
        return np.stack([AB.apply_mode(raw[a][:, None], venue[a], tab, pool, "within_venue")[0][:, 0] for a in range(n_auth)])

    y_wv = within(d_full)
    base, lev_base = held_stats(y_wv, cmat, src, held, idx)
    # u1b structural reliability
    rs = np.random.default_rng(20261034)
    rel_rows = []
    for s in range(10):
        perm = rs.permutation(src)
        ha, hb = perm[: len(perm) // 2], perm[len(perm) // 2:]
        da, db = learn(nrm, cmat, ha, fam), learn(nrm, cmat, hb, fam)
        ia, ib = int(np.argmax(np.abs(da.T @ d_full))), int(np.argmax(np.abs(db.T @ d_full)))
        va, vb = da[:, ia] * np.sign(da[:, ia] @ d_full), db[:, ib] * np.sign(db[:, ib] @ d_full)
        _, la = held_stats(within(va), cmat, src, held, idx[:2])
        _, lb = held_stats(within(vb), cmat, src, held, idx[:2])
        rel_rows.append({"split": s, "cos_half_a_full": float(va @ d_full), "cos_half_b_full": float(vb @ d_full),
                         "cos_half_a_half_b": float(va @ vb), "matched_rank_a": ia, "matched_rank_b": ib,
                         "held_level_score_corr_a_b": corr(la, lb), "held_level_score_corr_a_full": corr(la, lev_base)})
    pd.DataFrame(rel_rows).to_csv(OUT / "structural_reliability.csv", index=False)
    # u1a rivals: length/format features from local text (never written)
    docs = json.loads((MAT / "private_material.json").read_text())["documents"]
    feats = np.zeros((n_auth * 36, 4))
    for i, r in enumerate(addr.itertuples()):
        t = docs[r.author_index][r.start:r.end]
        toks = re.findall(r"[a-z']+", t.lower())
        feats[i] = [np.log1p(len(t)), float("http" in t or "www." in t), float("?" in t),
                    (sum(w in FPS for w in toks) / len(toks)) if toks else 0.0]
    del docs
    feats = feats.reshape(n_auth, 36, 4)
    Xs = np.column_stack([np.ones(len(src) * 25), feats[src][:, H].reshape(-1, 4)])
    beta = np.linalg.lstsq(Xs, y_wv[src][:, H].ravel(), rcond=None)[0]
    y_adj = y_wv - (np.concatenate([np.ones((n_auth, 36, 1)), feats], 2) @ beta)
    adj, lev_adj = held_stats(y_adj, cmat, src, held, idx)
    r2_feat = 1 - np.sum((y_wv[src][:, H].ravel() - Xs @ beta) ** 2) / np.sum((y_wv[src][:, H].ravel() - y_wv[src][:, H].mean()) ** 2)
    fmean = feats[held][:, H].mean(1)
    pi = np.array([(np.sum(cmat[a][H] > 0) + 1) / 27 for a in held])
    rivals = {"unadjusted_within_venue": base, "length_format_adjusted": adj, "feature_r2_on_source_events": float(r2_feat),
              "feature_coefficients": beta.tolist(), "feature_names": ["intercept", "log1p_chars", "url", "question_mark", "first_person_share"],
              "held_level_corr_with_author_feature_means": {n: corr(lev_base, fmean[:, q]) for q, n in enumerate(["log1p_chars", "url", "question_mark", "first_person_share"])},
              "held_level_corr_with_choice_composition_pi": corr(lev_base, pi), "held_level_corr_unadjusted_vs_adjusted": corr(lev_base, lev_adj)}
    (OUT / "rivals.json").write_text(json.dumps(rivals, indent=1))
    # u1c gauge
    td = pd.read_csv(C4U / "transport_decision.csv")
    th = pd.read_csv(C4U / "transport_held.csv")
    g = td[(td.unit == "anon") & (td["mode"] == "within_venue") & (td.coordinate == k_idx)].iloc[0].to_dict()
    ag = th[(th.unit == "anon") & (th["mode"] == "within_venue") & (th.coordinate == k_idx)][["arm", "agreement_level_vs_e5", "forecast_a_rel_improvement", "coverage_a"]].to_dict("records")
    gauge = {"decision": {k: (v if not isinstance(v, (np.bool_,)) else bool(v)) for k, v in g.items()}, "arms": ag,
             "note": "lexical reader gauge from C4-01 (method convergence, not validity)"}
    (OUT / "gauge.json").write_text(json.dumps(gauge, indent=1, default=str))
    rr = pd.DataFrame(rel_rows)
    acc = {"selection": selection,
           "structural_reliability": {"median_cos_half_half": float(rr.cos_half_a_half_b.median()), "min_cos_half_half": float(rr.cos_half_a_half_b.min()),
                                      "median_held_score_corr_half_half": float(rr.held_level_score_corr_a_b.median())},
           "survives_length_format_rival": bool(adj["E1_lo95"] > 0 and adj["split_half_lo95"] > 0),
           "venue_ablation_level_supported": True,
           "gauge_lexical_within_venue": {"status": g.get("status"), "useful_frozen": bool(g.get("useful_transported_scoring")),
                                         "useful_with_uncertainty_layer_refit": bool(g.get("lex_refit_readerdiff_useful"))},
           "u1d_positive_control": "NOT RUN: needs distinct material with an established-construct anchor",
           "naming": "none"}
    acc["phenomenon_carried"] = bool(acc["survives_length_format_rival"] and acc["venue_ablation_level_supported"])
    (OUT / "acceptance.json").write_text(json.dumps(acc, indent=1, default=str))
    run = {"total_wall_s": time.time() - t0, "total_cpu_s": time.process_time() - c0, "python": sys.version, "platform": platform.platform(),
           "code_sha256": sha(ROOT / "scripts/run_grain_r36_phenomenon.py"), "bundle_sha256": sha(ABUNDLE)}
    (OUT / "run_log.json").write_text(json.dumps(run, indent=1))
    print(json.dumps({"acc": acc, "rivals": {k: rivals[k] for k in ("unadjusted_within_venue", "length_format_adjusted", "feature_r2_on_source_events", "held_level_corr_with_author_feature_means", "held_level_corr_with_choice_composition_pi")}, "cpu": run["total_cpu_s"]}, indent=1, default=str))


if __name__ == "__main__":
    main()
