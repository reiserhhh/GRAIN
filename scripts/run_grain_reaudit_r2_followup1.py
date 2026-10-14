#!/usr/bin/env python3
"""Re-audit r2 follow-up packet 1 (FU1-a..g); zero neural calls, no labels."""

from __future__ import annotations

import collections
import hashlib
import itertools
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from grain_research.gptpro_foundation_v3.r36_gates import gates
from scripts.run_grain_a01_bisimulation_metric import metric

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/GRAIN_REAUDIT_R2_FOLLOWUP1_PROTOCOL_V1.md"
PROTOCOL_SHA = "3dc3a4bcd0e729b42d4a9be3b0ba555eff0281754aa8b83be66c0372d49c62dd"
V3 = ROOT / "results/grain_gptpro_foundation_v3"
LB = ROOT / "results/grain_vector_mechanisms/all_plan_closure_v1/local_bilinear_r2"
OP = ROOT / "results/grain_vector_mechanisms/conditional_operator_v1/r2"
RUNTIME = ROOT / "results/v7_multiview_projection/e1_v72_full_20260715/artifacts/common_source_comment_representation.joblib"
OUT = V3 / "F17/reaudit_r2_followup1"
SEED = 20260926


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def subspace_sim(a, b, k=5):
    wa, va = np.linalg.eigh(a)
    wb, vb = np.linalg.eigh(b)
    qa, qb = va[:, np.argsort(wa)[::-1][:k]], vb[:, np.argsort(wb)[::-1][:k]]
    return float(np.linalg.norm(qa.T @ qb) ** 2 / k), float(abs(qa[:, 0] @ qb[:, 0]))


def fu1_a():
    res = {}
    for corpus in ("pandora", "essays"):
        with np.load(LB / corpus / "private_units.npz", allow_pickle=False) as z:
            g, a = z["g"].astype(float), z["authors"]
        ids, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
        means = np.zeros((len(ids), g.shape[1]))
        np.add.at(means, inv, g)
        means /= cnt[:, None]
        dev = g - means[inv]
        sw = dev.T @ dev / (len(g) - len(ids))
        sb = np.cov(means, rowvar=False) - sw * np.mean(1 / cnt)
        s, c1 = subspace_sim(sw, sb)
        res[corpus] = {"authors": len(ids), "units": len(g), "top5_subspace_similarity": s,
                       "first_eigvec_congruence": c1, "trace_between_over_within": float(np.trace(sb) / np.trace(sw))}
    return res


def fu1_b():
    m = pd.read_csv(V3 / "F16/text_transport_full_r1/reader_scores.csv")
    res = {}
    for role, g in m.groupby("role"):
        x = np.column_stack([(g[c] - g[c].mean()) / g[c].std() for c in ("nli_qualification", "grain_qualification")])
        n, r = x.shape
        grand = x.mean()
        ms_i = r * ((x.mean(1) - grand) ** 2).sum() / (n - 1)
        ms_r = n * ((x.mean(0) - grand) ** 2).sum() / (r - 1)
        resid = x - x.mean(1, keepdims=True) - x.mean(0, keepdims=True) + grand
        ms_e = (resid ** 2).sum() / ((n - 1) * (r - 1))
        v_i, v_r, v_e = max((ms_i - ms_e) / r, 0), max((ms_r - ms_e) / n, 0), ms_e
        tot = v_i + v_r + v_e
        res[role] = {"items": n, "item_var": v_i / tot, "reader_var": v_r / tot, "reader_x_item_plus_error": v_e / tot,
                     "reader_variance_ratio": (v_r + v_e) / tot}
    return res


def fu1_c():
    from scripts.prepare_grain_predecessor_reference import DEFAULT
    material = json.loads((DEFAULT / "private_material.json").read_text())
    subs = [[c["subreddit"] for v in b["views"] for c in v] for b in material["bundles"]]
    with np.load(LB / "pandora/private_predictions.npz", allow_pickle=False) as z:
        held, pred = z["held_indices"], z["semantic"]
    counts = collections.Counter(s for i in held for s in subs[int(i)])
    top = [s for s, _ in counts.most_common(30)]
    share = np.array([[sum(x == s for x in subs[int(i)]) / 36 for s in top] + [sum(x not in top for x in subs[int(i)]) / 36] for i in held])
    u, sv, _ = np.linalg.svd(share - share.mean(0), full_matrices=False)
    X = np.column_stack([np.ones(len(held)), u[:, :5] * sv[:5]])
    res = {}
    for t, trait in enumerate(("Extraversion", "Neuroticism", "Agreeableness", "Conscientiousness", "Openness")):
        y = pred[:, t]
        fit = np.empty_like(y)
        for i in range(len(y)):
            mk = np.ones(len(y), bool)
            mk[i] = False
            fit[i] = X[i] @ np.linalg.lstsq(X[mk], y[mk], rcond=None)[0]
        resid = y - fit
        res[trait] = {"residual_variance_share": float(resid.var() / y.var()), "corr_residual_with_original": float(np.corrcoef(resid, y)[0, 1])}
    return res


def fu1_d():
    arr = np.load(OP / "action_arrays.npz", allow_pickle=False)
    order = json.loads((OP / "pair_order.json").read_text())
    pairs = pd.read_csv(OP / "pairs.csv").set_index("pair").loc[order].reset_index()
    texts = {json.loads(x)["text_id"]: json.loads(x)["text"] for x in (ROOT / "results/grain_vector_mechanisms/persona_generation_trace_v1/expanded/generations.jsonl").read_text().splitlines()}
    rt = joblib.load(RUNTIME)
    d24 = np.asarray(rt.transform([texts[t] for t in pairs.high_id]) - rt.transform([texts[t] for t in pairs.low_id]), float)
    src, ds = arr["source"], arr["ds"]
    eff = pd.DataFrame([json.loads(x) for x in (OP / "native_effects.jsonl").read_text().splitlines()])
    app = ~src
    cells = (pairs.axis + "|" + pairs.family + "|" + pairs.situation).to_numpy()[app]
    res = {}
    rng = np.random.default_rng(SEED)
    for qi, q in enumerate(("extraversion", "neuroticism")):
        e = eff[(eff["query"] == q) & (eff.kind == "donor")].set_index("pair")["effect"]
        donor = np.array([e[p] for p in pairs.pair[app]])
        preds = {}
        for name, X in (("four", arr["dx"]), ("twentyfour", d24)):
            w = np.linalg.lstsq(X[src], ds[src, qi], rcond=None)[0]
            preds[name] = X[app] @ w
        out = {}
        for tgt_name, tgt in (("donor_action", donor), ("full_contrast", ds[app, qi])):
            r4 = np.corrcoef(preds["four"], tgt)[0, 1]
            r24 = np.corrcoef(preds["twentyfour"], tgt)[0, 1]
            uc = np.unique(cells)
            diffs = []
            for _ in range(2000):
                pick = rng.choice(uc, len(uc))
                idx = np.concatenate([np.flatnonzero(cells == c) for c in pick])
                diffs.append(np.corrcoef(preds["twentyfour"][idx], tgt[idx])[0, 1] - np.corrcoef(preds["four"][idx], tgt[idx])[0, 1])
            out[tgt_name] = {"r_four": float(r4), "r_twentyfour": float(r24), "diff": float(r24 - r4),
                             "diff_ci": [float(x) for x in np.nanpercentile(diffs, [2.5, 97.5])]}
        res[q] = out
    return res


def fu1_e():
    from scripts.prepare_grain_predecessor_reference import DEFAULT
    material = json.loads((DEFAULT / "private_material.json").read_text())
    subs = np.array([c["subreddit"] for b in material["bundles"] for v in b["views"] for c in v])
    freq = collections.Counter(subs.tolist())
    top = [s for s, _ in freq.most_common(50)]
    logf = np.log(np.array([freq[s] for s in top], float))
    res = {"recovery": {}, "mechanism_auc": {}}
    feats = {"composition_only": [], "composition_plus_exposure_log": []}
    labels = []
    for rep in range(5):
        rng = np.random.default_rng(np.random.SeedSequence([SEED, 77, rep]))
        w = rng.standard_normal(50)
        w = (w - w.mean()) / w.std()
        for mech in ("choice", "opportunity"):
            z = rng.standard_normal(400)
            comp, expo = [], []
            for u in range(400):
                if mech == "choice":
                    offered = np.ones(50, bool)
                    logits = logf + z[u] * w
                else:
                    offered = rng.random(50) < 1 / (1 + np.exp(-(z[u] * w)))
                    offered[np.argmax(logf)] = True
                    logits = logf.copy()
                logits = np.where(offered, logits, -np.inf)
                p = np.exp(logits - logits[offered].max())
                p /= p.sum()
                k = rng.choice(50, 36, p=p)
                comp.append(np.bincount(k, minlength=50) / 36)
                expo.append(offered.astype(float))
            comp, expo = np.array(comp), np.array(expo)
            uu, _ss, _ = np.linalg.svd(comp - comp.mean(0), full_matrices=False)
            res["recovery"].setdefault(mech, []).append(abs(float(np.corrcoef(uu[:, 0], z)[0, 1])))
            feats["composition_only"].append(comp)
            feats["composition_plus_exposure_log"].append(np.hstack([comp, expo]))
            labels.append(np.full(400, mech == "opportunity", float))
    y = np.concatenate(labels)
    for name, blocks in feats.items():
        X = np.vstack(blocks)
        res["mechanism_auc"][name] = float(cross_val_score(LogisticRegression(max_iter=2000, C=1.0), X, y, cv=5, scoring="roc_auc").mean())
    res["recovery"] = {k: float(np.mean(v)) for k, v in res["recovery"].items()}
    return res


def fu1_f():
    states = list(itertools.product((0, 1), (0, 1), (0, 1)))
    idx = {s: i for i, s in enumerate(states)}

    def p_one(w):
        return 0.75 if (w[0] ^ w[1]) == 1 else 0.25

    n = len(states)
    R, P = np.zeros((n, 1)), np.zeros((1, n, n))
    for s in states:
        q = p_one(s[:2])
        R[idx[s], 0] = q
        for y, py in ((1, q), (0, 1 - q)):
            for nb in (0, 1):
                P[0, idx[s], idx[(s[1], y, nb)]] += py * 0.5
    d_full, _ = metric(R, P)
    pairs = [(idx[s], idx[(s[0], s[1], 1 - s[2])]) for s in states if s[2] == 0]
    full_max = max(d_full[i, j] for i, j in pairs)
    Rq = np.zeros((n, 1))
    Rq[:, 0] = R[:, 0]
    nuisance_reward = R.copy()
    for s in states:
        nuisance_reward[idx[s], 0] = R[idx[s], 0] + (0.1 if s[2] == 1 else 0.0)
    d_nr, _ = metric(nuisance_reward, P)
    return {"max_d_nuisance_only_pairs_target_reward": float(full_max),
            "max_d_nuisance_only_pairs_when_reward_reads_nuisance": float(max(d_nr[i, j] for i, j in pairs)),
            "interpretation": "with a G-invariant reward the metric already identifies nuisance-only pairs (d=0); a reward that reads the nuisance separates them, so the G-quotient requirement is a constraint on the scoring functional"}


def fu1_g():
    p1 = {"replicated_regularity": "PASS", "structural_reliability": "NOT_RUN", "admission": "FAIL",
          "shortcut_control": "NOT_RUN", "rival_explanations": "NOT_RUN", "discriminating_intervention": "BLOCKED",
          "reader_gauge": "BLOCKED", "transport": "NOT_RUN", "prospective_validation": "BLOCKED"}
    p2 = {"replicated_regularity": "PASS", "structural_reliability": "NOT_RUN", "admission": "FAIL",
          "shortcut_control": "FAIL", "rival_explanations": "NOT_RUN", "discriminating_intervention": "BLOCKED",
          "reader_gauge": "NOT_RUN", "transport": "NOT_RUN", "prospective_validation": "BLOCKED"}
    return {"self_reference": gates(p1), "choice_composition": gates(p2)}


def main():
    if digest(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("Protocol changed")
    OUT.mkdir(parents=True, exist_ok=True)
    out = {"protocol_sha256": PROTOCOL_SHA}
    for name, fn in (("FU1_a_R28_within_between", fu1_a), ("FU1_b_R16_reader_variance", fu1_b),
                     ("FU1_c_R31_topic_residual", fu1_c), ("FU1_d_R01_matched_4_vs_24", fu1_d),
                     ("FU1_e_R27_nonidentifiability", fu1_e), ("FU1_f_R08_G_invariance", fu1_f),
                     ("FU1_g_R36_gates", fu1_g)):
        out[name] = fn()
        print(name, json.dumps(out[name])[:600], flush=True)
    out.update({"code_sha256": digest(Path(__file__)), "neural_calls": 0, "labels_read": False,
                "created": time.strftime("%Y-%m-%d %H:%M:%S %Z")})
    (OUT / "followup1.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
