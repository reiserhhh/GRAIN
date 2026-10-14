#!/usr/bin/env python3
"""Re-audit r2 follow-up packet 3 (FU3-a..f); zero neural calls, no labels."""

from __future__ import annotations

import collections
import csv
import hashlib
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from grain_research.gptpro_foundation_v3.f08_materials import material_bank

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/GRAIN_REAUDIT_R2_FOLLOWUP3_PROTOCOL_V1.md"
PROTOCOL_SHA = "e75ed9491051422e8e7338a5702bd537d166311db0d9eef1cf480470c0bbd4c4"
V3 = ROOT / "results/grain_gptpro_foundation_v3"
TT = V3 / "F16/text_transport_full_r1"
R17 = V3 / "R17/natural_sensitivity_r1"
LB = ROOT / "results/grain_vector_mechanisms/all_plan_closure_v1/local_bilinear_r2"
OUT = V3 / "F17/reaudit_r2_followup3"
SEED = 20260926


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ols_fit(x, y):
    return np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)[0]


def ols_apply(w, x):
    return np.column_stack([np.ones(len(x)), x]) @ w


def fu3_a():
    mats = {r["material_id"]: r for r in csv.DictReader((TT / "materials.csv").open())}
    ids = [r["material_id"] for r in csv.DictReader((TT / "reader_scores.csv").open())]
    grain = np.load(TT / "grain_features.npz")["features"].astype(float)
    nli = defaultdict(dict)
    for r in csv.DictReader((TT / "nli_probabilities.csv").open()):
        mid, hyp = r["audit_id"].rsplit(":", 1)
        nli[mid][hyp] = [float(r["p_contradiction"]), float(r["p_entailment"]), float(r["p_neutral"])]
    nx = np.array([[v for h in ("positive", "negative", "qualification") for v in nli[m][h]] for m in ids])
    template = np.array([m.rsplit("_", 2)[0] for m in ids])
    perm = np.random.default_rng(SEED).permutation(np.unique(template))
    fold = np.array([{t: i % 2 for i, t in enumerate(perm)}[t] for t in template])
    pole = np.array([int(mats[m]["pole"]) for m in ids])
    qual = np.array([int(mats[m]["qualification"]) for m in ids])
    res = {}
    for f in (0, 1):
        w = ols_fit(grain[fold != f], nx[fold != f])
        pred = ols_apply(w, grain)
        for inter in ("qualification_flip", "pole_flip"):
            me, ae = [], []
            for t in np.unique(template[fold == f]):
                ix = np.flatnonzero(template == t)
                for fixed in (0, 1):
                    if inter == "qualification_flip":
                        a = [i for i in ix if pole[i] == fixed and qual[i] == 0]
                        b = [i for i in ix if pole[i] == fixed and qual[i] == 1]
                    else:
                        a = [i for i in ix if qual[i] == fixed and pole[i] == 0]
                        b = [i for i in ix if qual[i] == fixed and pole[i] == 1]
                    if a and b:
                        me.append(pred[b[0]] - pred[a[0]])
                        ae.append(nx[b[0]] - nx[a[0]])
            me, ae = np.array(me), np.array(ae)
            cos = np.sum(me * ae, 1) / (np.linalg.norm(me, axis=1) * np.linalg.norm(ae, axis=1) + 1e-12)
            q = 7
            res[f"fold{f}|{inter}"] = {"pairs": len(me), "median_vector_cosine": float(np.median(cos)),
                                       "qual_entailment_mean_effect_mapped": float(me[:, q].mean()),
                                       "qual_entailment_mean_effect_actual": float(ae[:, q].mean()),
                                       "qual_entailment_corr": float(np.corrcoef(me[:, q], ae[:, q])[0, 1]) if me[:, q].std() > 0 else None}
    return res


def fu3_b():
    m = pd.read_csv(R17 / "manifest.csv")
    vec = {}
    for corpus in ("pandora", "essays"):
        mc = m[m.corpus == corpus]
        with np.load(LB / corpus / "private_units.npz", allow_pickle=False) as z:
            h = z["h"].astype(float)
        d = {}
        for v in ("base", "contraction_expand", "sentence_reverse", "punctuation_strip"):
            d[("e5", v)] = np.array([np.load(R17 / "private_vectors/e5" / f"{corpus}_{a}_{v}.npy") for a in mc.author_index], float)
            d[("bge", v)] = np.array([np.load(R17 / "private_vectors/bge" / f"{corpus}_{a}_{v}.npy") for a in mc.author_index], float)
        d[("e5", "partner")] = h[mc.partner_row.to_numpy()]
        d[("bge", "partner")] = np.array([np.load(R17 / "private_vectors/bge" / f"{corpus}_{a}_partner.npy") for a in mc.author_index], float)
        vec[corpus] = d
    res = {}
    for fit_c, test_c in (("pandora", "essays"), ("essays", "pandora")):
        xs = np.vstack([vec[fit_c][("e5", "base")], vec[fit_c][("e5", "partner")]])
        ys = np.vstack([vec[fit_c][("bge", "base")], vec[fit_c][("bge", "partner")]])
        w = ols_fit(xs, ys)[1:]
        for v in ("contraction_expand", "sentence_reverse", "punctuation_strip", "partner"):
            md = (vec[test_c][("e5", v)] - vec[test_c][("e5", "base")]) @ w
            ad = vec[test_c][("bge", v)] - vec[test_c][("bge", "base")]
            cos = np.sum(md * ad, 1) / (np.linalg.norm(md, axis=1) * np.linalg.norm(ad, axis=1) + 1e-12)
            res[f"fit_{fit_c}_test_{test_c}|{v}"] = {"median_cosine": float(np.median(cos)),
                                                    "norm_ratio_mapped_over_actual": float(np.median(np.linalg.norm(md, axis=1) / (np.linalg.norm(ad, axis=1) + 1e-12)))}
    return res


def fu3_c():
    from scripts.prepare_grain_predecessor_reference import DEFAULT
    material = json.loads((DEFAULT / "private_material.json").read_text())
    subs = [[c["subreddit"] for v in b["views"] for c in v] for b in material["bundles"]]
    with np.load(LB / "pandora/private_predictions.npz", allow_pickle=False) as z:
        held, pred = z["held_indices"], z["semantic"][:, 0]
    counts = collections.Counter(s for i in held for s in subs[int(i)])
    res = {}
    for k in (20, 30, 50):
        top = [s for s, _ in counts.most_common(k)]
        share = np.array([[sum(x == s for x in subs[int(i)]) / 36 for s in top] + [sum(x not in top for x in subs[int(i)]) / 36] for i in held])
        for transform in ("share", "logshare"):
            x = share if transform == "share" else np.log(share + 1 / 36)
            u, sv, _ = np.linalg.svd(x - x.mean(0), full_matrices=False)
            for mm in (3, 5, 10):
                X = np.column_stack([np.ones(len(held)), u[:, :mm] * sv[:mm]])
                fit = np.empty_like(pred)
                for i in range(len(pred)):
                    mk = np.ones(len(pred), bool)
                    mk[i] = False
                    fit[i] = X[i] @ np.linalg.lstsq(X[mk], pred[mk], rcond=None)[0]
                res[f"K{k}|{transform}|pc{mm}"] = float(1 - ((pred - fit) ** 2).sum() / ((pred - pred.mean()) ** 2).sum())
    vals = np.array(list(res.values()))
    return {"universes": res, "min": float(vals.min()), "median": float(np.median(vals)), "max": float(vals.max()),
            "all_positive": bool((vals > 0).all())}


def readability(texts):
    words = sents = syl = 0
    for t in texts:
        w = re.findall(r"[A-Za-z']+", t)
        words += len(w)
        sents += max(1, len(re.findall(r"[.!?]+", t)))
        syl += sum(max(1, len(re.findall(r"[aeiouy]+", x.lower()))) for x in w)
    wps, spw = words / sents, syl / words
    return {"texts": len(texts), "words_per_sentence": round(wps, 2), "syllables_per_word": round(spw, 3),
            "flesch_reading_ease": round(206.835 - 1.015 * wps - 84.6 * spw, 1)}


def fu3_d():
    f08 = [r["text"] for r in material_bank()]
    f16 = pd.read_csv(TT / "materials.csv").text.tolist()
    gen = pd.read_json(ROOT / "results/grain_vector_mechanisms/persona_generation_trace_v1/expanded/generations.jsonl", lines=True).text.tolist()
    return {"F08_bank": readability(f08), "F16_bank": readability(f16), "persona_generations_reference": readability(gen)}


def fu3_e():
    beta_c = np.array([-1.0, 0.0, 1.0])
    rho, alpha, sw, se = 0.6, 0.3, 0.4, 0.5
    readers = (0.3, 0.5)
    T, xi_true = 40, 0.25

    def simulate(rng, theta, kappa, K):
        s, prev, rows, acts = 0.0, 1, [], []
        for t in range(T):
            a = float(rng.random() < 0.5)
            acts.append(a)
            if rng.random() < 0.7:
                q = np.exp(0.8 * theta * beta_c + 0.5 * s * beta_c + 0.3 * (np.arange(3) == prev))
                c = int(rng.choice(3, p=q / q.sum()))
                x = theta + kappa * beta_c[c] + s + rng.normal(0, se)
                rd = t % 2
                rows.append((t, c, rd, x + rng.normal(0, readers[rd] * K)))
                prev = c
                s = rho * s + alpha * a + xi_true * x + rng.normal(0, sw)
            else:
                s = rho * s + alpha * a + rng.normal(0, sw)
        return rows, acts

    def design(rows, acts, xi):
        n = 2 + 2 * T + len(rows)
        s_coef, s_const = np.zeros(n), 0.0
        M, d, k = [], [], 0
        obs = {r[0]: r for r in rows}
        for t in range(T):
            if t in obs:
                _, c, _rd, _y = obs[t]
                x = s_coef.copy()
                x[0] += 1
                x[1] += beta_c[c]
                x[2 + T + t] += 1
                row = x.copy()
                row[2 + 2 * T + k] += 1
                M.append(row)
                d.append(s_const)
                s_coef = rho * s_coef + xi * x
                s_const = rho * s_const + alpha * acts[t] + xi * s_const
                k += 1
            else:
                s_coef = rho * s_coef
                s_const = rho * s_const + alpha * acts[t]
            s_coef[2 + t] += 1
        return np.array(M), np.array(d), s_coef, s_const

    def pvar(rows):
        return np.r_[1.0, 0.16, np.full(T, sw ** 2), np.full(T, se ** 2), [readers[r[2]] ** 2 for r in rows]]

    def loglik(rows, acts, xi):
        M, d, _, _ = design(rows, acts, xi)
        mu = np.zeros(M.shape[1])
        mu[1] = 1.0
        S = (M * pvar(rows)) @ M.T
        y = np.array([r[3] for r in rows]) - M @ mu - d
        _sign, logdet = np.linalg.slogdet(S)
        return -0.5 * (logdet + y @ np.linalg.solve(S, y))

    def posterior(rows, acts, xi):
        M, d, sc, scc = design(rows, acts, xi)
        mu = np.zeros(M.shape[1])
        mu[1] = 1.0
        v = pvar(rows)
        S = (M * v) @ M.T
        y = np.array([r[3] for r in rows]) - M @ mu - d
        targets = np.vstack([np.eye(M.shape[1])[0], np.eye(M.shape[1])[1], sc])
        cross = (targets * v) @ M.T
        mean = targets @ mu + cross @ np.linalg.solve(S, y)
        cov = (targets * v) @ targets.T - cross @ np.linalg.solve(S, cross.T)
        mean[2] += scc
        return mean, np.sqrt(np.maximum(np.diag(cov), 0))

    rng = np.random.default_rng(SEED + 11)
    src = [simulate(rng, rng.standard_normal(), 1 + 0.4 * rng.standard_normal(), rng.normal(1, 0.2)) for _ in range(200)]
    grid = [round(float(g), 2) for g in np.arange(-0.10, 0.351, 0.05)]
    ll = {g: float(sum(loglik(r, a, g) for r, a in src)) for g in grid}
    xi_hat = max(ll, key=ll.get)
    nu = np.full(3, 1 / 3)
    err = {"E2_xi0": [], "E3_xi_hat": [], "E1_naive": []}
    sd_state = []
    for _ in range(300):
        th, ka, K = rng.standard_normal(), 1 + 0.4 * rng.standard_normal(), rng.normal(1, 0.2)
        rows, acts = simulate(rng, th, ka, K)
        truth = th + ka * (nu @ beta_c)
        err["E1_naive"].append(np.mean([r[3] for r in rows]) - truth)
        for name, xi in (("E2_xi0", 0.0), ("E3_xi_hat", xi_hat)):
            mean, sd = posterior(rows, acts, xi)
            err[name].append(mean[0] + mean[1] * (nu @ beta_c) - truth)
            if name == "E3_xi_hat":
                sd_state.append(sd[2])
    return {"xi_true": xi_true, "xi_hat_profile_likelihood": xi_hat, "loglik_by_xi": ll,
            "rmse_nu1": {k: float(np.sqrt(np.mean(np.square(v)))) for k, v in err.items()},
            "final_state_posterior_sd_mean": float(np.mean(sd_state)),
            "misspecification_declared": "estimator assumes proficiency K=1 and ignores that selection depends on latent state and history"}


def fu3_f():
    rng = np.random.default_rng(SEED + 13)
    sd = {"topic_selection": 0.6, "interpretation": 0.5, "style": 0.7, "realization": 0.4}
    n, err = 500, 0.8

    def arm(chans):
        return sum(rng.normal(0, sd[c], n) for c in chans) + rng.normal(0, err, n)

    A = arm(("topic_selection", "interpretation", "style", "realization"))
    B = arm(("interpretation", "style", "realization"))
    C = arm(("style", "realization"))
    return {"true_var": {k: v ** 2 for k, v in sd.items()},
            "est_topic_selection_var(A-B)": float(A.var(ddof=1) - B.var(ddof=1)),
            "est_interpretation_var(B-C)": float(B.var(ddof=1) - C.var(ddof=1)),
            "est_style_plus_realization_plus_error(C)": float(C.var(ddof=1)),
            "true_style_plus_realization_plus_error": sd["style"] ** 2 + sd["realization"] ** 2 + err ** 2,
            "identification": "arms A/B/C identify topic-selection and interpretation variance by differences; style and realization are NOT separable by these arms and need a further realization-constrained arm"}


def main():
    if digest(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("Protocol changed")
    OUT.mkdir(parents=True, exist_ok=True)
    out = {"protocol_sha256": PROTOCOL_SHA}
    for name, fn in (("FU3_a_R21_signature_under_map", fu3_a), ("FU3_b_R23_edit_signature_across_encoders", fu3_b),
                     ("FU3_c_R35_topic_share_multiverse", fu3_c), ("FU3_d_R33_readability", fu3_d),
                     ("FU3_e_R04_xi_estimation", fu3_e), ("FU3_f_R05_arms", fu3_f)):
        out[name] = fn()
        print(name, json.dumps(out[name])[:1000], flush=True)
    out.update({"code_sha256": digest(Path(__file__)), "neural_calls": 0, "labels_read": False,
                "created": time.strftime("%Y-%m-%d %H:%M:%S %Z")})
    (OUT / "followup3.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
