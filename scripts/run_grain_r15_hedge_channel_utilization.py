#!/usr/bin/env python
"""R15 hedge-evidence utilization via a source-calibrated reader channel
(docs/GRAIN_R15_HEDGE_CHANNEL_UTILIZATION_PROTOCOL_V1.md). CPU only, 0 neural calls, frozen readings."""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import csv
import hashlib
import itertools
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog, nnls

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "results/grain_gptpro_foundation_v3/F17/hedge_commitment_reader_r1"
OUT = ROOT / "results/grain_gptpro_foundation_v3/F17/hedge_channel_utilization_r1"
PROTOCOL = ROOT / "docs/GRAIN_R15_HEDGE_CHANNEL_UTILIZATION_PROTOCOL_V1.md"
READ_SHA = "962b58740cd9aecbe49fde78cafb0fafda232fce9c23f811bdd1e1d26897f11e"
MAT_SHA = "1b07fb5336e4909e267d97e844b47d6cc5330abc62a00fefeca20adbc53d12ac"
FORMS = ["bare", "epistemic", "deferential", "noncommittal"]
OWNERS = ["self", "other"]
FRAMES = ["neutral", "disagreeing"]
CELLS = list(itertools.product(FORMS, OWNERS, FRAMES))
Q5 = ["own_commitment", "uncertainty", "deference", "reported_other", "no_commitment"]
VARIANTS = {"V1": Q5, "V2": Q5[1:], "V3": Q5 + ["truth_diagnostic"]}
FORM_Q = {"bare": "own_commitment", "epistemic": "uncertainty", "deferential": "deference", "noncommittal": "no_commitment"}
N, R, R_LP, SEED = 100, 200, 10, 1601
LAB = np.array(["E", "N", "C"])


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load() -> pd.DataFrame:
    assert sha(IN / "readings.jsonl") == READ_SHA and sha(IN / "materials_frozen.json") == MAT_SHA
    rows = [json.loads(l) for l in (IN / "readings.jsonl").read_text().splitlines()]
    df = pd.DataFrame(rows)
    assert len(df) == 576
    meta = df.groupby("text_id")[["core", "split", "form", "owner", "frame", "wording"]].first()
    wide = {}
    for p, tag in (("p_entailment", "ent"), ("p_contradiction", "con"), ("p_neutral", "neu")):
        w = df.pivot(index="text_id", columns="role", values=p)
        w.columns = [f"{c}:{tag}" for c in w.columns]
        wide[tag] = w
    T = meta.join(wide["ent"]).join(wide["con"]).join(wide["neu"])
    assert len(T) == 96 and T.notna().all().all()
    T["cell"] = [CELLS.index((f, o, fr)) for f, o, fr in zip(T.form, T.owner, T.frame)]
    return T.sort_values(["core", "cell", "wording"])


def feats(T, qs):
    return T[[f"{q}:{t}" for q in qs for t in ("ent", "con")]].to_numpy(float)


def code_of(T):
    ent = T[[f"{q}:ent" for q in Q5]].to_numpy(); neu = T[[f"{q}:neu" for q in Q5]].to_numpy(); con = T[[f"{q}:con" for q in Q5]].to_numpy()
    idx = np.stack([ent, neu, con], -1).argmax(-1)
    return ["".join(LAB[r]) for r in idx]


def fit_mu(T, qs):
    X = feats(T, qs)
    return np.stack([X[T.cell.to_numpy() == c].mean(0) for c in range(16)])  # (16, d)


def tolerance(Tsrc, qs):
    X = feats(Tsrc, qs); cell = Tsrc.cell.to_numpy()
    mu = np.stack([X[cell == c].mean(0) for c in range(16)])
    res = X - mu[cell]
    S = res.T @ res / (len(X) - 16)
    m1 = fit_mu(Tsrc[Tsrc.core == 1], qs); m2 = fit_mu(Tsrc[Tsrc.core == 2], qs)
    shift2 = ((m1 - m2) ** 2).mean(0)
    d = lambda n: 2 * np.sqrt(0.75 * shift2 + (np.diag(S) / n if np.isfinite(n) else 0))
    return S, shift2, d


def code_channel(Tfit, cols):
    codes = code_of(Tfit)
    dic = sorted(set(codes))
    K = np.zeros((len(dic) + 1, len(cols)))
    cell = Tfit.cell.to_numpy()
    for j, c in enumerate(cols):
        cc = [codes[i] for i in np.where(cell == c)[0]]
        for k in cc:
            K[dic.index(k), j] += 1
        K[:, j] = (K[:, j] + .5) / (len(cc) + .5 * (len(dic) + 1))
    return dic, K


SCEN = {"K0_self_neutral": ("self", "neutral"), "K0_self_disagreeing": ("self", "disagreeing"),
        "K0_other_neutral": ("other", "neutral"), "K0_other_disagreeing": ("other", "disagreeing"),
        "K1_self": ("self", None), "K1_other": ("other", None), "K2_neutral": (None, "neutral"),
        "K2_disagreeing": (None, "disagreeing"), "K3": (None, None)}
UTIL_SCEN = ["K0_self_neutral", "K1_self", "K2_neutral", "K3"]


def columns(name):
    o, fr = SCEN[name]
    return [i for i, (f, oo, ff) in enumerate(CELLS) if (o is None or oo == o) and (fr is None or ff == fr)]


def functionals(name, cols):
    fs = {f"pi_{f}": np.array([CELLS[c][0] == f for c in cols], float) for f in FORMS}
    fs["pi_epi_plus_non"] = np.array([CELLS[c][0] in ("epistemic", "noncommittal") for c in cols], float)
    if SCEN[name][0] is None:
        fs["pi_other"] = np.array([CELLS[c][1] == "other" for c in cols], float)
    return fs


def lp_interval(M, mbar, delta, a):
    k = M.shape[1]
    A_ub = np.vstack([M, -M]); b_ub = np.concatenate([mbar + delta, -(mbar - delta)])
    kw = dict(A_ub=A_ub, b_ub=b_ub, A_eq=np.ones((1, k)), b_eq=[1.0], bounds=(0, None), method="highs")
    lo = linprog(a, **kw)
    if lo.status != 0:
        return None
    hi = linprog(-a, **kw)
    return float(lo.fun), float(-hi.fun)


def simplex_ls(M, mbar, delta):
    W = 1 / delta
    A = np.vstack([M * W[:, None], 1e3 * np.ones((1, M.shape[1]))])
    b = np.concatenate([mbar * W, [1e3]])
    w, _ = nnls(A, b, maxiter=5000)
    return w / w.sum()


def em_batch(K, counts, iters=500, tol=1e-10):
    Rr, k = counts.shape[0], K.shape[1]
    w = np.full((Rr, k), 1 / k)
    n = counts.sum(1, keepdims=True)
    for _ in range(iters):
        pc = w @ K.T  # (R, codes)
        new = w * ((counts / np.maximum(pc, 1e-300)) @ K) / n
        if np.abs(new - w).max() < tol:
            w = new
            break
        w = new
    return w


def targets():
    rng = np.random.default_rng(SEED)
    return [np.full(4, .25)] + [rng.dirichlet(np.ones(4)) for _ in range(11)]


def experiment(Tfit, Tapp, name, delta_fn, tag, seed_ctx, lp_variants=("V1", "V2", "V3"), rows=None, lprows=None):
    """Resample Tapp texts (finite material resampling); channel fitted on Tfit only."""
    cols = columns(name)
    o, fr = SCEN[name]
    fs = functionals(name, cols)
    app_cell = Tapp.cell.to_numpy()
    by_cell = {c: np.where(app_cell == c)[0] for c in cols}
    assert all(len(v) >= 1 for v in by_cell.values())
    Xapp = {v: feats(Tapp, qs) for v, qs in VARIANTS.items()}
    Mv = {v: fit_mu(Tfit, qs)[cols].T for v, qs in VARIANTS.items()}
    dN = {v: delta_fn[v](N) for v in VARIANTS}
    dic, K = code_channel(Tfit, cols)
    app_codes = code_of(Tapp)
    code_idx = np.array([dic.index(c) if c in dic else len(dic) for c in app_codes])
    ent_formq = Tapp[[f"{FORM_Q[f]}:ent" for f in FORMS]].to_numpy()
    ent_rep = Tapp["reported_other:ent"].to_numpy()
    app_form = np.array([FORMS.index(f) for f in Tapp.form]); app_owner = np.array([OWNERS.index(o) for o in Tapp.owner])
    summary = []
    for ti, pi in enumerate(targets()):
        pw = np.array([.5, .5]) if o is None else np.array([1.0, 0.0] if o == "self" else [0.0, 1.0])
        fw = np.array([.5, .5]) if fr is None else np.array([1.0, 0.0] if fr == "neutral" else [0.0, 1.0])
        wcell = np.array([pi[FORMS.index(CELLS[c][0])] * pw[OWNERS.index(CELLS[c][1])] * fw[FRAMES.index(CELLS[c][2])] for c in cols])
        wcell = wcell / wcell.sum()
        truth = {k: float(a @ wcell) for k, a in fs.items()}
        rng = np.random.default_rng([SEED, *seed_ctx, UTIL_SCEN.index(name), ti])
        picks = np.empty((R, N), int)
        for r in range(R):
            cc = rng.choice(len(cols), N, p=wcell)
            u = rng.random(N)
            picks[r] = [by_cell[cols[c]][int(x * len(by_cell[cols[c]]))] for c, x in zip(cc, u)]
        unseen = float((code_idx[picks] == len(dic)).mean())
        comp = np.stack([(app_form[picks] == j).mean(1) for j in range(4)], 1)  # realized collection composition (protocol target)
        comp_other = (app_owner[picks] == 1).mean(1)
        est = {}
        m = ent_formq[picks].mean(1); est["N1_query_average"] = m / m.sum(1, keepdims=True)
        hard = ent_formq[picks].argmax(2); est["N2_hard_count"] = np.stack([(hard == j).mean(1) for j in range(4)], 1)
        counts = np.zeros((R, len(dic) + 1))
        for r in range(R):
            np.add.at(counts[r], code_idx[picks[r]], 1)
        wem = em_batch(K, counts)
        fmat = np.stack([fs[f"pi_{f}"] for f in FORMS], 1)  # (k, 4)
        est["C2_channel_hard_V1"] = wem @ fmat
        wv = {}
        for v in VARIANTS:
            W = np.stack([simplex_ls(Mv[v], Xapp[v][picks[r]].mean(0), dN[v]) for r in range(R)])
            wv[v] = W
            est[f"C1_channel_soft_{v}"] = W @ fmat
        own_naive = ent_rep[picks].mean(1)
        for en, E in est.items():
            merged = E[:, 1] + E[:, 3]
            if en.startswith("C1"):
                ow = wv[en.split("_")[-1]] @ fs["pi_other"] if "pi_other" in fs else None
            elif en.startswith("C2"):
                ow = wem @ fs["pi_other"] if "pi_other" in fs else None
            elif en.startswith("N1"):
                ow = own_naive if "pi_other" in fs else None
            else:
                ow = None
            summary.append({"context": tag, "scenario": name, "target": ti, "estimator": en,
                            "mae_form": float(np.abs(E - comp).mean()),
                            "mae_merged": float(np.abs(merged - (comp[:, 1] + comp[:, 3])).mean()),
                            "mae_owner": None if ow is None else float(np.abs(ow - comp_other).mean()),
                            "mae_form_vs_generating": float(np.abs(E - pi).mean()),
                            "mae_merged_vs_generating": float(np.abs(merged - (pi[1] + pi[3])).mean()),
                            "mae_owner_vs_generating": None if ow is None else float(np.abs(ow - truth["pi_other"]).mean()),
                            "unseen_code_mass": unseen})
            if rows is not None:
                for r in range(R):
                    rows.append([tag, name, ti, r, en, *np.round(E[r], 5), round(float(merged[r]), 5),
                                 "" if ow is None else round(float(ow[r]), 5), *np.round(pi, 5), round(truth.get("pi_other", float("nan")), 5),
                                 *np.round(comp[r], 5), round(float(comp_other[r]), 5)])
        for v in lp_variants:
            for r in range(R_LP):
                mbar = Xapp[v][picks[r]].mean(0)
                real = {f"pi_{f}": comp[r, j] for j, f in enumerate(FORMS)}
                real["pi_epi_plus_non"] = comp[r, 1] + comp[r, 3]; real["pi_other"] = comp_other[r]
                for fk, a in fs.items():
                    iv = lp_interval(Mv[v], mbar, dN[v], a)
                    lprows.append([tag, name, ti, r, v, fk, float(real[fk]), truth[fk], "" if iv is None else round(iv[0], 6),
                                   "" if iv is None else round(iv[1], 6), iv is None,
                                   "" if iv is None else bool(iv[0] - 1e-9 <= real[fk] <= iv[1] + 1e-9),
                                   "" if iv is None else bool(iv[0] - 1e-9 <= truth[fk] <= iv[1] + 1e-9)])
    return summary


def main():
    t0 = time.time(); started = datetime.now(timezone.utc).isoformat()
    if OUT.exists():
        raise FileExistsError("hedge_channel_utilization_r1 exists; preserve earlier runs")
    OUT.mkdir(parents=True)
    T = load()
    src, test = T[T.split == "source"], T[T.split == "test"]
    assert set(src.core) == {1, 2} and set(test.core) == {3} and len(src) == 64 and len(test) == 32
    assert (src.groupby("cell").size() == 4).all() and (test.groupby("cell").size() == 2).all()
    res = {"protocol": str(PROTOCOL.relative_to(ROOT)), "protocol_sha256": sha(PROTOCOL), "readings_sha256": READ_SHA,
           "materials_sha256": MAT_SHA, "unit": "text (joint query response)", "N": N, "R": R, "R_LP": R_LP, "seed": SEED,
           "neural_calls": 0}
    tol, chan = {}, {}
    for v, qs in VARIANTS.items():
        S, shift2, dfn = tolerance(src, qs)
        tol[v] = dfn
        sd = np.sqrt(np.diag(S)); corr = S / np.outer(sd, sd)
        ev = np.clip(np.linalg.eigvalsh(corr), 0, None)
        chan[v] = {"features": [f"{q}:{t}" for q in qs for t in ("ent", "con")], "mu_source": fit_mu(src, qs).round(6).tolist(),
                   "S_within_diag": np.diag(S).round(8).tolist(), "shift2": shift2.round(8).tolist(),
                   "delta_N100": dfn(N).round(6).tolist(), "delta_inf": dfn(np.inf).round(6).tolist(),
                   "within_cell_corr": corr.round(4).tolist(), "participation_ratio": float(ev.sum() ** 2 / (ev ** 2).sum()),
                   "dims": len(qs) * 2}
    ent5 = src[[f"{q}:ent" for q in Q5]].to_numpy(); cell = src.cell.to_numpy()
    r5 = ent5 - np.stack([ent5[cell == c].mean(0) for c in range(16)])[cell]
    c5 = np.corrcoef(r5.T); ev5 = np.clip(np.linalg.eigvalsh(c5), 0, None)
    res["within_cell_p_ent_corr_V1"] = {"queries": Q5, "corr": c5.round(4).tolist(), "participation_ratio": float(ev5.sum() ** 2 / (ev5 ** 2).sum())}
    dic, _ = code_channel(src, list(range(16)))
    test_codes = code_of(test)
    res["hard_code"] = {"source_dictionary_size": len(dic), "source_dictionary": dic,
                        "test_unseen_texts": int(sum(c not in dic for c in test_codes)), "test_texts": len(test_codes)}
    (OUT / "channel_source.json").write_text(json.dumps({"cells": CELLS, "variants": chan, "code_dictionary": dic}, indent=1) + "\n")
    conf = {}
    for nm, TT in (("source", src), ("test", test)):
        pred = TT[[f"{FORM_Q[f]}:ent" for f in FORMS]].to_numpy().argmax(1)
        cm = pd.crosstab(pd.Series([FORMS[i] for i in pred], name="pred"), TT.form.rename("true").reset_index(drop=True)).reindex(index=FORMS, columns=FORMS, fill_value=0)
        conf[nm] = {"accuracy": float((np.array([FORMS[i] for i in pred]) == TT.form.to_numpy()).mean()), "confusion_pred_by_true": cm.to_dict()}
    res["naive_hard_rule_per_text_diagnostic"] = conf
    # 1. algebraic identifiability and 2. population compatible-set widths
    alg = []
    for v, qs in VARIANTS.items():
        mu = fit_mu(src, qs)
        for name in SCEN:
            cols = columns(name); M = mu[cols].T; A = np.vstack([M, np.ones(len(cols))])
            P = np.linalg.pinv(A) @ A; rank = int(np.linalg.matrix_rank(A))
            wref = np.full(len(cols), 1 / len(cols)); mref = M @ wref
            for fk, a in functionals(name, cols).items():
                resid = float(np.linalg.norm(a - P @ a) / np.linalg.norm(a))
                i100 = lp_interval(M, mref, tol[v](N), a); iinf = lp_interval(M, mref, tol[v](np.inf), a)
                alg.append({"variant": v, "scenario": name, "columns": len(cols), "rank": rank, "null_dim": len(cols) - rank,
                            "functional": fk, "rowspace_residual": round(resid, 6), "truth_ref": float(a @ wref),
                            "width_N100": None if i100 is None else round(i100[1] - i100[0], 4),
                            "width_inf": None if iinf is None else round(iinf[1] - iinf[0], 4),
                            "interval_N100": i100, "interval_inf": iinf})
    res["identifiability"] = alg
    # 3. merged grouping
    merged = {}
    for v, qs in VARIANTS.items():
        mu = fit_mu(src, qs); dinf = tol[v](np.inf)
        per = {}
        for o, fr in itertools.product(OWNERS, FRAMES):
            e = mu[CELLS.index(("epistemic", o, fr))]; n_ = mu[CELLS.index(("noncommittal", o, fr))]
            z = np.abs(e - n_) / dinf
            per[f"{o}_{fr}"] = {"max_std_diff": round(float(z.max()), 3), "features_over_1": int((z > 1).sum()),
                                "argmax_feature": chan[v]["features"][int(z.argmax())], "std_diff": z.round(3).tolist()}
        merged[v] = per
    codes_src = code_of(src)
    ce = {c for c, f in zip(codes_src, src.form) if f == "epistemic"}; cn = {c for c, f in zip(codes_src, src.form) if f == "noncommittal"}
    merged["hard_code_overlap"] = {"epistemic_codes": sorted(ce), "noncommittal_codes": sorted(cn), "jaccard": len(ce & cn) / len(ce | cn)}
    res["merged_grouping"] = merged
    # 5. mapping requirements
    mapping = {}
    for v, qs in VARIANTS.items():
        mu = fit_mu(src, qs); dinf = tol[v](np.inf)
        sep = {}
        for o, fr in itertools.product(OWNERS, FRAMES):
            sep[f"{o}_{fr}"] = {f"{a}|{b}": round(float((np.abs(mu[CELLS.index((a, o, fr))] - mu[CELLS.index((b, o, fr))]) / dinf).max()), 3)
                                for a, b in itertools.combinations(FORMS, 2)}
        one_d = {}
        for fr in FRAMES:
            Z = np.stack([mu[CELLS.index((f, "self", fr))] / dinf for f in FORMS]); Z = Z - Z.mean(0)
            U, s, Vt = np.linalg.svd(Z, full_matrices=False)
            one_d[f"self_{fr}"] = {"axis1_share": round(float(s[0] ** 2 / (s ** 2).sum()), 4), "axis2_share": round(float(s[1] ** 2 / (s ** 2).sum()), 4),
                                   "axis1_scores": dict(zip(FORMS, (U[:, 0] * s[0]).round(3).tolist())),
                                   "axis2_scores": dict(zip(FORMS, (U[:, 1] * s[1]).round(3).tolist())),
                                   "axis2_top_features": [chan[v]["features"][i] for i in np.argsort(-np.abs(Vt[1]))[:3]]}
        mapping[v] = {"pairwise_separability_max_std": sep, "one_dimensionality": one_d}
    res["mapping"] = mapping
    # test shift diagnostic
    shift = {}
    for v, qs in VARIANTS.items():
        z = np.abs(fit_mu(test, qs) - fit_mu(src, qs)) / tol[v](np.inf)
        shift[v] = {"max": round(float(z.max()), 3), "mean": round(float(z.mean()), 3), "frac_over_1": round(float((z > 1).mean()), 4),
                    "worst_cell": list(CELLS[int(np.unravel_index(z.argmax(), z.shape)[0])]),
                    "worst_feature": chan[v]["features"][int(np.unravel_index(z.argmax(), z.shape)[1])]}
    res["test_shift_diagnostic"] = shift
    # 4. utilization experiments
    rows, lprows, summ = [], [], []
    t_src = time.time()
    for a_, b_ in ((1, 2), (2, 1)):  # source-internal cross-core check first
        for name in UTIL_SCEN:
            summ += experiment(src[src.core == a_], src[src.core == b_], name, tol, f"source_crosscore_{a_}to{b_}", (a_, b_), ("V1",), rows, lprows)
    res["seconds_crosscore"] = time.time() - t_src
    t_app = time.time()
    for name in UTIL_SCEN:
        summ += experiment(src, test, name, tol, "test_pooled_source_channel", (0, 3), ("V1", "V2", "V3"), rows, lprows)
    res["seconds_test"] = time.time() - t_app
    for cc in (1, 2):
        for name in UTIL_SCEN:
            summ += experiment(src[src.core == cc], test, name, tol, f"test_core{cc}_only_channel", (cc, 3), (), None, lprows)  # summaries only
    S = pd.DataFrame(summ)
    agg = S.groupby(["context", "scenario", "estimator"])[["mae_form", "mae_merged", "mae_owner", "mae_form_vs_generating", "mae_merged_vs_generating",
                                                          "mae_owner_vs_generating", "unseen_code_mass"]].mean().round(4).reset_index()
    res["utilization_summary"] = agg.to_dict(orient="records")
    L = pd.DataFrame(lprows, columns=["context", "scenario", "target", "rep", "variant", "functional", "truth_realized", "truth_generating", "lo", "hi",
                                      "infeasible", "covered", "covered_generating"])
    L["width"] = pd.to_numeric(L.hi, errors="coerce") - pd.to_numeric(L.lo, errors="coerce")
    lagg = L.groupby(["context", "scenario", "variant", "functional"]).agg(n=("rep", "size"), infeasible=("infeasible", "mean"),
                                                                           coverage=("covered", lambda x: float(np.mean([bool(v) for v in x if v != ""])) if any(v != "" for v in x) else None),
                                                                           coverage_generating=("covered_generating", lambda x: float(np.mean([bool(v) for v in x if v != ""])) if any(v != "" for v in x) else None),
                                                                           mean_width=("width", "mean")).round(4).reset_index()
    res["compatible_set_summary"] = lagg.to_dict(orient="records")
    hdr = ["context", "scenario", "target", "rep", "estimator", *[f"est_{f}" for f in FORMS], "est_merged", "est_other", *[f"gen_{f}" for f in FORMS], "gen_other", *[f"real_{f}" for f in FORMS], "real_other"]
    with (OUT / "utilization_rows.csv").open("w", newline="") as h:
        w = csv.writer(h); w.writerow(hdr); w.writerows(rows)
    L.to_csv(OUT / "compatible_set_rows.csv", index=False)
    res["seconds_total"] = time.time() - t0
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    (OUT / "runtime.json").write_text(json.dumps({"started": started, "finished": datetime.now(timezone.utc).isoformat(),
                                                  "seconds": res["seconds_total"], "neural_calls": 0, "threads": 1,
                                                  "inputs": {"readings.jsonl": READ_SHA, "materials_frozen.json": MAT_SHA},
                                                  "code_sha256": sha(Path(__file__)), "protocol_sha256": res["protocol_sha256"]}, indent=1) + "\n")
    print(f"done {res['seconds_total']:.1f}s rows={len(rows)} lp={len(lprows)}")


if __name__ == "__main__":
    main()
