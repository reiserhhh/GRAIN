#!/usr/bin/env python
"""R25 designed wording-perturbation facet (docs/GRAIN_R25_DESIGNED_WORDING_FACET_PROTOCOL_V1.md). CPU only, 0 reader calls."""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_grain_r15_hedge_channel_utilization as ch  # noqa: E402  (frozen packet-16 channel functions)

PROTOCOL = ROOT / "docs/GRAIN_R25_DESIGNED_WORDING_FACET_PROTOCOL_V1.md"
PROTOCOL_SHA = "65cb041b452873731949b5d8e80884cfe8846dc6d81ad1469c59536dd6a13de2"
MAT = ROOT / "results/grain_gptpro_foundation_v3/F17/hedge_commitment_reader_r1/materials_frozen.json"
OUT = ROOT / "results/grain_gptpro_foundation_v3/F12/designed_wording_facet_r1"
QUERIES6 = ["own_commitment", "uncertainty", "deference", "reported_other", "no_commitment", "truth_diagnostic"]
FORMS = ["bare", "epistemic", "deferential", "noncommittal"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def effect(x):
    lv = sorted(set(x)); X = np.zeros((len(x), len(lv) - 1))
    for j, l in enumerate(lv[:-1]):
        X[:, j] = (np.asarray(x) == l).astype(float)
    X[np.asarray(x) == lv[-1], :] = -1.0
    return X


def inter(A, B):
    return np.column_stack([A[:, i] * B[:, j] for i in range(A.shape[1]) for j in range(B.shape[1])])


def proj_ss(X, y):
    yc = y - y.mean()
    b, *_ = np.linalg.lstsq(X, yc, rcond=None)
    return float(((X @ b) ** 2).sum())


def anova(T, y, with_core):
    cellkey = T.core.astype(str) + "|" + T.form + "|" + T.owner + "|" + T.frame
    grand = y.mean(); cm = pd.Series(y).groupby(cellkey.values).transform("mean").to_numpy()
    ss_tot = float(((y - grand) ** 2).sum()); ss_between = float(((cm - grand) ** 2).sum()); ss_within = float(((y - cm) ** 2).sum())
    F, O, Fr, W = effect(T.form), effect(T.owner), effect(T.frame), effect(T.wording)
    terms = {}
    if with_core:
        terms["core"] = effect(T.core)
    terms.update({"form": F, "owner": O, "frame": Fr, "form_x_owner": inter(F, O), "form_x_frame": inter(F, Fr), "owner_x_frame": inter(O, Fr)})
    if not with_core:
        terms["form_x_owner_x_frame"] = inter(inter(F, O), Fr)
    wterms = {"wording": W, "wording_x_form": inter(W, F)}
    rows, named_b, named_w = [], 0.0, 0.0
    allX = []
    for name, X in {**terms, **wterms}.items():
        ss = proj_ss(X, y); df = int(np.linalg.matrix_rank(X))
        rows.append({"term": name, "df": df, "ss": ss, "rank_ok": df == X.shape[1]}); allX.append(X)
        if name in terms:
            named_b += ss
        else:
            named_w += ss
    n_cells = cellkey.nunique()
    df_b_named = sum(r["df"] for r in rows if r["term"] in terms)
    rows.append({"term": "between_lumped_higher_order", "df": int(n_cells - 1 - df_b_named), "ss": ss_between - named_b, "rank_ok": True})
    df_w_named = sum(r["df"] for r in rows if r["term"] in wterms)  # wording (1) + wording_x_form (3)
    rows.append({"term": "wording_x_context_lumped", "df": int(len(y) - n_cells - df_w_named), "ss": ss_within - named_w, "rank_ok": True})
    ss_joint = proj_ss(np.column_stack(allX), y)  # orthogonality check: joint = sum of separate
    for r in rows:
        r["share"] = r["ss"] / ss_tot if ss_tot > 0 else float("nan"); r["ms"] = r["ss"] / r["df"] if r["df"] > 0 else float("nan")
    return rows, {"ss_total": ss_tot, "ss_between": ss_between, "ss_within": ss_within, "df_total": len(y) - 1,
                  "sum_of_terms_minus_total": sum(r["ss"] for r in rows) - ss_tot, "orthogonality_gap": ss_joint - (named_b + named_w)}


def main():
    t0 = time.time()
    if sha(PROTOCOL) != PROTOCOL_SHA:
        raise RuntimeError("protocol changed")
    if OUT.exists():
        raise FileExistsError("designed_wording_facet_r1 exists; preserve earlier runs")
    OUT.mkdir(parents=True)
    T = ch.load()  # verifies readings + materials hashes; one row per text with p_ent/p_con/p_neu per query
    mats = json.loads(MAT.read_text())
    prem = {t["text_id"]: t["premise"] for t in mats["texts"]}
    # ---- pair audit
    audit = []
    for (core, form, owner, frame), g in T.groupby(["core", "form", "owner", "frame"]):
        ro = T[(T.core == core) & (T.form == form) & (T.owner == owner) & (T.frame == frame)]
        ids = dict(zip(ro.wording, ro.index))
        po, pp = prem[ids["original"]], prem[ids["paraphrase"]]
        prop = [c for c in mats["cores"] if c["core"] == core][0]
        ptxt = prop["clause"]
        fr_txt = mats["frames"][frame]
        same_prefix = po.split(" the speaker says:")[0] == pp.split(" the speaker says:")[0]
        wrapper = ("My neighbour told me:" in po) == ("My neighbour told me:" in pp) == (owner == "other")
        has_p = (ptxt.lower().rstrip(".") in po.lower()) and (ptxt.lower().rstrip(".") in pp.lower())
        audit.append({"core": core, "form": form, "owner": owner, "frame": frame, "text_original": ids["original"], "text_paraphrase": ids["paraphrase"],
                      "same_frame_prefix": bool(same_prefix), "owner_wrapper_consistent": bool(wrapper), "proposition_in_both": bool(has_p),
                      "frame_text_found": bool(fr_txt is None or po.startswith(fr_txt)), "premise_original": po, "premise_paraphrase": pp})
    A = pd.DataFrame(audit)
    assert len(A) == 48
    exceptions = A[~(A.same_frame_prefix & A.owner_wrapper_consistent & A.proposition_in_both & A.frame_text_found)]
    # ---- ANOVA per query/response, source and test separately
    an_rows, an_checks = [], {}
    for split, TT, wc in (("source", T[T.split == "source"], True), ("test", T[T.split == "test"], False)):
        for q in QUERIES6:
            for resp in ("ent", "con"):
                y = TT[f"{q}:{resp}"].to_numpy(float)
                rows, chk = anova(TT, y, wc)
                for r in rows:
                    an_rows.append({"split": split, "query": q, "response": resp, **r})
                an_checks[f"{split}|{q}|{resp}"] = chk
    AN = pd.DataFrame(an_rows)
    # ---- wording contrasts
    W = []
    for (core, form, owner, frame), g in T.groupby(["core", "form", "owner", "frame"]):
        o = g[g.wording == "original"].iloc[0]; p = g[g.wording == "paraphrase"].iloc[0]
        W.append({"core": core, "split": o.split, "form": form, "owner": owner, "frame": frame,
                  **{f"d_{q}:{r}": float(o[f"{q}:{r}"] - p[f"{q}:{r}"]) for q in QUERIES6 for r in ("ent", "con")}})
    Wd = pd.DataFrame(W)
    dsum = {}
    for q in QUERIES6:
        col = f"d_{q}:ent"
        dsum[q] = {f: {"mean_d": float(Wd[Wd.form == f][col].mean()), "mean_d2": float((Wd[Wd.form == f][col] ** 2).mean()),
                       "max_abs_d": float(Wd[Wd.form == f][col].abs().max())} for f in FORMS}
    D = Wd[[f"d_{q}:ent" for q in QUERIES6]].to_numpy()
    cq = {"queries": QUERIES6, "cov": np.cov(D.T).round(6).tolist(), "corr": pd.DataFrame(D).corr().round(4).to_numpy().tolist()}
    # ---- merged functional sensitivity (packet-16 channel, K3, delta at N=16)
    src = T[T.split == "source"]; qs = ch.VARIANTS["V1"]
    S_, shift2, dfn = ch.tolerance(src, qs); delta = dfn(16)
    cols = ch.columns("K3"); a = ch.functionals("K3", cols)["pi_epi_plus_non"]
    chans = {"pooled_source": ch.fit_mu(src, qs), "source_original_only": ch.fit_mu(src[src.wording == "original"], qs),
             "source_paraphrase_only": ch.fit_mu(src[src.wording == "paraphrase"], qs)}
    sens = []
    for cname, mu in chans.items():
        M = mu[cols].T
        for core in (1, 2, 3):
            for w in ("original", "paraphrase"):
                coll = T[(T.core == core) & (T.wording == w)]
                assert len(coll) == 16 and coll.cell.nunique() == 16
                mbar = ch.feats(coll, qs).mean(0)
                est = float(ch.simplex_ls(M, mbar, delta) @ a)
                sens.append({"channel": cname, "core": core, "wording": w, "in_sample_for_channel": bool(core in (1, 2) and (cname == "pooled_source" or cname.split("_")[1] == w)),
                             "estimate_merged": est, "truth": .5, "error": est - .5})
    SE = pd.DataFrame(sens)
    diffs = SE.pivot_table(index=["channel", "core"], columns="wording", values="estimate_merged")
    diffs["orig_minus_para"] = diffs["original"] - diffs["paraphrase"]
    # ---- summaries
    key = AN[(AN.response == "ent")].pivot_table(index=["split", "query"], columns="term", values="share").round(4)
    res = {"protocol_sha256": PROTOCOL_SHA, "readings_sha256": ch.READ_SHA, "reader_calls": 0, "label": "designed wording-perturbation facet (2 fixed template wordings per cell; one deterministic reader)",
           "pair_audit": {"pairs": int(len(A)), "exceptions": exceptions[["core", "form", "owner", "frame"]].to_dict(orient="records"), "dropped": 0,
                          "template_pairs": mats["forms"]},
           "anova_checks": an_checks, "max_abs_sum_check": float(max(abs(v["sum_of_terms_minus_total"]) for v in an_checks.values())),
           "max_abs_orthogonality_gap": float(max(abs(v["orthogonality_gap"]) for v in an_checks.values())),
           "ss_share_p_ent": {f"{s}|{q}": row.dropna().to_dict() for (s, q), row in key.iterrows()},
           "within_pair_d_p_ent_by_form": dsum, "cross_query_wording_difference": cq,
           "merged_functional_sensitivity": SE.to_dict(orient="records"),
           "merged_orig_minus_para": {f"{c}|core{k}": float(v) for (c, k), v in diffs["orig_minus_para"].items()},
           "delta_N16": delta.round(6).tolist(), "seconds": time.time() - t0}
    A.to_csv(OUT / "pair_audit.csv", index=False); AN.to_csv(OUT / "anova_rows.csv", index=False); Wd.to_csv(OUT / "wording_contrast_rows.csv", index=False)
    SE.to_csv(OUT / "merged_functional_sensitivity.csv", index=False)
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    (OUT / "runtime.json").write_text(json.dumps({"finished": datetime.now(timezone.utc).isoformat(), "seconds": res["seconds"], "reader_calls": 0,
                                                  "code_sha256": sha(Path(__file__)), "protocol_sha256": PROTOCOL_SHA,
                                                  "channel_code_sha256": sha(ROOT / "scripts/run_grain_r15_hedge_channel_utilization.py")}, indent=1) + "\n")
    print(json.dumps({k: res[k] for k in ("pair_audit", "max_abs_sum_check", "max_abs_orthogonality_gap", "merged_orig_minus_para")}, indent=1, default=float)[:3000])


if __name__ == "__main__":
    main()
