#!/usr/bin/env python
"""Frozen contrasts for the R15 hedge-commitment reader (protocol V1). Source flags use cores 1-2 rows only."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/grain_gptpro_foundation_v3/F17/hedge_commitment_reader_r1"
ROLES = ["own_commitment", "uncertainty", "deference", "reported_other", "no_commitment", "truth_diagnostic"]
EXPECT = {"own_commitment": "bare", "uncertainty": "epistemic", "deference": "deferential", "no_commitment": "noncommittal"}
FORMS = ["bare", "epistemic", "deferential", "noncommittal"]

df = pd.DataFrame([json.loads(l) for l in (OUT / "readings.jsonl").read_text().splitlines()])
assert len(df) == 576
res = {"pairs": int(len(df)), "note": "source-core flags computed after the run from cores 1-2 rows only; they govern interpretation, not reading"}


def form_table(sub):
    return sub[sub.owner == "self"].groupby(["form", "role"]).p_entailment.mean().unstack()


src = df[df.split == "source"]
ft_src = form_table(src)
own_self_bare = src[(src.owner == "self") & (src.form == "bare") & (src.role == "own_commitment")].p_entailment.mean()
own_other_bare = src[(src.owner == "other") & (src.form == "bare") & (src.role == "own_commitment")].p_entailment.mean()
rep_self = src[(src.owner == "self") & (src.role == "reported_other")].p_entailment.mean()
rep_other = src[(src.owner == "other") & (src.role == "reported_other")].p_entailment.mean()
flags = {"ownership_failure": bool(own_other_bare >= own_self_bare or rep_other <= rep_self),
         "own_commitment_self_bare": float(own_self_bare), "own_commitment_other_bare": float(own_other_bare),
         "reported_other_self": float(rep_self), "reported_other_other": float(rep_other),
         "form_failure": {r: bool(ft_src[r].idxmax() != f) for r, f in EXPECT.items()}, "argmax_form_source": {r: ft_src[r].idxmax() for r in EXPECT}}
res["source_boundary_flags"] = flags
per_core = {}
for core, sub in df.groupby("core"):
    ft = form_table(sub)
    para = sub[sub.owner == "self"].pivot_table(index=["form", "frame", "role"], columns="wording", values="p_entailment")
    para_var = (para["original"] - para["paraphrase"]).abs().groupby(level=["form", "role"]).mean().unstack()
    effects = {r: {f: float(ft.loc[f, r] - ft.loc["bare", r]) for f in FORMS if f != "bare"} for r in ROLES}
    own = sub.pivot_table(index=["form", "frame", "wording", "role"], columns="owner", values="p_entailment")
    owner_delta = (own["other"] - own["self"]).groupby(level=["form", "role"]).mean().unstack()
    frame = sub.pivot_table(index=["form", "owner", "wording", "role"], columns="frame", values="p_entailment")
    frame_delta = (frame["disagreeing"] - frame["neutral"]).groupby(level="role").mean()
    per_core[int(core)] = {"split": sub.split.iloc[0],
                           "mean_entailment_self_by_form_role": ft.round(4).to_dict(),
                           "form_minus_bare": {r: {f: round(v, 4) for f, v in d.items()} for r, d in effects.items()},
                           "paraphrase_abs_variation_by_form_role": para_var.round(4).to_dict(),
                           "expected_form_is_max": {r: bool(ft[r].idxmax() == f) for r, f in EXPECT.items()},
                           "argmax_form": {r: ft[r].idxmax() for r in EXPECT},
                           "reported_minus_self_by_form": owner_delta[["own_commitment", "reported_other", "truth_diagnostic"]].round(4).to_dict(),
                           "disagreeing_minus_neutral_by_role": frame_delta.round(4).to_dict(),
                           "truth_diagnostic_by_owner_form": sub[sub.role == "truth_diagnostic"].groupby(["owner", "form"]).p_entailment.mean().round(4).unstack().to_dict()}
for core, sub in df.groupby("core"):
    s_ = sub[sub.owner == "self"]
    marg = {}
    for r, f in EXPECT.items():
        m = s_[s_.role == r].groupby("form").p_entailment.mean()
        rival = m.drop(f).idxmax()
        pv = s_[s_.role == r].pivot_table(index=["form", "frame"], columns="wording", values="p_entailment")
        pv = (pv["original"] - pv["paraphrase"]).abs().groupby(level="form").mean()
        marg[r] = {"expected": f, "rival": rival, "margin": round(float(m[f] - m[rival]), 4),
                   "paraphrase_var_expected": round(float(pv[f]), 4), "paraphrase_var_rival": round(float(pv[rival]), 4),
                   "margin_exceeds_both_paraphrase_vars": bool(m[f] - m[rival] > max(pv[f], pv[rival]))}
    per_core[int(core)]["expected_vs_rival_margin"] = marg
res["per_core"] = per_core
# POST HOC (after reading; descriptive only, no rescue): frame split, because the disagreeing frame names a
# third party holding a view, which is lexical evidence for the reported_other query.
ph = df[df.role.isin(["own_commitment", "reported_other", "deference", "no_commitment"])]
res["posthoc_frame_split"] = {
    "note": "post hoc, descriptive; the disagreeing frame introduces another person's view (material confound found after reading)",
    "table": ph.groupby(["role", "form", "owner", "frame"]).p_entailment.mean().round(4).reset_index().to_dict(orient="records")}
m3 = df[["p_entailment", "p_neutral", "p_contradiction"]].max(axis=1)
res["ambiguous_rows_max_prob_below_0.6"] = {"count": int((m3 < .6).sum()),
                                           "pair_ids": df.loc[m3 < .6, "pair_id"].tolist()}
res["mean_probs_by_role_form_owner"] = df.groupby(["role", "form", "owner"])[["p_entailment", "p_neutral", "p_contradiction"]].mean().round(4).reset_index().to_dict(orient="records")
(OUT / "contrasts.json").write_text(json.dumps(res, indent=1) + "\n")
print(json.dumps(res["source_boundary_flags"], indent=1))
for c, v in per_core.items():
    print("== core", c, v["split"], "argmax", v["argmax_form"], "expected_max", v["expected_form_is_max"])
    print("   own_commitment by form (self):", {f: v["mean_entailment_self_by_form_role"]["own_commitment"][f] for f in FORMS})
    print("   uncertainty:", {f: v["mean_entailment_self_by_form_role"]["uncertainty"][f] for f in FORMS})
    print("   deference:", {f: v["mean_entailment_self_by_form_role"]["deference"][f] for f in FORMS})
    print("   no_commitment:", {f: v["mean_entailment_self_by_form_role"]["no_commitment"][f] for f in FORMS})
    print("   reported-self own_commit:", v["reported_minus_self_by_form"]["own_commitment"], " reported_other:", v["reported_minus_self_by_form"]["reported_other"])
    print("   truth diag:", v["truth_diagnostic_by_owner_form"])
    print("   frame delta:", v["disagreeing_minus_neutral_by_role"])
    print("   paraphrase var (own_commitment):", v["paraphrase_abs_variation_by_form_role"]["own_commitment"])
