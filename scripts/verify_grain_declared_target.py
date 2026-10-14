#!/usr/bin/env python3
"""Same-agent alternative recomputation for GRAIN-C3-01 (writes only into the C3 root)."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import declared_target as DT
from grain_research.gptpro_foundation_v3.f04_recovery import score_response

OUT = ROOT / "results/grain_construction_v1/C3/declared_target_r1"
CR = ROOT / "results/grain_vector_mechanisms/all_plan_closure_v1/conditional_reference_r2"
chk = {"kind": "same-agent alternative computation (not third-party)"}
b = json.loads((OUT / "source_fitted_bundle.json").read_text())
chk["bundle_sha_matches_brief"] = hashlib.sha256((OUT / "source_fitted_bundle.json").read_bytes()).hexdigest() == json.loads((OUT / "construction_brief.json").read_text())["bundle_sha256"]
units = np.load(CR / "pandora/private_units.npz")
y = units["scores"].reshape(-1, 36, 30)
addr = pd.read_csv(ROOT / "results/grain_gptpro_foundation_v3/F16/event_addresses_r1/private_pandora_event_addresses.csv", usecols=["author_index", "unit", "condition"]).sort_values(["author_index", "unit"])
venue = addr.condition.to_numpy().reshape(-1, 36)
c = np.where(np.isin(venue, b["context_rule"]["top"]), 1.0, -1.0)
roles = json.loads((ROOT / "results/grain_gptpro_foundation_v3/F14/pandora_venue_oom_r1/author_roles.json").read_text())
held = np.array(roles["held"])
# BLUP vs F04 score_response (covariance form) for 5 held authors, facet 0
diffs = []
for a in held[:5]:
    fit = {"mode": "linear", "mu": b["fits"][0]["mu"], "Sigma": b["fits"][0]["Sigma"], "sigma2": b["fits"][0]["sigma2"]}
    pre = [{"z": float(c[a][u]), "y": float(y[a][u][0]), "event_id": f"{a}_{u}"} for u in range(25)]
    ref = score_response(pre, fit, conditions=(-1.0, 1.0))["coefficients"]
    mine, _ = DT.blup(b["fits"][0], y[a][:25, 0], c[a][:25])
    diffs.append(np.max(np.abs(ref - mine)))
chk["blup_vs_f04_score_response_max_abs_diff"] = float(max(diffs))
# E1 independent recomputation (relative to global MSE) for held authors
src = np.array(roles["source_train"])
cold = pd.read_csv(OUT / "private_cold_scores.csv")
rows, worst = [], 0.0
ck = pd.read_csv(OUT / "empirical_checks.csv")
for j in range(30):
    g = {s: y[src][:, :25][c[src][:, :25] == s][:, j].mean() for s in (1.0, -1.0)}
    cj = cold[cold.facet == j].set_index("author_index")
    eb, eg, n = 0.0, 0.0, 0
    for a in held:
        for s, col in ((1.0, "m_plus"), (-1.0, "m_minus")):
            m = c[a][25:] == s
            if m.any():
                obs = y[a][25:][m, j].mean()
                eb += (cj.loc[a, col] - obs) ** 2
                eg += (g[s] - obs) ** 2
                n += 1
    imp = (eg - eb) / n
    saved = ck[(ck.role == "held") & (ck.facet == j) & (ck.statistic == "E1_improvement_vs_global")].value.iloc[0]
    worst = max(worst, abs(imp - saved))
    rows.append({"facet": j, "mse_global": eg / n, "mse_bundle": eb / n, "relative_improvement": imp / (eg / n)})
pd.DataFrame(rows).to_csv(OUT / "e1_relative_improvement.csv", index=False)
chk["E1_recompute_max_abs_diff"] = float(worst)
rel = pd.DataFrame(rows).relative_improvement
chk["E1_relative_improvement_range"] = [float(rel.min()), float(rel.median()), float(rel.max())]
acc = json.loads((OUT / "acceptance.json").read_text())
chk["cold_replay_max_abs_diff"] = acc["cold_replay_max_abs_diff"]
chk["evidence_trace_sum_check"] = json.loads((OUT / "evidence_trace.json").read_text())["max_sum_check_abs_diff"]
fr = pd.read_csv(OUT / "acceptance_by_facet.csv")
re_lvl = []
for j in range(30):
    g = ck[(ck.role == "held") & (ck.facet == j)].set_index("statistic")
    ok = g.loc["E1_improvement_vs_global", "fw_lower"] > 0 and g.loc["E3_level_split_half_corr", "fw_lower"] > 0
    ok &= g.loc["sd_level_scores", "value"] >= .1 * g.loc["sd_level_scores", "prior_sd"]
    re_lvl.append(bool(ok))
chk["acceptance_level_recompute_matches"] = re_lvl == fr.level_supported.tolist()
chk["pass"] = bool(chk["bundle_sha_matches_brief"] and chk["blup_vs_f04_score_response_max_abs_diff"] < 1e-9
                   and worst < 1e-12 and chk["cold_replay_max_abs_diff"] <= 1e-12 and chk["evidence_trace_sum_check"] < 1e-9
                   and chk["acceptance_level_recompute_matches"])
(OUT / "verification.json").write_text(json.dumps(chk, indent=1))
print(json.dumps(chk, indent=1))
