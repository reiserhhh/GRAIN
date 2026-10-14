#!/usr/bin/env python3
"""Same-agent alternative recomputation for GRAIN-C3-02 (r1 and within-venue supplement)."""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import declared_target as DT

C3 = ROOT / "results/grain_construction_v1/C3"
roles = json.loads((ROOT / "results/grain_gptpro_foundation_v3/F14/pandora_venue_oom_r1/author_roles.json").read_text())
src, held = np.array(roles["source_train"]), np.array(roles["held"])
com = np.load(ROOT / "results/grain_vector_mechanisms/predecessor_integration_v1/material/private_arrays.npz")["comments"].astype(np.float64)
nrm = com / np.linalg.norm(com, axis=2, keepdims=True)
addr = pd.read_csv(ROOT / "results/grain_gptpro_foundation_v3/F16/event_addresses_r1/private_pandora_event_addresses.csv", usecols=["author_index", "unit", "condition"]).sort_values(["author_index", "unit"])
venue = addr.condition.to_numpy().reshape(-1, 36)
H = list(range(25))
chk = {"kind": "same-agent alternative computation (pandas groupby / explicit loops)"}
for root, wv in (("anonymous_evidence_r1", False), ("anonymous_evidence_r1_within_venue", True)):
    d = np.load(C3 / "anonymous_evidence_r1/private_learned_directions.npz")["response_stability"][:, 1]
    ev = nrm @ d
    if wv:
        vt, et = venue[src][:, H].ravel(), ev[src][:, H].ravel()
        cnt = Counter(vt.tolist())
        tab = pd.Series(et).groupby(vt).mean()
        sub = np.vectorize(lambda v: tab[v] if cnt.get(v, 0) >= 5 else et.mean())(venue)
        ev = ev - sub
    et = pd.DataFrame({"v": ev[src][:, H].ravel(), "venue": venue[src][:, H].ravel()})
    cnt = et.venue.value_counts()
    et["vg"] = np.where(et.venue.map(cnt) >= 5, et.venue, "__pooled__")
    r2 = 1 - ((et.v - et.groupby("vg").v.transform("mean")) ** 2).sum() / ((et.v - et.v.mean()) ** 2).sum()
    diag = pd.read_csv(C3 / root / "direction_diagnostics.csv")
    saved = diag[(diag.family == "response_stability") & (diag.direction == 1)].venue_r2.iloc[0]
    top = json.loads((C3 / "declared_target_r1/source_fitted_bundle.json").read_text())["context_rule"]["top"]
    c = np.where(np.isin(venue, top), 1.0, -1.0)
    fits, _ = DT.fit_moments([ev[a][H][:, None] for a in src], [c[a][H] for a in src])
    halves = [[DT.blup(fits[0], ev[a][hh], c[a][hh])[0][0] for a in held] for hh in (H[0::2], H[1::2])]
    r = float(np.corrcoef(halves[0], halves[1])[0, 1])
    ck = pd.read_csv(C3 / root / "empirical_checks.csv")
    saved_r = ck[(ck.family == "response_stability") & (ck.role == "held") & (ck.direction == 1) & (ck.statistic == "E3_level_split_half_corr")].value.iloc[0]
    chk[root] = {"venue_r2_recompute_abs_diff": float(abs(r2 - saved)), "E3_level_recompute_abs_diff": float(abs(r - saved_r))}
    acc = json.loads((C3 / root / "acceptance.json").read_text())
    ad = pd.read_csv(C3 / root / "acceptance_by_direction.csv")
    learned = ad[ad.family != "keyed_baseline"]
    chk[root]["acceptance_flags_consistent"] = bool(acc["contrast_discovery"] == learned.contrast_supported.any()
                                                    and acc["anonymous_route_positive"] == (learned.level_supported | learned.contrast_supported).any())
chk["pass"] = all(v["venue_r2_recompute_abs_diff"] < 1e-9 and v["E3_level_recompute_abs_diff"] < 1e-9 and v["acceptance_flags_consistent"]
                  for k, v in chk.items() if isinstance(v, dict))
(C3 / "anonymous_evidence_r1_within_venue/verification.json").write_text(json.dumps(chk, indent=1))
print(json.dumps(chk, indent=1))
