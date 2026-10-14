#!/usr/bin/env python3
"""Same-agent alternative recomputation for GRAIN-C4-01 (no text output)."""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import anonymous_bundle as AB
from grain_research.construction import declared_target as DT
from grain_research.gptpro_foundation_v3.f04_recovery import score_response

OUT = ROOT / "results/grain_construction_v1/C4/transport_r1"
MAT = ROOT / "results/grain_vector_mechanisms/predecessor_integration_v1/material"
chk = {"kind": "same-agent alternative computation (F04 covariance-form scorer, explicit R2 formula)"}
b = json.loads((OUT / "anonymous_bundle.json").read_text())
AB.verify_bundle(b)
chk["bundle_sha_matches_replay_record"] = hashlib.sha256((OUT / "anonymous_bundle.json").read_bytes()).hexdigest() == json.loads((OUT / "bundle_replay.json").read_text())["bundle_sha256"]
roles = json.loads((ROOT / "results/grain_gptpro_foundation_v3/F14/pandora_venue_oom_r1/author_roles.json").read_text())
src, va, held = (np.array(roles[k]) for k in ("source_train", "source_validation", "held"))
addr = pd.read_csv(ROOT / "results/grain_gptpro_foundation_v3/F16/event_addresses_r1/private_pandora_event_addresses.csv", usecols=["author_index", "unit", "start", "end", "condition"]).sort_values(["author_index", "unit"])
docs = json.loads((MAT / "private_material.json").read_text())["documents"]
texts = [docs[r.author_index][r.start:r.end] for r in addr.itertuples()]
del docs
H = list(range(25))
rows = np.arange(len(texts)).reshape(-1, 36)
s_rows = rows[src][:, H].ravel()
vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=5, max_features=20000, sublinear_tf=True, norm="l2").fit([texts[i] for i in s_rows])
X = vec.transform(texts)
del texts
nrm = AB.normalize(np.load(MAT / "private_arrays.npz")["comments"].astype(np.float64))
raw = (nrm @ np.asarray(b["directions"])).reshape(-1, 15)
ridge = Ridge(alpha=1.0).fit(X[s_rows], raw[s_rows])
pred = ridge.predict(X)
v_rows = rows[va][:, H].ravel()
r2 = 1 - ((raw[v_rows] - pred[v_rows]) ** 2).sum(0) / ((raw[v_rows] - raw[v_rows].mean(0)) ** 2).sum(0)
mv = pd.read_csv(OUT / "map_validation.csv")
saved = mv[(mv.unit == "anon") & (mv["mode"] == "composition_inclusive")].sort_values("coordinate").validation_r2.to_numpy()
chk["map_validation_r2_recompute_max_abs_diff"] = float(np.max(np.abs(r2 - saved)))
venue = addr.condition.to_numpy().reshape(-1, 36)
c = DT.contexts(venue.ravel(), b["context_rule"]).reshape(-1, 36)
j = 9
fit = {"mode": "linear", **{k: b["fits"]["composition_inclusive"][j][k] for k in ("mu", "Sigma", "sigma2")}}
lev = {}
for name, ev in (("e5", raw.reshape(-1, 36, 15)), ("lex", pred.reshape(-1, 36, 15))):
    lev[name] = np.array([score_response([{"z": float(c[a][u]), "y": float(ev[a][u][j]), "event_id": f"{a}_{u}"} for u in H], fit,
                                         conditions=(-1.0, 1.0))["coefficients"][0] for a in held])
agree = float(np.corrcoef(lev["e5"], lev["lex"])[0, 1])
th = pd.read_csv(OUT / "transport_held.csv")
saved_a = th[(th.unit == "anon") & (th["mode"] == "composition_inclusive") & (th.coordinate == j) & (th.arm == "lex_frozen")].agreement_level_vs_e5.iloc[0]
chk["agreement_recompute_abs_diff_coord9"] = abs(agree - saved_a)
chk["pass"] = bool(chk["bundle_sha_matches_replay_record"] and chk["map_validation_r2_recompute_max_abs_diff"] < 1e-9
                   and chk["agreement_recompute_abs_diff_coord9"] < 1e-9)
(OUT.parent / "transport_r1_uncertainty_layer" / "verification_c4_01.json").write_text(json.dumps(chk, indent=1))
print(json.dumps(chk, indent=1))
