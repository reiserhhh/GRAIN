#!/usr/bin/env python3
"""Same-agent alternative recomputation for GRAIN-C2-01; writes only into the new root."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/grain_construction_v1/C2/evidence_unit_r1"
chk = {"kind": "same-agent alternative formulas (not third-party)"}
cold = pd.read_csv(OUT / "cold_scores.csv")
truth = pd.read_csv(OUT / "evaluator_truth.csv")
down = pd.read_csv(OUT / "downstream_comparison.csv")
down = down[down["section"].isna()] if "section" in down else down
m = cold.merge(truth, on=["seed", "author_index"], how="left", validate="many_to_one")
worst = 0.0
for (reader, arm, role, kappa), g in m.groupby(["reader", "arm", "family_role", "kappa"]):
    err = g.pi_mean.to_numpy() - g.pi_true.to_numpy()
    rec = {"mse": float(np.dot(err, err) / len(err)), "bias": float(err.sum() / len(err)),
           "coverage": float(((g.pi_true >= g.pi_lo) & (g.pi_true <= g.pi_hi)).sum() / len(g)),
           "mean_width": float((g.pi_hi - g.pi_lo).sum() / len(g)),
           "r": float(np.cov(g.pi_mean, g.pi_true)[0, 1] / np.sqrt(np.var(g.pi_mean, ddof=1) * np.var(g.pi_true, ddof=1)))}
    row = down[(down.reader == reader) & (down.arm == arm) & (down.family_role == role) & (down.kappa == kappa)].iloc[0]
    worst = max(worst, *(abs(rec[k] - row[k]) for k in rec))
chk["downstream_recompute_max_abs_diff"] = float(worst)
held = pd.read_csv(OUT / "held_material_checks.csv")
law = json.loads((OUT / "observation_law.json").read_text())
st = ["actual", "quoted", "hypothetical", "negated"]
cnt_worst = 0.0
for reader in law:
    if reader in ("bundles_sha256", "lexical_insample_source"):
        continue
    for role in ("validation", "application") + (("source",) if reader.startswith("nli") else ()):
        h = held[(held.reader == reader) & (held.role == role)]
        n = np.zeros((2, 4, 4))
        for r in h.itertuples():
            n[int(r.pole), st.index(r.subject_status), int(r.category)] += 1
        cnt_worst = max(cnt_worst, float(np.max(np.abs(n - np.array(law[reader][role]["counts"])))))
chk["channel_counts_recompute_max_abs_diff"] = cnt_worst
chk["bundle_hashes_match"] = all(
    hashlib.sha256((OUT / f).read_bytes()).hexdigest() == law["bundles_sha256"][k]
    for k, f in (("lexical_tfidf_f08", "bundle_lexical.json"), ("nli_deberta_v3_base_6c749ce", "bundle_nli.json"),
                 ("lexical_insample", "bundle_lexical_insample_ablation.json")))
rep = json.loads((OUT / "cold_replay.json").read_text())
chk["cold_replay_max_abs_diff"] = max(v["max_abs_diff_vs_warm"] for v in rep.values())
chk["no_evidence_author_prior_only"] = all(v["no_evidence_author"]["status"] == "no_events" for v in rep.values())
chk["pass"] = bool(worst < 1e-9 and cnt_worst == 0 and chk["bundle_hashes_match"] and chk["cold_replay_max_abs_diff"] <= 1e-12
                   and chk["no_evidence_author_prior_only"])
(OUT / "verification.json").write_text(json.dumps(chk, indent=1))
print(json.dumps(chk, indent=1))
