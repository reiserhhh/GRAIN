#!/usr/bin/env python3
"""Additive C2/C3 interpretation addenda (new roots only; old roots untouched)."""

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
C2 = ROOT / "results/grain_construction_v1/C2/evidence_unit_r1"
C3 = ROOT / "results/grain_construction_v1/C3/declared_target_r1"
OUT2 = ROOT / "results/grain_construction_v1/C2/evidence_unit_r1_addendum_20261003"
OUT3 = ROOT / "results/grain_construction_v1/C3/declared_target_r1_addendum_20261003"


def root_hash_check(root):
    saved = json.loads((root / "output_hashes.json").read_text())
    return all(hashlib.sha256((root / f).read_bytes()).hexdigest() == h for f, h in saved.items())


p = pd.read_csv(C2 / "paired_differences.csv")
p = p[(p.reference == "naive_uncalibrated") & (p.family_role == "application")]
rows = []
for r in p.itertuples():
    rows.append({"reader": r.reader, "kappa": r.kappa, "mse_primary_minus_naive": r.mse_primary_minus_ref,
                 "ci_lo": r.ci_lo, "ci_hi": r.ci_hi,
                 "legacy_P4_screen_ci_lo_le_0": bool(r.ci_lo <= 0),
                 "zero_margin_upper_bound_lt_0 (primary better than naive)": bool(r.ci_hi < 0),
                 "lower_bound_gt_0 (primary worse than naive)": bool(r.ci_lo > 0)})
OUT2.mkdir(parents=True, exist_ok=False)
c2 = {"date": "2026-10-03", "old_root_hashes_intact": root_hash_check(C2), "rows": rows,
      "wording": {
          "P4": "The frozen P4 rule (paired CI lower bound <= 0) is a legacy screen meaning 'not significantly worse than naive'. It is not a non-inferiority test; non-inferiority needs an upper bound below a declared margin. Frozen results and verdicts are unchanged.",
          "zero_margin_reading": "Using the saved intervals: NLI calibrated MSE minus naive has upper bounds below 0 at every share (significantly better). Lexical has lower bounds above 0 at shares 1 and .5 (significantly worse) and an upper bound below 0 at share .25.",
          "posterior": "The calibrated arms average per-draw normalized posteriors over source-only channel draws: a source-frozen modular (cut-posterior) working uncertainty, not a full joint Bayesian marginalization that would update channel weights with application data."}}
(OUT2 / "c2_interpretation_addendum.json").write_text(json.dumps(c2, indent=1))
OUT3.mkdir(parents=True, exist_ok=False)
c3 = {"date": "2026-10-03", "old_root_hashes_intact": root_hash_check(C3),
      "wording": {"E1": "C3-01 E1 compares predicted future venue-class means (m_plus, m_minus) jointly with observed later comments. It is conditional-prediction evidence for the level-plus-contrast model, not a test of the level (intercept) target alone and not psychological validity. Level evidence (split-half), conditional prediction (E1) and meaning are separate columns.",
                  "anonymous_bundle": "C3-02 had frozen directions but no standalone cold bundle; C4-01 transport_r1/anonymous_bundle.json now carries all source-fitted transformations."}}
(OUT3 / "c3_interpretation_addendum.json").write_text(json.dumps(c3, indent=1))
print(json.dumps({"c2_old_root_intact": c2["old_root_hashes_intact"], "c3_old_root_intact": c3["old_root_hashes_intact"],
                  "rows": [(r["reader"][:4], r["kappa"], r["zero_margin_upper_bound_lt_0 (primary better than naive)"], r["lower_bound_gt_0 (primary worse than naive)"]) for r in rows]}, indent=1))
