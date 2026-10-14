#!/usr/bin/env python3
"""A08 r2: separate-control sensitivity for the reader RSA (cached data only).

Follows the 03:35 supervisor findings on rsa_readers_r1:
- r1's partial RSA conditioned on ONE ranked sum of variant+template RDMs;
  here each control is entered separately and jointly (two regressors).
- Mantel p-values are one-sided (upper tail). p = 1.0 is not evidence of zero
  nuisance sensitivity; material permutation preserves RDM symmetry but assumes
  exchangeability of related templates (a design assumption, not tested).
- Scope: the 24-d F16 historical lexical GRAIN reader and 9 NLI probability
  coordinates on the same 380 synthetic items; not every GRAIN representation,
  not construct identity.
No neural calls; reuses r1's RDM definitions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_grain_gptpro_rsa_readers as r1


def residualize(y, controls):
    x = np.column_stack([np.ones(len(y))] + list(controls))
    return y - x @ np.linalg.lstsq(x, y, rcond=None)[0]


def partial(a, b, controls):
    ra, rb = rankdata(a), rankdata(b)
    rc = [rankdata(c) for c in controls]
    return float(np.corrcoef(residualize(ra, rc), residualize(rb, rc))[0, 1])


def main():
    """Write rsa_sensitivity.json."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import csv
    from collections import defaultdict

    mats = {r["material_id"]: r for r in csv.DictReader((r1.TT / "materials.csv").open())}
    ids = [r["material_id"] for r in csv.DictReader((r1.TT / "reader_scores.csv").open())]
    grain = np.load(r1.TT / "grain_features.npz")["features"]
    nli = defaultdict(dict)
    for r in csv.DictReader((r1.TT / "nli_probabilities.csv").open()):
        mid, hyp = r["audit_id"].rsplit(":", 1)
        nli[mid][hyp] = [float(r["p_contradiction"]), float(r["p_entailment"]), float(r["p_neutral"])]
    nli_x = np.array([nli[m]["positive"] + nli[m]["negative"] + nli[m]["qualification"] for m in ids])
    systems = {"GRAIN_features": r1.upper(r1.rdm_from_vectors(grain)), "NLI_probabilities": r1.upper(r1.rdm_from_vectors(nli_x))}
    models = {k: r1.upper(r1.model_rdm([mats[m][k] for m in ids])) for k in ("pole", "qualification", "subject_status", "variant")}
    models["template"] = r1.upper(r1.model_rdm([m.rsplit("_", 2)[0] for m in ids]))
    out = {"n_items": len(ids), "n_pair_entries": len(systems["GRAIN_features"]),
           "note": "pair entries are dependent (shared items); not independent samples", "partial": {}}
    for s, sv in systems.items():
        for m in ("pole", "qualification", "subject_status"):
            out["partial"][f"{s}~{m}"] = {
                "raw": r1.spearman(sv, models[m]),
                "control_variant_only": partial(sv, models[m], [models["variant"]]),
                "control_template_only": partial(sv, models[m], [models["template"]]),
                "control_both_separately": partial(sv, models[m], [models["variant"], models["template"]]),
                "r1_control_ranked_sum": partial(sv, models[m], [models["variant"] + models["template"]]),
            }
    (args.output / "rsa_sensitivity.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
