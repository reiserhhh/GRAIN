#!/usr/bin/env python3
"""A08 Representational Similarity Analysis across reader systems (P04 pp.7,13,24).

Frozen before running (2026-09-26, Claude Opus 5.5). No new forwards: the 380
F16 text-transport materials with cached (i) GRAIN features (24-d) and (ii) NLI
probabilities (3 hypotheses x 3 classes = 9-d). Model RDMs from the design:
pole, qualification (ownership/actuality), subject status, variant (nuisance),
template (nuisance). RDMs: 1 - Pearson correlation between z-scored material
vectors (system RDMs); 0/1 different-label (model RDMs). Statistics: Spearman
correlation of upper triangles; Mantel p-value by 2000 material permutations
(seed 20260926); partial Spearman of system RDM vs pole/qualification RDMs
controlling for the variant+template nuisance RDM. RSA describes shared
relational geometry; it is not a mechanism or validity claim (reverse
inference caution, P04 p.7).
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
TT = ROOT / "results/grain_gptpro_foundation_v3/F16/text_transport_full_r1"


def rdm_from_vectors(x):
    z = (x - x.mean(0)) / np.where(x.std(0) > 0, x.std(0), 1)
    return 1 - np.corrcoef(z)


def model_rdm(labels):
    lab = np.asarray(labels)
    return (lab[:, None] != lab[None, :]).astype(float)


def upper(m):
    i, j = np.triu_indices(len(m), 1)
    return m[i, j]


def spearman(a, b):
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def partial_spearman(a, b, c):
    ra, rb, rc = rankdata(a), rankdata(b), rankdata(c)
    def resid(y, x):
        x1 = np.column_stack([np.ones_like(x), x])
        return y - x1 @ np.linalg.lstsq(x1, y, rcond=None)[0]
    return float(np.corrcoef(resid(ra, rc), resid(rb, rc))[0, 1])


def mantel(m1, m2, rng, perms=2000):
    obs = spearman(upper(m1), upper(m2))
    n = len(m1)
    count = 0
    r2 = rankdata(upper(m2))
    for _ in range(perms):
        p = rng.permutation(n)
        if np.corrcoef(rankdata(upper(m1[np.ix_(p, p)])), r2)[0, 1] >= obs:
            count += 1
    return obs, (count + 1) / (perms + 1)


def main():
    """Write rsa.json."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    mats = {r["material_id"]: r for r in csv.DictReader((TT / "materials.csv").open())}
    ids = [r["material_id"] for r in csv.DictReader((TT / "reader_scores.csv").open())]
    grain = np.load(TT / "grain_features.npz")["features"]
    nli = defaultdict(dict)
    for r in csv.DictReader((TT / "nli_probabilities.csv").open()):
        mid, hyp = r["audit_id"].rsplit(":", 1)
        nli[mid][hyp] = [float(r["p_contradiction"]), float(r["p_entailment"]), float(r["p_neutral"])]
    nli_x = np.array([sum((nli[m][h] for h in ("positive", "negative", "qualification")), []) for m in ids])
    systems = {"GRAIN_features": rdm_from_vectors(grain), "NLI_probabilities": rdm_from_vectors(nli_x)}
    lab = {k: [mats[m][k] for m in ids] for k in ("pole", "qualification", "subject_status", "variant")}
    template = [m.rsplit("_", 2)[0] for m in ids]
    models = {k: model_rdm(v) for k, v in lab.items()}
    models["template"] = model_rdm(template)
    nuisance = upper(models["variant"]) + upper(models["template"])
    rng = np.random.default_rng(20260926)
    out = {"n_materials": len(ids), "system_vs_system": {}, "system_vs_model": {}, "partial_controlling_nuisance": {}}
    r, p = mantel(systems["GRAIN_features"], systems["NLI_probabilities"], rng)
    out["system_vs_system"]["GRAIN_vs_NLI"] = {"spearman": r, "mantel_p": p}
    for sname, s in systems.items():
        for mname in ("pole", "qualification", "subject_status", "variant", "template"):
            r, p = mantel(s, models[mname], rng, perms=500)
            out["system_vs_model"][f"{sname}~{mname}"] = {"spearman": r, "mantel_p": p}
        for mname in ("pole", "qualification", "subject_status"):
            out["partial_controlling_nuisance"][f"{sname}~{mname}|nuisance"] = partial_spearman(upper(s), upper(models[mname]), nuisance)
    (args.output / "rsa.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
