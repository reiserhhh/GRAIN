#!/usr/bin/env python3
"""Criterion preservation (seminar-05 queued item 2): exploratory reuse of cached labels, no fitting to labels."""

from __future__ import annotations

import collections
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from grain_research.vector_mechanisms.external_heads import TRAITS
from scripts.run_grain_r21_encoder_gauge import load_items, minnorm, procrustes

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/GRAIN_CRITERION_PRESERVATION_PROTOCOL_V1.md"
PROTOCOL_SHA = "edaf74e602cf7e68c97e4b3c42b66e6814a97855939dea33e333dec657f02c3e"
LB = ROOT / "results/grain_vector_mechanisms/all_plan_closure_v1/local_bilinear_r2"
R17 = ROOT / "results/grain_gptpro_foundation_v3/R17/natural_sensitivity_r1"
OUT = ROOT / "results/grain_gptpro_foundation_v3/F12/criterion_preservation_r1"
SEED, BOOT = 20260926, 2000


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def labels(corpus):
    if corpus == "pandora":
        from scripts.prepare_grain_predecessor_reference import DEFAULT
        with np.load(DEFAULT / "private_arrays.npz", allow_pickle=False) as z:
            return z["y"].astype(float)
    from scripts.run_grain_local_construction import ESSAYS
    return pd.read_csv(ESSAYS, usecols=list(TRAITS))[list(TRAITS)].to_numpy(float)


def head_scores(corpus, authors, rows):
    obj = joblib.load(LB / corpus / "private_model_semantic.joblib")
    with np.load(LB / corpus / "private_blocks.npz", allow_pickle=False) as z:
        blocks = {k: z[k] for k in z.files}
    out = []
    for a, row in zip(authors, rows, strict=True):
        b = dict(blocks)
        b["semantic"] = blocks["semantic"].copy()
        b["semantic"][a] = row
        out.append(obj["head"].predict(obj["features"].transform(b, None, [int(a)]))[0])
    return np.array(out)


def rcol(p, y):
    return np.array([np.corrcoef(p[:, t], y[:, t])[0, 1] for t in range(y.shape[1])])


def paired(p1, p2, y, rng):
    n = len(y)
    idx = rng.integers(0, n, (BOOT, n))
    d = np.array([rcol(p2[i], y[i]) - rcol(p1[i], y[i]) for i in idx])
    return np.percentile(d, [2.5, 97.5], axis=0)


def part_a(rng):
    items = load_items()
    m = pd.read_csv(R17 / "manifest.csv")
    res = {}
    for corpus, other in (("pandora", "essays"), ("essays", "pandora")):
        authors = m[m.corpus == corpus].author_index.to_numpy()
        n = len(authors)
        e5, bge = items[corpus]["E5"], items[corpus]["BGE"]
        e5_mean = (e5[:n] + e5[n:]) / 2
        y = labels(corpus)[authors]
        missing = int(np.isnan(y).any(axis=1).sum())
        p_e5 = head_scores(corpus, authors, e5_mean)
        out = {"n_authors": n, "authors_with_missing_labels": missing}
        for mname, maker in (("minnorm_primary", minnorm), ("procrustes", procrustes)):
            f = maker(items[other]["BGE"], items[other]["E5"])
            mapped = (f(bge[:n]) + f(bge[n:])) / 2
            p_map = head_scores(corpus, authors, mapped)
            r1, r2 = rcol(p_e5, y), rcol(p_map, y)
            ci = paired(p_e5, p_map, y, rng)
            out[mname] = {t: {"r_e5": float(r1[k]), "r_mapped_bge": float(r2[k]), "delta_r": float(r2[k] - r1[k]),
                              "delta_ci": [float(ci[0, k]), float(ci[1, k])],
                              "prediction_agreement": float(np.corrcoef(p_e5[:, k], p_map[:, k])[0, 1])}
                          for k, t in enumerate(TRAITS)}
        res[corpus] = out
    return res


def part_b(rng):
    from scripts.prepare_grain_predecessor_reference import DEFAULT
    material = json.loads((DEFAULT / "private_material.json").read_text())
    subs = [[c["subreddit"] for v in b["views"] for c in v] for b in material["bundles"]]
    with np.load(LB / "pandora/private_predictions.npz", allow_pickle=False) as z:
        held, pred = z["held_indices"], z["semantic"].astype(float)
    counts = collections.Counter(s for i in held for s in subs[int(i)])
    top = [s for s, _ in counts.most_common(30)]
    share = np.array([[sum(x == s for x in subs[int(i)]) / 36 for s in top] + [sum(x not in top for x in subs[int(i)]) / 36] for i in held])
    u, sv, _ = np.linalg.svd(share - share.mean(0), full_matrices=False)
    X = np.column_stack([np.ones(len(held)), u[:, :5] * sv[:5]])
    resid = np.empty_like(pred)
    for t in range(pred.shape[1]):
        for i in range(len(held)):
            mk = np.ones(len(held), bool)
            mk[i] = False
            resid[i, t] = pred[i, t] - X[i] @ np.linalg.lstsq(X[mk], pred[mk, t], rcond=None)[0]
    y = labels("pandora")[held]
    r1, r2 = rcol(pred, y), rcol(resid, y)
    ci = paired(pred, resid, y, rng)
    return {"n_authors": len(held), "authors_with_missing_labels": int(np.isnan(y).any(axis=1).sum()),
            **{t: {"r_original": float(r1[k]), "r_topic_residualized": float(r2[k]), "delta_r": float(r2[k] - r1[k]),
                   "delta_ci": [float(ci[0, k]), float(ci[1, k])]} for k, t in enumerate(TRAITS)}}


def main():
    if digest(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("Protocol changed")
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    out = {"protocol_sha256": PROTOCOL_SHA, "eligibility": {"F12/heldout_map_r1": "INCOMPATIBLE (synthetic items, no labels)",
                                                             "F12/encoder_gauge_r1": "ELIGIBLE", "F10/shortcut_r1 + FU1-c": "ELIGIBLE"},
           "part_a_encoder_map": part_a(rng), "part_b_topic_residualization": part_b(rng),
           "code_sha256": digest(Path(__file__)), "neural_calls": 0, "labels": "existing cached public-corpus labels; exploratory reuse",
           "created": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    (OUT / "criterion_preservation.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1)[:5000])


if __name__ == "__main__":
    main()
