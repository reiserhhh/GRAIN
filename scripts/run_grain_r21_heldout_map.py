#!/usr/bin/env python3
"""Packet 3 Part B (R21): source-fitted map between cached representations, held out by template."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/GRAIN_R18_R21_PACKET3_PROTOCOL_V1.md"
PROTOCOL_SHA = "5b72f2dd3f678b9e7690d8107147ba8479181479fe068e67cd29f4bd7ada6128"  # V1 (Part B ran under V1)
TT = ROOT / "results/grain_gptpro_foundation_v3/F16/text_transport_full_r1"
OUT = ROOT / "results/grain_gptpro_foundation_v3/F12/heldout_map_r1"
SEED, N_PERM = 20260926, 200


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pdist(x):
    d = np.sqrt(np.maximum(((x[:, None, :] - x[None, :, :]) ** 2).sum(-1), 0))
    return d[np.triu_indices(len(x), 1)]


def model_upper(labels):
    lab = np.asarray(labels)
    m = (lab[:, None] != lab[None, :]).astype(float)
    return m[np.triu_indices(len(lab), 1)]


def fit(x, y):
    X = np.column_stack([np.ones(len(x)), x])
    return np.linalg.lstsq(X, y, rcond=None)[0]


def apply(w, x):
    return np.column_stack([np.ones(len(x)), x]) @ w


def evaluate(xs, ys, xh, yh, rng):
    w = fit(xs, ys)
    pred = apply(w, xh)
    sst = ((yh - ys.mean(0)) ** 2).sum()
    r2 = 1 - ((yh - pred) ** 2).sum() / sst
    dt = pdist(yh)
    rdm_mapped = spearmanr(pdist(pred), dt).statistic
    rdm_unmapped = spearmanr(pdist(xh), dt).statistic
    err = np.linalg.norm(pred - yh, axis=1)
    med = float(np.median(dt))
    null_rdm, null_sup = [], []
    for _ in range(N_PERM):
        wp = fit(xs[rng.permutation(len(xs))], ys)
        pp = apply(wp, xh)
        null_rdm.append(spearmanr(pdist(pp), dt).statistic)
        null_sup.append(np.linalg.norm(pp - yh, axis=1).max() / med)
    return pred, {
        "heldout_r2": float(r2),
        "rdm_spearman_mapped": float(rdm_mapped),
        "rdm_spearman_unmapped": float(rdm_unmapped),
        "rdm_null_q95": float(np.percentile(null_rdm, 95)),
        "sup_distance_over_median_between": float(err.max() / med),
        "q95_distance_over_median_between": float(np.percentile(err, 95) / med),
        "median_distance_over_median_between": float(np.median(err) / med),
        "null_sup_q05": float(np.percentile(null_sup, 5)),
        "n_fit": len(xs), "n_heldout": len(xh),
    }


def main():
    if digest(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("Protocol changed")
    OUT.mkdir(parents=True, exist_ok=True)
    mats = {r["material_id"]: r for r in csv.DictReader((TT / "materials.csv").open())}
    ids = [r["material_id"] for r in csv.DictReader((TT / "reader_scores.csv").open())]
    grain = np.load(TT / "grain_features.npz")["features"].astype(float)
    nli = defaultdict(dict)
    for r in csv.DictReader((TT / "nli_probabilities.csv").open()):
        mid, hyp = r["audit_id"].rsplit(":", 1)
        nli[mid][hyp] = [float(r["p_contradiction"]), float(r["p_entailment"]), float(r["p_neutral"])]
    nli_x = np.array([[v for h in ("positive", "negative", "qualification") for v in nli[m][h]] for m in ids])
    template = np.array([m.rsplit("_", 2)[0] for m in ids])
    temps = np.unique(template)
    perm = np.random.default_rng(SEED).permutation(temps)
    half = {t: i % 2 for i, t in enumerate(perm)}
    fold = np.array([half[t] for t in template])
    labels = {k: np.array([mats[m][k] for m in ids]) for k in ("pole", "subject_status")}
    rng = np.random.default_rng(SEED + 1)
    reps = {"GRAIN": grain, "NLI": nli_x}
    out = {"protocol_sha256": PROTOCOL_SHA, "n_items": len(ids), "n_templates": len(temps),
           "fold_sizes": [int((fold == 0).sum()), int((fold == 1).sum())], "maps": {}}
    for src_name, tgt_name in (("GRAIN", "NLI"), ("NLI", "GRAIN")):
        x, y = reps[src_name], reps[tgt_name]
        folds = {}
        for f in (0, 1):
            fit_i, held = fold != f, fold == f
            pred, m = evaluate(x[fit_i], y[fit_i], x[held], y[held], rng)
            for lab, v in labels.items():
                mu = model_upper(v[held])
                m[f"mapped_rsa_{lab}"] = float(spearmanr(pdist(pred), mu).statistic)
                m[f"target_rsa_{lab}"] = float(spearmanr(pdist(y[held]), mu).statistic)
                m[f"source_rsa_{lab}"] = float(spearmanr(pdist(x[held]), mu).statistic)
            m["functional_agreement"] = bool(m["rdm_spearman_mapped"] > m["rdm_null_q95"]
                                             and m["sup_distance_over_median_between"] < 1.0)
            folds[f"heldout_fold_{f}"] = m
        out["maps"][f"{src_name}_to_{tgt_name}"] = folds
    out["code_sha256"] = digest(Path(__file__))
    out["inputs"] = {n: digest(TT / n) for n in ("materials.csv", "reader_scores.csv", "grain_features.npz", "nli_probabilities.csv")}
    out["neural_calls"] = 0
    out["created"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    (OUT / "heldout_map.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["maps"], indent=1))


if __name__ == "__main__":
    main()
