#!/usr/bin/env python3
"""Packet 3 Part C (R21 amendment A1): E5 vs BGE gauge map across corpora, zero neural calls."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from scripts.run_grain_r21_heldout_map import PROTOCOL, digest, pdist

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SHA_A1 = "16e22582eece02b3adb4b4049d48d70505c4aa30b45ab01ba775aea40e7da9e3"
R17 = ROOT / "results/grain_gptpro_foundation_v3/R17/natural_sensitivity_r1"
LB = ROOT / "results/grain_vector_mechanisms/all_plan_closure_v1/local_bilinear_r2"
OUT = ROOT / "results/grain_gptpro_foundation_v3/F12/encoder_gauge_r1"
SEED, N_PERM = 20260926, 200


def procrustes(xs, ys):
    mx, my = xs.mean(0), ys.mean(0)
    u, _, vt = np.linalg.svd((xs - mx).T @ (ys - my), full_matrices=False)
    rot = u @ vt
    return lambda x: (x - mx) @ rot + my


def minnorm(xs, ys):
    w = np.linalg.lstsq(np.column_stack([np.ones(len(xs)), xs]), ys, rcond=None)[0]
    return lambda x: np.column_stack([np.ones(len(x)), x]) @ w


def cos_rows(a, b):
    return np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def load_items():
    m = pd.read_csv(R17 / "manifest.csv")
    items = {}
    for corpus in ("pandora", "essays"):
        mc = m[m.corpus == corpus]
        with np.load(LB / corpus / "private_units.npz", allow_pickle=False) as z:
            h = z["h"]
        e5, bge = [], []
        for rec in mc.itertuples():
            e5.append(np.load(R17 / "private_vectors/e5" / f"{corpus}_{rec.author_index}_base.npy"))
            bge.append(np.load(R17 / "private_vectors/bge" / f"{corpus}_{rec.author_index}_base.npy"))
        for rec in mc.itertuples():
            e5.append(h[rec.partner_row])
            bge.append(np.load(R17 / "private_vectors/bge" / f"{corpus}_{rec.author_index}_partner.npy"))
        items[corpus] = {"E5": np.array(e5, float), "BGE": np.array(bge, float)}
    return items


def main():
    if digest(PROTOCOL) != PROTOCOL_SHA_A1:
        raise ValueError("Protocol changed after A1")
    OUT.mkdir(parents=True, exist_ok=True)
    items = load_items()
    rng = np.random.default_rng(SEED)
    res = {"protocol_sha256": PROTOCOL_SHA_A1, "n_per_corpus": {k: len(v["E5"]) for k, v in items.items()}, "maps": {}}
    for fit_c, test_c in (("pandora", "essays"), ("essays", "pandora")):
        for src, tgt in (("E5", "BGE"), ("BGE", "E5")):
            xs, ys = items[fit_c][src], items[fit_c][tgt]
            xh, yh = items[test_c][src], items[test_c][tgt]
            dt = pdist(yh)
            med = float(np.median(dt))
            unm = float(spearmanr(pdist(xh), dt).statistic)
            for mname, maker in (("procrustes", procrustes), ("minnorm_ols", minnorm)):
                f = maker(xs, ys)
                pred = f(xh)
                err = np.linalg.norm(pred - yh, axis=1)
                nr, ns = [], []
                for _ in range(N_PERM):
                    fp = maker(xs[rng.permutation(len(xs))], ys)
                    pp = fp(xh)
                    nr.append(spearmanr(pdist(pp), dt).statistic)
                    ns.append(np.linalg.norm(pp - yh, axis=1).max() / med)
                entry = {
                    "heldout_mean_cos": float(cos_rows(pred, yh).mean()),
                    "rdm_spearman_mapped": float(spearmanr(pdist(pred), dt).statistic),
                    "rdm_spearman_unmapped": unm,
                    "rdm_null_q95": float(np.percentile(nr, 95)),
                    "sup_over_median_between": float(err.max() / med),
                    "q95_over_median_between": float(np.percentile(err, 95) / med),
                    "null_sup_q05": float(np.percentile(ns, 5)),
                }
                entry["functional_agreement"] = bool(entry["rdm_spearman_mapped"] > entry["rdm_null_q95"]
                                                     and entry["sup_over_median_between"] < 1.0)
                res["maps"][f"fit_{fit_c}_test_{test_c}|{src}_to_{tgt}|{mname}"] = entry
    res["code_sha256"] = digest(Path(__file__))
    res["neural_calls"] = 0
    res["created"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    (OUT / "encoder_gauge.json").write_text(json.dumps(res, indent=1) + "\n")
    print(pd.DataFrame(res["maps"]).T.round(3).to_string())


if __name__ == "__main__":
    main()
