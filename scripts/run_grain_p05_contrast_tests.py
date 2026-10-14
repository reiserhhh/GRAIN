#!/usr/bin/env python3
"""P05 task A / R27 C1 zero-forward tests (docs/GRAIN_P05_CONTRAST_LEDGER_PROTOCOL_V1.md)."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/GRAIN_P05_CONTRAST_LEDGER_PROTOCOL_V1.md"
PROTOCOL_SHA = "5d68a680854b1ddb75aa6ce55bf0b5742086bfa3bd78b6fc4005b6740cc95211"
OP = ROOT / "results/grain_vector_mechanisms/conditional_operator_v1/r2"
OUT = ROOT / "results/grain_gptpro_foundation_v3/P05/contrast_ledger_r1"
QUERIES = ("extraversion", "neuroticism")
SEED, N_PERM = 20260915, 500


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ols0(x, y):
    return np.linalg.lstsq(x, y, rcond=None)[0]


def r(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def loco(x, y, cells):
    pred = np.zeros_like(y)
    for c in np.unique(cells):
        test = cells == c
        pred[test] = x[test] @ ols0(x[~test], y[~test])
    return pred


def main():
    if digest(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("Protocol changed")
    OUT.mkdir(parents=True, exist_ok=True)
    arrays = np.load(OP / "action_arrays.npz", allow_pickle=False)
    order = json.loads((OP / "pair_order.json").read_text())
    pairs = pd.read_csv(OP / "pairs.csv").set_index("pair").loc[order].reset_index()
    dx, dh, ds, pred_state = arrays["dx"], arrays["dh"], arrays["ds"], arrays["predicted"]
    src = arrays["source"]
    if not np.array_equal(src, (pairs.split == "source").to_numpy()):
        raise ValueError("Split mismatch")
    shapes = []
    for q in QUERIES:
        key = hashlib.sha256(f"{pairs.high_id[0]}|{q}".encode()).hexdigest()
        with np.load(OP / "states" / f"{key}.npz") as z:
            shapes.append(z["head"].shape)
    offset = np.cumsum([0] + [int(np.prod(s)) for s in shapes])
    if offset[-1] != dh.shape[1]:
        raise ValueError("Segment layout mismatch")
    eff = pd.DataFrame([json.loads(x) for x in (OP / "native_effects.jsonl").read_text().splitlines()])
    app = ~src
    ap = pairs[app].reset_index(drop=True)
    cells = (ap.axis + "|" + ap.family + "|" + ap.situation).to_numpy()
    rng = np.random.default_rng(SEED)
    perms = [rng.permutation(len(ap)) for _ in range(N_PERM)]
    out = {"protocol_sha256": PROTOCOL_SHA, "segments": [list(s) for s in shapes],
           "n_application": int(app.sum()), "n_source": int(src.sum()),
           "n_cells": len(np.unique(cells)), "queries": {}}
    for qi, q in enumerate(QUERIES):
        e = eff[eff["query"] == q].set_index(["pair", "kind"])["effect"]
        D = np.array([e[(p, "donor")] for p in ap.pair])
        A = np.array([e[(p, "compiled")] for p in ap.pair])
        full = ds[app, qi]
        X = dx[app]
        res = {}
        for name, y in (("donor_action", D), ("full_contrast", full)):
            pr = loco(X, y, cells)
            null = [r(loco(X[pi], y, cells), y) for pi in perms]
            res[f"T1_{name}"] = {
                "loco_r": r(pr, y), "loco_rmse": float(np.sqrt(np.mean((pr - y) ** 2))),
                "zero_rmse": float(np.sqrt(np.mean(y ** 2))),
                "null_q95": float(np.nanpercentile(null, 95)),
                "null_mean": float(np.nanmean(null)),
                "p_perm": float((1 + np.sum(np.asarray(null) >= r(pr, y))) / (1 + N_PERM)),
            }
        sl = slice(offset[qi], offset[qi + 1])
        dh_q, a_q = dh[app, sl], pred_state[app, sl]
        eu = 1 - np.sum((dh_q - a_q) ** 2) / np.sum(dh_q ** 2)
        ac = 1 - np.sum((D - A) ** 2) / np.sum(D ** 2)
        per_pair_eu = 1 - np.sum((dh_q - a_q) ** 2, 1) / np.sum(dh_q ** 2, 1)
        proj = np.sum(dh_q * a_q, 1) / np.maximum(np.sum(a_q ** 2, 1), 1e-30)
        res["T2_metric"] = {
            "euclidean_capture_pooled": float(eu),
            "action_capture_pooled": float(ac),
            "difference": float(eu - ac),
            "euclidean_capture_median_pair": float(np.median(per_pair_eu)),
            "cos_dh_a_median": float(np.median(np.sum(dh_q * a_q, 1) / (
                np.linalg.norm(dh_q, axis=1) * np.linalg.norm(a_q, axis=1)))),
            "norm_ratio_a_over_dh_median": float(np.median(
                np.linalg.norm(a_q, axis=1) / np.linalg.norm(dh_q, axis=1))),
            "projection_coefficient_median": float(np.median(proj)),
            "compiled_vs_donor_action_r": r(A, D),
        }
        w = ols0(dx[src], ds[src, qi])
        pa = dx[app] @ w
        res["T3_transfer"] = {
            "source_fit_application_r": r(pa, full),
            "source_fit_application_rmse": float(np.sqrt(np.mean((pa - full) ** 2))),
            "loco_application_r": res["T1_full_contrast"]["loco_r"],
            "source_fit_on_donor_action_r": r(pa, D),
            "source_in_sample_r": r(dx[src] @ w, ds[src, qi]),
        }
        out["queries"][q] = res
    rem = pd.DataFrame([json.loads(x) for x in (OP / "remainder_effects.jsonl").read_text().splitlines()])
    out["rival_N_cited"] = {
        q: {"interaction_rms": float(np.sqrt(np.mean(g.interaction ** 2))),
            "donor_rms": float(np.sqrt(np.mean(g.donor ** 2))),
            "remainder_rms": float(np.sqrt(np.mean(g.residual ** 2)))}
        for q, g in rem.groupby("query")}
    tests = {}
    for q, res in out["queries"].items():
        t1 = res["T1_donor_action"]
        tests[q] = {
            "T1_rival_I_linear": "SUPPORTED" if not t1["loco_r"] > t1["null_q95"] else "NOT_SUPPORTED",
            "T2_M_star_metric": "SUPPORTED" if res["T2_metric"]["difference"] >= 0.10 else "NOT_SUPPORTED",
            "T3_rival_T": "SUPPORTED" if (res["T3_transfer"]["loco_application_r"]
                                         - res["T3_transfer"]["source_fit_application_r"]) >= 0.20
            else "NOT_SUPPORTED",
        }
    out["rule_outcomes"] = tests
    out["inputs"] = {p.name: digest(p) for p in (OP / "action_arrays.npz", OP / "pairs.csv",
                                                  OP / "native_effects.jsonl", OP / "remainder_effects.jsonl")}
    out["code_sha256"] = digest(Path(__file__))
    out["neural_forwards"] = 0
    out["created"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    (OUT / "c1_tests.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("queries", "rival_N_cited", "rule_outcomes")}, indent=1))


if __name__ == "__main__":
    main()
