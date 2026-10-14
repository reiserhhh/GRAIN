#!/usr/bin/env python3
"""Analyze the P05 native-action packet (no model calls). Primary reference: cached measured donor action."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_grain_p05_native_action import (
    CAP,
    OP,
    OUT,
    PROTOCOL_SHA,
    QUERIES,
    digest,
    load_panel,
    verify,
)

ARMS = ("E4", "J4", "E24", "J24", "RAND_J24")
CONTRASTS = (("J24", "E24", "metric at 24"), ("J4", "E4", "metric at 4"), ("J24", "J4", "representation under action loss"), ("E24", "E4", "representation under Euclidean loss"))
SEED, BOOT = 20260926, 2000


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")


def main():
    verify()
    arr, pairs, _ids = load_panel()
    src = arr["source"]
    eff = pd.DataFrame([json.loads(x) for x in (OUT / "effects.jsonl").read_text().splitlines()])
    old = pd.DataFrame([json.loads(x) for x in (OP / "native_effects.jsonl").read_text().splitlines()])
    donor = old[old.kind == "donor"].set_index(["pair", "query"])["effect"]
    hist = {k: old[old.kind == k].set_index(["pair", "query"])["effect"] for k in ("compiled", "random")}
    app = pairs[~src].reset_index(drop=True)
    contrast = {(p, q): float(arr["ds"][np.flatnonzero(pairs.pair == p)[0], qi]) for p in app.pair for qi, q in enumerate(QUERIES)}
    cell = {p: "|".join(p.split("|")[:3]) for p in app.pair}
    split = dict(zip(app.pair, app.split, strict=True))
    wide = eff.pivot_table(index=["pair", "query"], columns="kind", values="effect")
    if wide.shape[0] != 144 or set(ARMS) - set(wide.columns):
        raise SystemExit(f"incomplete effects: {wide.shape}")
    wide["donor"] = [donor[k] for k in wide.index]
    wide["contrast"] = [contrast[k] for k in wide.index]
    wide["hist_compiled_r2"] = [hist["compiled"][k] for k in wide.index]
    wide["hist_random_r2"] = [hist["random"][k] for k in wide.index]
    wide = wide.reset_index()
    wide["cell"] = wide.pair.map(cell)
    wide["split"] = wide.pair.map(split)
    res = {"per_query": {}, "contrasts": {}, "strata": {}, "diagnostics": {}}
    rng = np.random.default_rng(SEED)
    for q, g in wide.groupby("query"):
        d = g.donor.to_numpy()
        prev = max((d > 0).mean(), (d < 0).mean())
        rows = {"zero_rmse_vs_donor": rmse(0 * d, d), "donor_sign_prevalence_reference": float(prev)}
        for arm in (*ARMS, "hist_compiled_r2", "hist_random_r2"):
            a = g[arm].to_numpy()
            rows[arm] = {"rmse_vs_donor": rmse(a, d), "r_vs_donor": corr(a, d), "sign_agreement": float(np.mean(np.sign(a) == np.sign(d))),
                         "rmse_vs_full_contrast_secondary": rmse(a, g.contrast), "r_vs_full_contrast_secondary": corr(a, g.contrast)}
        res["per_query"][q] = rows
        cells = g.cell.to_numpy()
        uc = np.unique(cells)
        idx = [np.flatnonzero(cells == c) for c in uc]
        draws = [np.concatenate([idx[j] for j in rng.integers(0, len(uc), len(uc))]) for _ in range(BOOT)]
        for a1, a2, label in CONTRASTS:
            x1, x2 = g[a1].to_numpy(), g[a2].to_numpy()
            dr = [rmse(x1[b], d[b]) - rmse(x2[b], d[b]) for b in draws]
            dc = [corr(x1[b], d[b]) - corr(x2[b], d[b]) for b in draws]
            res["contrasts"][f"{q}|{a1}-{a2}"] = {"label": label, "delta_rmse": rmse(x1, d) - rmse(x2, d),
                                                   "delta_rmse_ci": [float(v) for v in np.nanpercentile(dr, [2.5, 97.5])],
                                                   "delta_r": corr(x1, d) - corr(x2, d),
                                                   "delta_r_ci": [float(v) for v in np.nanpercentile(dc, [2.5, 97.5])]}
        for arm in ARMS:
            a = g[arm].to_numpy()
            dz = [rmse(a[b], d[b]) - rmse(0 * d[b], d[b]) for b in draws]
            res["contrasts"][f"{q}|{arm}-ZERO"] = {"label": "arm vs zero reference", "delta_rmse": rmse(a, d) - rmse(0 * d, d),
                                                   "delta_rmse_ci": [float(v) for v in np.nanpercentile(dz, [2.5, 97.5])]}
        for s, gs in g.groupby("split"):
            ds_ = gs.donor.to_numpy()
            res["strata"][f"{q}|{s}"] = {"n": len(gs), "zero_rmse": rmse(0 * ds_, ds_),
                                         **{arm: {"rmse": rmse(gs[arm], ds_), "r": corr(gs[arm], ds_)} for arm in ARMS}}
    e = eff.set_index(["pair", "query", "kind"])
    res["diagnostics"]["action_norm_median"] = {f"{q}|{arm}": float(eff[(eff["query"] == q) & (eff.kind == arm)].action_norm.median()) for q in QUERIES for arm in ARMS}
    res["diagnostics"]["candidate_mass_min"] = {arm: float(eff[eff.kind == arm].mass.min()) for arm in ARMS}
    lin_path = OUT / "linearization.jsonl"
    if lin_path.exists():
        lin = pd.DataFrame([json.loads(x) for x in lin_path.read_text().splitlines()])
        res["diagnostics"]["J24_source_linearization"] = {
            q: {"n": len(g), "corr_predicted_actual": corr(g.predicted_linear, g.actual),
                "median_abs_rel_err": float(np.median(np.abs(g.actual - g.predicted_linear) / np.maximum(np.abs(g.predicted_linear), 1e-9))),
                "rmse_actual_vs_predicted": rmse(g.actual, g.predicted_linear), "actual_rms": rmse(g.actual, 0 * g.actual)}
            for q, g in lin.groupby("query")}
    acc = [json.loads(x) for x in (OUT / "accounting.jsonl").read_text().splitlines()]
    res["accounting"] = {"passes_attempted": sum(a.get("passes", 0) for a in acc if a["status"] == "STARTED"), "cap": CAP,
                         "failed": sum(a["status"] == "FAILED" for a in acc),
                         "by_step": pd.Series({a["step"]: 0 for a in acc if a["status"] == "STARTED"}).to_dict()}
    steps = {}
    for a in acc:
        if a["status"] == "STARTED":
            steps[a["step"]] = steps.get(a["step"], 0) + a["passes"]
    res["accounting"]["by_step"] = steps
    res["protocol_sha256"] = PROTOCOL_SHA
    res["code_sha256"] = digest(Path(__file__))
    res["created"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    wide.to_csv(OUT / "effects_wide.csv", index=False)
    del e
    print(json.dumps({"per_query": res["per_query"], "contrasts": res["contrasts"], "accounting": res["accounting"]}, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
