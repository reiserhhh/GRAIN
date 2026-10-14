#!/usr/bin/env python
"""R08 observational PANDORA history comparison (docs/GRAIN_R08_PANDORA_HISTORY_COMPARISON_PROTOCOL_V1.md)."""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/GRAIN_R08_PANDORA_HISTORY_COMPARISON_PROTOCOL_V1.md"
PROTOCOL_SHA = "d2f43a193537994765b30a1e52f8f970ff658f76492a79a75073e0b2ad62cee0"
OUT = ROOT / "results/grain_gptpro_foundation_v3/F14/pandora_history_comparison_r1"
EA = ROOT / "results/grain_gptpro_foundation_v3/F16/event_addresses_r1/private_pandora_event_addresses.csv"
REFD = ROOT / "results/grain_vector_mechanisms/all_plan_closure_v1/conditional_reference_r2/pandora"
PINS = {EA: "61c98ca01fc7acd067ef924611009b2e9ca2cf5d9888a4514a91f6cfb4a10a7e", REFD / "private_units.npz": "184887cb4962c8872b670557a988fbaf85e0bef696a97466cd9836f4a5d30a89",
        REFD / "private_addresses.csv": "bb622a49ee48fd282eb085885e1c031e9734e687a659c8fdb5997c500f09cddf"}
FACETS = [f"{d}{i}" for d in "ACENO" for i in range(1, 7)]
K_TOP = 20


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load():
    for p, h in PINS.items():
        if sha(p) != h:
            raise RuntimeError(f"pinned input changed: {p}")
    ea = pd.read_csv(EA)
    ad = pd.read_csv(REFD / "private_addresses.csv").reset_index().rename(columns={"index": "score_row"})
    with np.load(REFD / "private_units.npz", allow_pickle=False) as u:
        scores, authors, conds = u["scores"], u["authors"], u["conditions"].astype(str)
    if not (np.array_equal(ad.author_index.to_numpy(), authors) and np.array_equal(ad.condition.astype(str).to_numpy(), conds)):
        raise RuntimeError("score rows do not match reference addresses")
    m = ea.merge(ad, on=["author_index", "unit"], suffixes=("", "_ad"), validate="one_to_one")
    ok = len(m) == len(ea) and (m.start == m.start_ad).all() and (m.end == m.end_ad).all() and (m.condition == m.condition_ad).all()
    if not ok:
        raise RuntimeError("row identity failed")
    m = m.sort_values(["author_index", "timestamp_unix_seconds"]).reset_index(drop=True)
    X = scores[m.score_row.to_numpy()]
    return m, X, {"rows": int(len(m)), "one_to_one": True, "start_end_venue_equal": True}


def main() -> int:
    if sha(PROTOCOL) != PROTOCOL_SHA:
        raise RuntimeError("protocol sha mismatch")
    if OUT.exists():
        raise FileExistsError("pandora_history_comparison_r1 exists")
    OUT.mkdir(parents=True)
    t0 = time.time()
    m, X, ident = load()
    m["pos"] = m.groupby("author_index").cumcount()
    src_auth = sorted(m[m.split == "source"].author_index.unique())
    val_auth = set(a for j, a in enumerate(src_auth) if j % 5 == 0)
    train_auth = set(src_auth) - val_auth
    held_auth = sorted(m[m.split != "source"].author_index.unique())
    tr = m.author_index.isin(train_auth).to_numpy()
    top = m[tr].condition.value_counts().index[:K_TOP].tolist()
    cat = {v: i for i, v in enumerate(top)}
    m["cat"] = m.condition.map(lambda v: cat.get(v, K_TOP))
    C = K_TOP + 1
    g = np.bincount(m[tr].cat, minlength=C) + 1.0; g /= g.sum()
    T = np.ones((C, C))
    for a, d in m[tr].groupby("author_index"):
        c = d.cat.to_numpy()
        np.add.at(T, (c[:-1], c[1:]), 1)
    T /= T.sum(1, keepdims=True)
    mu, sd = X[tr].mean(0), X[tr].std(0)
    Z = (X - mu) / sd
    by_author = {a: np.where(m.author_index.to_numpy() == a)[0] for a in m.author_index.unique()}
    means = np.array([Z[by_author[a]].mean(0) for a in sorted(train_auth)])
    U, s, Vt = np.linalg.svd(means - means.mean(0), full_matrices=False)
    V5, cmean = Vt[:5], means.mean(0)
    num, den = np.zeros(30), np.zeros(30)
    for a in train_auth:
        ix = by_author[a]
        for t in range(2, len(ix)):
            pm = Z[ix[:t]].mean(0)
            dl, dn = Z[ix[t - 1]] - pm, Z[ix[t]] - pm
            num += dl * dn; den += dl * dl
    phi = num / den

    def venue_preds(ix, t, alpha, w):
        prev = m.cat.to_numpy()[ix[:t]]
        cnt = np.bincount(prev, minlength=C)
        static = (cnt + alpha * g) / (cnt.sum() + alpha)
        o1 = T[prev[-1]]
        return {"global": g, "static": static, "order1": o1, "combined": w * o1 + (1 - w) * static}

    def keyed_preds(ix, t, w):
        pm = Z[ix[:t]].mean(0)
        ar = pm + phi * (Z[ix[t - 1]] - pm)
        pca = cmean + (pm - cmean) @ V5.T @ V5
        return {"global": np.zeros(30), "static": pm, "pca5": pca, "ar1": ar, "combined": pm + w * (ar - pm)}

    def evaluate(auths, alpha, wv, wk, collect=False):
        rows = []
        cats, raw = m.cat.to_numpy(), m.condition.to_numpy()
        ts = m.timestamp_unix_seconds.to_numpy()
        for a in auths:
            ix = by_author[a]
            for t in range(25, len(ix)):
                y = cats[ix[t]]
                vp = venue_preds(ix, t, alpha, wv)
                kp = keyed_preds(ix, t, wk)
                r = {"author": a, "t": t, "novel_raw_venue": raw[ix[t]] not in set(raw[ix[:t]]), "other_cat": y == K_TOP,
                     "gap_days": (ts[ix[t]] - ts[ix[t - 1]]) / 86400}
                for k, p in vp.items():
                    r[f"venue_ll_{k}"] = -np.log(p[y]); r[f"venue_brier_{k}"] = float(((p - np.eye(C)[y]) ** 2).sum())
                for k, p in kp.items():
                    r[f"keyed_mse_{k}"] = float(((p - Z[ix[t]]) ** 2).mean())
                rows.append(r)
        return pd.DataFrame(rows)
    grid = {}
    for alpha in (1, 3, 10, 30, 100):
        grid[alpha] = evaluate(sorted(val_auth), alpha, 0.0, 0.0)["venue_ll_static"].mean()
    alpha = min(grid, key=grid.get)
    wgrid = np.round(np.arange(0, 1.01, .1), 1)
    vv = {w: evaluate(sorted(val_auth), alpha, w, 0.0)["venue_ll_combined"].mean() for w in wgrid}
    kk = {w: evaluate(sorted(val_auth), alpha, 0.0, w)["keyed_mse_combined"].mean() for w in wgrid}
    wv, wk = min(vv, key=vv.get), min(kk, key=kk.get)
    df = evaluate(held_auth, alpha, wv, wk)
    df.to_csv(OUT / "held_event_scores.csv", index=False)
    rng = np.random.default_rng(20260926)
    per = df.groupby("author").mean(numeric_only=True)
    ids = per.index.to_numpy()
    draws = rng.integers(0, len(ids), size=(2000, len(ids)))

    def delta(col, ref):
        d = (per[col] - per[ref]).to_numpy()
        bs = d[draws].mean(1)
        return {"mean": float(d.mean()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    venue = {k: {"logloss": float(df[f"venue_ll_{k}"].mean()), "brier": float(df[f"venue_brier_{k}"].mean())} for k in ("global", "static", "order1", "combined")}
    for k in ("global", "order1", "combined"):
        venue[k]["delta_vs_static"] = delta(f"venue_ll_{k}", "venue_ll_static")
    keyed = {k: {"mse": float(df[f"keyed_mse_{k}"].mean())} for k in ("global", "static", "pca5", "ar1", "combined")}
    for k in ("global", "pca5", "ar1", "combined"):
        keyed[k]["delta_vs_static"] = delta(f"keyed_mse_{k}", "keyed_mse_static")
    split = {}
    for name, mask in (("known_raw_venue", ~df.novel_raw_venue), ("novel_raw_venue", df.novel_raw_venue)):
        split[name] = {"events": int(mask.sum()), **{f"venue_ll_{k}": float(df.loc[mask, f"venue_ll_{k}"].mean()) for k in ("static", "order1", "combined")},
                       **{f"keyed_mse_{k}": float(df.loc[mask, f"keyed_mse_{k}"].mean()) for k in ("static", "ar1")}}
    df["gap_tercile"] = pd.qcut(df.gap_days, 3, labels=["short", "mid", "long"])
    gaps = df.groupby("gap_tercile", observed=True)[["gap_days", "venue_ll_static", "venue_ll_order1", "keyed_mse_static", "keyed_mse_ar1"]].mean().round(5).to_dict(orient="index")
    res = {"row_identity": ident, "facets": FACETS, "roles": {"source_train_authors": len(train_auth), "source_validation_authors": len(val_auth), "held_authors": len(held_auth),
                                                            "held_scored_events": int(len(df))},
           "chosen_on_source_validation": {"alpha": alpha, "w_venue": float(wv), "w_keyed": float(wk)}, "phi_ar1_per_facet": dict(zip(FACETS, np.round(phi, 4).tolist())),
           "top20_share_of_held_scored": float((~df.other_cat).mean()), "venue": venue, "keyed": keyed, "novelty_split": split, "gap_terciles": {str(k): v for k, v in gaps.items()},
           "seconds": time.time() - t0, "neural_calls": 0}
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(json.dumps({"venue": venue, "keyed": keyed, "split": split, "chosen": res["chosen_on_source_validation"]}, indent=1, default=str)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
