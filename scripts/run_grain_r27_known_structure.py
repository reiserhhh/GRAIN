#!/usr/bin/env python3
"""R27/R36 known-structure (planted choice) benchmark on real PANDORA comments."""

from __future__ import annotations

import collections
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/GRAIN_R27_KNOWN_STRUCTURE_BENCHMARK_PROTOCOL_V1.md"
PROTOCOL_SHA = "c0ba7342310e7f539f5c1e54d60e8eece1912a6aff784c53efd3d869382c7e9f"
LB = ROOT / "results/grain_vector_mechanisms/all_plan_closure_v1/local_bilinear_r2/pandora"
OUT = ROOT / "results/grain_gptpro_foundation_v3/F14/known_structure_r1"
SEED, NA, NC, TOP, REPS = 20260926, 400, 36, 50, 5


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def recover(x, z, seed):
    xc = x - x.mean(0)
    u, _s, _ = np.linalg.svd(xc, full_matrices=False)
    pc1 = abs(float(np.corrcoef(u[:, 0], z)[0, 1]))
    pred = cross_val_predict(Ridge(alpha=1.0), x, z, cv=KFold(5, shuffle=True, random_state=seed))
    return pc1, float(np.corrcoef(pred, z)[0, 1])


def main():
    if digest(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("Protocol changed")
    OUT.mkdir(parents=True, exist_ok=True)
    from scripts.prepare_grain_predecessor_reference import DEFAULT

    material = json.loads((DEFAULT / "private_material.json").read_text())
    subs = np.array([c["subreddit"] for b in material["bundles"] for v in b["views"] for c in v])
    with np.load(LB / "private_units.npz", allow_pickle=False) as z:
        g, h = z["g"].astype(float), z["h"].astype(float)
    if len(subs) != len(g):
        raise ValueError("Subreddit/unit mismatch")
    freq = collections.Counter(subs.tolist())
    top = [s for s, _ in freq.most_common(TOP)]
    pools = {s: np.flatnonzero(subs == s) for s in top}
    logf = np.log(np.array([freq[s] for s in top], float))
    gmean = {s: g[pools[s]].mean(0) for s in top}
    rows = []
    for beta in (0.0, 0.5, 1.0):
        for rep in range(REPS):
            rng = np.random.default_rng(np.random.SeedSequence([SEED, int(beta * 10), rep]))
            w = rng.standard_normal(TOP)
            w = (w - w.mean()) / w.std()
            zz = rng.standard_normal(NA)
            M1, M2, M3, M4 = [], [], [], []
            for u in range(NA):
                logits = logf + beta * zz[u] * w
                p = np.exp(logits - logits.max())
                p /= p.sum()
                chosen = rng.choice(TOP, NC, p=p)
                idx, sidx = [], []
                used = collections.defaultdict(set)
                for k in chosen:
                    pool = pools[top[k]]
                    j = int(pool[rng.integers(len(pool))])
                    while j in used[k] and len(used[k]) < len(pool):
                        j = int(pool[rng.integers(len(pool))])
                    used[k].add(j)
                    idx.append(j)
                    sidx.append(k)
                idx = np.array(idx)
                M1.append(np.bincount(sidx, minlength=TOP) / NC)
                M2.append(g[idx].mean(0))
                M3.append(h[idx].mean(0))
                M4.append((g[idx] - np.array([gmean[top[k]] for k in sidx])).mean(0))
            for name, x in (("M1_choice_composition", M1), ("M2_grain_mean", M2), ("M3_e5_mean", M3), ("M4_decentred_grain", M4)):
                pc1, sup = recover(np.array(x), zz, SEED + rep)
                rows.append({"beta": beta, "rep": rep, "measure": name, "pc1_abs_r": pc1, "supervised_cv_r": sup})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "recovery.csv", index=False)
    summ = df.groupby(["measure", "beta"])[["pc1_abs_r", "supervised_cv_r"]].agg(["mean", "std"]).round(4)
    null = df[df.beta == 0].groupby("measure").supervised_cv_r.quantile(0.95)
    top_b = df[df.beta == 1.0].groupby("measure").supervised_cv_r.mean()
    rule = {m: {"beta1_mean": float(top_b[m]), "null_q95": float(null[m]),
                "recovers": bool(top_b[m] - null[m] >= 0.10)} for m in top_b.index}
    out = {"protocol_sha256": PROTOCOL_SHA, "code_sha256": digest(Path(__file__)), "pool_comments": len(g),
           "top_subreddits": TOP, "rule": rule, "neural_calls": 0, "labels_read": False,
           "created": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    (OUT / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    print(summ.to_string())
    print(json.dumps(rule, indent=1))


if __name__ == "__main__":
    main()
