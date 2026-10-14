#!/usr/bin/env python3
"""Seminar-05 item 4: paired CRN adaptive-design comparison and H01 final-design recovery (simulation)."""

from __future__ import annotations

import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np
from scipy.special import expit
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/GRAIN_SEMINAR05_ITEM4_PROTOCOL_V1.md"
PROTOCOL_SHA = "919f7a530af9a0127845aa23e5d25917123fd2b25c2ffc275c98d4431a5cab58"
OUT = ROOT / "results/grain_gptpro_foundation_v3/F17/seminar05_item4_r1"
SEED, BOOT, THRESH = 20260926, 2000, (0.90, 0.95, 0.97)


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def entropy(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def part_a():
    cells = np.array(list(itertools.product([0, 1], repeat=5)), float)
    B, e, r, k, g = cells.T
    P = expit(-1 + 1.5 * np.column_stack([1 - B, e, r, B * k, g]))
    rng = np.random.default_rng(SEED)
    truths = np.repeat(np.arange(5), 200)
    U = rng.random((len(truths), 60))
    res = {}
    per = {}
    for policy in ("FIXED", "SCALAR_CAT", "RIVAL_INFO"):
        stop = {t: np.full(len(truths), np.nan) for t in THRESH}
        mis_stop = {t: np.full(len(truths), np.nan) for t in THRESH}
        mis_budget = np.zeros(len(truths))
        for n, truth in enumerate(truths):
            post = np.full(5, 0.2)
            for t in range(60):
                if policy == "FIXED":
                    c = t % 32
                else:
                    pbar = P @ post
                    if policy == "SCALAR_CAT":
                        c = int(np.argmax(pbar * (1 - pbar)))
                    else:
                        gains = [entropy(post) - pbar[ci] * entropy(post * P[ci] / pbar[ci])
                                 - (1 - pbar[ci]) * entropy(post * (1 - P[ci]) / (1 - pbar[ci])) for ci in range(32)]
                        c = int(np.argmax(gains))
                y = U[n, t] < P[c, truth]
                post = post * (P[c] if y else 1 - P[c])
                post /= post.sum()
                for th in THRESH:
                    if np.isnan(stop[th][n]) and post.max() >= th:
                        stop[th][n] = t + 1
                        mis_stop[th][n] = float(np.argmax(post) != truth)
            mis_budget[n] = float(np.argmax(post) != truth)
        per[policy] = (stop, mis_stop, mis_budget)
        res[policy] = {"misclassification_at_budget": float(mis_budget.mean()),
                       **{f"th{th}": {"median_trials": float(np.nanmedian(stop[th])), "never_reached": float(np.isnan(stop[th]).mean()),
                                      "misclassification_at_stop": float(np.nanmean(mis_stop[th]))} for th in THRESH}}
    idx = np.random.default_rng(SEED + 1).integers(0, len(truths), (BOOT, len(truths)))
    diffs = {}
    for other in ("SCALAR_CAT", "FIXED"):
        a, b = per["RIVAL_INFO"][2], per[other][2]
        d = (a - b)[idx].mean(1)
        diffs[f"RIVAL_INFO-{other}|mis_at_budget"] = {"point": float((a - b).mean()), "ci": [float(x) for x in np.percentile(d, [2.5, 97.5])]}
        for th in THRESH:
            sa, sb = np.nan_to_num(per["RIVAL_INFO"][0][th], nan=61), np.nan_to_num(per[other][0][th], nan=61)
            dd = (sa - sb)[idx].mean(1)
            diffs[f"RIVAL_INFO-{other}|trials_th{th}(never=61)"] = {"point": float((sa - sb).mean()), "ci": [float(x) for x in np.percentile(dd, [2.5, 97.5])]}
    choice = {}
    for policy in res:
        ok = [th for th in THRESH if res[policy][f"th{th}"]["misclassification_at_stop"] <= 0.05]
        choice[policy] = min(ok) if ok else None
    return {"policies": res, "paired_differences": diffs, "smallest_threshold_with_stop_error_le_.05": choice}


def part_b():
    Z = np.array([-1.5, -0.5, 0.5, 1.5])
    D, rho, si, se, m = 30, 0.6, 0.5, 0.7, 0.5
    arm_var = {"A": 0.61, "B": 0.25, "C": 0.0}
    rng = np.random.default_rng(SEED + 2)
    rows = []
    for n in (200, 250, 300, 400):
        pw, rels, miss = [], [], []
        for _ in range(200):
            arms = np.array(["A", "B", "C"])[np.arange(n) % 3]
            b = rng.normal(0.5, 0.5, n)
            a = rng.standard_normal(n)
            s = np.zeros((n, D))
            s[:, 0] = rng.normal(0, si / np.sqrt(1 - rho ** 2), n)
            for d in range(1, D):
                s[:, d] = rho * s[:, d - 1] + rng.normal(0, si, n)
            M = (rng.random((n, D)) < 0.2).astype(float)
            para = rng.normal(0, np.sqrt(0.05), (n, 3))
            observed = (rng.random((n, D)) >= 0.15) & (rng.random((n, D)) >= expit(-2.5 + 0.8 * s))
            av = np.array([arm_var[x] for x in arms])
            slopes, half1, half2 = np.zeros(n), np.zeros(n), np.zeros(n)
            for u in range(n):
                days = np.flatnonzero(observed[u])
                y = (a[u] + b[u] * Z[None, :] + s[u, days][:, None] + para[u, days % 3][:, None] + m * M[u, days][:, None]
                     + rng.normal(0, np.sqrt(av[u]), (len(days), 1)) + rng.normal(0, se, (len(days), 4)))
                X = np.column_stack([np.ones(len(days) * 4), np.tile(Z, len(days)), np.repeat(M[u, days], 4)])
                yy = y.ravel()
                slopes[u] = np.linalg.lstsq(X, yy, rcond=None)[0][1]
                for hsel, arr_ in ((days % 2 == 0, half1), (days % 2 == 1, half2)):
                    Xh = np.column_stack([np.ones(hsel.sum() * 4), np.tile(Z, hsel.sum()), np.repeat(M[u, days[hsel]], 4)])
                    arr_[u] = np.linalg.lstsq(Xh, y[hsel].ravel(), rcond=None)[0][1]
            zs = np.zeros(n)
            for arm in "ABC":
                mk = arms == arm
                zs[mk] = (slopes[mk] - slopes[mk].mean()) / slopes[mk].std()
            zb = (b - b.mean()) / b.std()
            crit_true = 0.2 * zb + np.sqrt(1 - 0.04) * rng.standard_normal(n)
            crit_obs = np.sqrt(0.8) * crit_true + np.sqrt(0.2) * rng.standard_normal(n)
            r, p = pearsonr(zs, crit_obs)
            pw.append(p < 0.05 and r > 0)
            r12 = np.corrcoef(half1, half2)[0, 1]
            rels.append(2 * r12 / (1 + r12))
            miss.append(1 - observed.mean())
        rows.append({"persons": n, "power": float(np.mean(pw)), "slope_split_half_reliability": float(np.mean(rels)), "missing_share": float(np.mean(miss))})
    rec = next((r["persons"] for r in rows if r["power"] >= 0.80), None)
    return {"grid": rows, "recommended_persons": rec}


def main():
    if digest(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("Protocol changed")
    OUT.mkdir(parents=True, exist_ok=True)
    out = {"protocol_sha256": PROTOCOL_SHA, "part_a_paired_crn": part_a(), "part_b_h01_final_design": part_b(),
           "code_sha256": digest(Path(__file__)), "neural_calls": 0, "created": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    (OUT / "item4.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1)[:4000])


if __name__ == "__main__":
    main()
