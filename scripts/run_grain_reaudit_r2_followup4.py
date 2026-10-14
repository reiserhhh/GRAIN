#!/usr/bin/env python3
"""Re-audit r2 follow-up packet 4 (FU4-a..c): rival-information design, politeness RSA, concept bottleneck."""

from __future__ import annotations

import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/GRAIN_REAUDIT_R2_FOLLOWUP4_PROTOCOL_V1.md"
PROTOCOL_SHA = "6aa0f13c5d03a66c5796a398836af17ef3b9a8180152dae507f2225f7ebd46c7"
V3 = ROOT / "results/grain_gptpro_foundation_v3"
OUT = V3 / "F17/reaudit_r2_followup4"
SEED = 20260926


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def entropy(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def fu4_a():
    cells = np.array(list(itertools.product([0, 1], repeat=5)), float)
    B, e, r, k, g = cells.T
    X = np.column_stack([1 - B, e, r, B * k, g])
    P = expit(-1 + 1.5 * X)  # cells x rivals
    rng = np.random.default_rng(SEED)
    out = {}
    for policy in ("FIXED", "SCALAR_CAT", "RIVAL_INFO"):
        ltc, mis_stop, mis_budget, never = [], [], [], 0
        for truth in range(5):
            for _ in range(100):
                prng = np.random.default_rng(rng.integers(2**32))
                post = np.full(5, 0.2)
                stop = None
                for t in range(60):
                    if policy == "FIXED":
                        c = t % 32
                    else:
                        pbar = P @ post
                        if policy == "SCALAR_CAT":
                            c = int(np.argmax(pbar * (1 - pbar)))
                        else:
                            gains = []
                            for ci in range(32):
                                p1 = pbar[ci]
                                post1 = post * P[ci] / p1
                                post0 = post * (1 - P[ci]) / (1 - p1)
                                gains.append(entropy(post) - p1 * entropy(post1) - (1 - p1) * entropy(post0))
                            c = int(np.argmax(gains))
                    y = prng.random() < P[c, truth]
                    post = post * (P[c] if y else 1 - P[c])
                    post /= post.sum()
                    if stop is None and post.max() >= 0.9:
                        stop = t + 1
                        mis_stop.append(int(np.argmax(post) != truth))
                if stop is None:
                    never += 1
                else:
                    ltc.append(stop)
                mis_budget.append(int(np.argmax(post) != truth))
        out[policy] = {"median_trials_to_.90": float(np.median(ltc)) if ltc else None, "mean_trials_to_.90": float(np.mean(ltc)) if ltc else None,
                       "never_reached_share": never / 500, "misclassification_at_stop": float(np.mean(mis_stop)) if mis_stop else None,
                       "misclassification_at_budget_60": float(np.mean(mis_budget))}
    return out


def fu4_b():
    lit = np.array([[.95, .2, .02, .01], [.6, .8, .1, .02], [.02, .1, .8, .6], [.01, .02, .2, .95]])
    lit = np.vstack([lit, 1 - lit])  # 8 utterances x 4 states
    value = np.arange(4.0)
    alpha = 3.0
    L0 = lit / lit.sum(1, keepdims=True)
    phis = np.array([0.1, 0.5, 0.9])

    def s1(phi):
        util = alpha * (phi * np.log(L0 + 1e-12) + (1 - phi) * (L0 @ value)[:, None])
        su = np.exp(util - util.max(0, keepdims=True))
        return su / su.sum(0, keepdims=True)  # u x s
    S1 = np.stack([s1(ph) for ph in phis])  # phi x u x s
    L1_joint = S1 / S1.sum(axis=(0, 2), keepdims=True)  # uniform priors
    L1_s = L1_joint.sum(0)  # u x s
    L1_phi = L1_joint.sum(2).T  # u x phi
    target_phi = 1  # presents as phi = .5
    grid = [(a / 4, b / 4, (4 - a - b) / 4) for a in range(5) for b in range(5 - a)]

    def s2(w):
        wi, ws, wp = w
        util = alpha * (wi * np.log(L1_s + 1e-12) + ws * (L1_s @ value)[:, None] + wp * np.log(L1_phi[:, [target_phi]] + 1e-12))
        su = np.exp(util - util.max(0, keepdims=True))
        return su / su.sum(0, keepdims=True)
    S2 = {w: s2(w) for w in grid}
    truth = (0.5, 0.25, 0.25)
    rng = np.random.default_rng(SEED + 1)
    res = {}
    for design, states in (("single_state_1", [1] * 200), ("all_states", [0, 1, 2, 3] * 50)):
        us = [rng.choice(8, p=S2[truth][:, s]) for s in states]
        ll = np.array([sum(np.log(S2[w][u, s] + 1e-300) for u, s in zip(us, states, strict=True)) for w in grid])
        post = np.exp(ll - ll.max())
        post /= post.sum()
        ws = np.array([w[1] for w in grid])
        wp = np.array([w[2] for w in grid])
        m1, m2 = (post * ws).sum(), (post * wp).sum()
        cov = (post * (ws - m1) * (wp - m2)).sum()
        corr = cov / np.sqrt((post * (ws - m1) ** 2).sum() * (post * (wp - m2) ** 2).sum() + 1e-300)
        res[design] = {"posterior_on_truth": float(post[grid.index(truth)]), "map": grid[int(np.argmax(post))],
                       "posterior_corr_social_vs_presentational": float(corr), "posterior_entropy": entropy(post)}
    return res


def fu4_c():
    m = pd.read_csv(V3 / "F16/text_transport_full_r1/reader_scores.csv")
    m["template"] = m.material_id.str.rsplit("_", n=2).str[0]
    cell = m.pole.astype(str) + "|" + m.qualification.astype(str) + "|" + m.subject_status
    X = pd.get_dummies(cell).to_numpy(float)
    res = {}
    for col in ("nli_qualification", "pole_readout", "grain_qualification"):
        y = m[col].to_numpy(float)
        w = np.linalg.lstsq(X, y, rcond=None)[0]
        resid = y - X @ w
        r2 = 1 - resid.var() / y.var()
        tm = pd.Series(resid).groupby(m.template.to_numpy()).transform("mean").to_numpy()
        res[col] = {"concept_R2": float(r2), "residual_template_share": float(tm.var() / resid.var()) if resid.var() > 0 else None,
                    "design_cells": int(X.shape[1])}
    return res


def main():
    if digest(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("Protocol changed")
    OUT.mkdir(parents=True, exist_ok=True)
    out = {"protocol_sha256": PROTOCOL_SHA}
    for name, fn in (("FU4_a_R26_A07_rival_information_design", fu4_a), ("FU4_b_R15_politeness_RSA", fu4_b),
                     ("FU4_c_A08_concept_bottleneck", fu4_c)):
        out[name] = fn()
        print(name, json.dumps(out[name]), flush=True)
    out.update({"code_sha256": digest(Path(__file__)), "neural_calls": 0, "created": time.strftime("%Y-%m-%d %H:%M:%S %Z")})
    (OUT / "followup4.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
