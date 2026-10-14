#!/usr/bin/env python
"""H02-specific design recovery (protocol docs/GRAIN_H02_RECOVERY_PROTOCOL_V1.md).

Simulation with the actual analysis estimators: leave-persons-out CV predictive log score
for rival identification, held-out-cell log score, Hedge split-half reliability, the
incremental F-test and ICC(2,1). No human data.
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
PROTOCOL = REPO / "docs/GRAIN_H02_RECOVERY_PROTOCOL_V1.md"
PROTOCOL_SHA = "96a92840afad0b8f8b60630df4ee7123704a23d965c751bd12ba3505f5d0bfe2"
AMENDMENT = REPO / "docs/GRAIN_H02_RECOVERY_PROTOCOL_V1_A1.md"
AMENDMENT_SHA = "0c8cb6c34f19fe10d517ebf8d6fb7365c2101a05bb3611a759f1225ff4950f19"
OUT = REPO / "results/grain_gptpro_foundation_v3/F17/h02_recovery_r1"
SEED = 20260926
SIMS = 100
TRUTHS = ["H1", "H2", "H3", "H4"]
MODELS = ["H1", "H2", "H3", "H4"]
KEY_FACTOR = {"H1": 0, "H2": 1, "H3": 2, "H4": 3, "H1+H2": 1}
CELLS = np.array([c for c in itertools.product([-0.5, 0.5], repeat=4) if np.prod(np.sign(c)) > 0])
HELD = np.array([0.5, 0.5, 0.5, -0.5])
CODER_REL = 0.7


def simulate(truth: str, d: float, tau: float, n: int, r: int, rng: np.random.Generator) -> dict:
    theta, v, xi = rng.normal(size=(3, n))
    u_b = rng.normal(0, 0.5, n)
    tau_i = rng.normal(0, tau, n)
    person = np.concatenate([np.repeat(np.arange(n), len(CELLS) * r), np.arange(n)])
    rep = np.concatenate([np.tile(np.repeat(np.arange(r), len(CELLS)), n), np.zeros(n, int)])
    cell_id = np.concatenate([np.tile(np.tile(np.arange(len(CELLS)), r), n), np.full(n, -1)])
    c = np.vstack([np.tile(np.tile(CELLS, (r, 1)), (n, 1)), np.tile(HELD, (n, 1))])
    held = cell_id < 0
    m = len(person)
    b = 0.5 * c[:, 0] + u_b[person] + rng.normal(size=m)
    a_i = 0.5 * (0.8 * v + 0.6 * xi) if truth == "H4" else 0.5 * theta
    eff = d + tau_i[person]
    if truth == "H1":
        mech = -eff * b
    elif truth == "H2":
        mech = eff * c[:, 1]
    elif truth == "H3":
        mech = eff * c[:, 2]
    elif truth == "H4":
        mech = -eff * c[:, 3]
    else:  # compound H1+H2 at half effect each
        mech = -0.5 * eff * b + 0.5 * eff * c[:, 1]
    x_true = a_i[person] + mech + rng.normal(size=m)
    sigma_c2 = np.var(x_true) * (1 / CODER_REL - 1)
    x_obs = x_true + rng.normal(0, np.sqrt(sigma_c2), size=(2, m)).mean(axis=0)
    a_lin = 0.8 * b
    if truth == "H2":
        a_lin = a_lin - eff * c[:, 1]
    elif truth == "H1+H2":
        a_lin = a_lin - 0.5 * eff * c[:, 1]
    act = (rng.random(m) < 1 / (1 + np.exp(-a_lin))).astype(float)
    return {"person": person, "rep": rep, "cell": cell_id, "c": c, "held": held, "b": b, "v": v[person], "x": x_obs, "a": act, "n": n}


def designs(sim: dict, b_observed: bool) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    one = np.ones(len(sim["x"]))
    bv = sim["b"] if b_observed else sim["c"][:, 0]
    c = sim["c"]
    return {
        "H1": (np.column_stack([one, bv]), np.column_stack([one, bv])),
        "H2": (np.column_stack([one, c[:, 1]]), np.column_stack([one, bv, c[:, 1]])),
        "H3": (np.column_stack([one, c[:, 2]]), np.column_stack([one, bv])),
        "H4": (np.column_stack([one, c[:, 3], sim["v"]]), np.column_stack([one, bv])),
    }


def ols_fit(x: np.ndarray, y: np.ndarray):
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ coef
    return coef, float(np.mean(resid**2))


def gauss_ls(x, y, coef, s2):
    return -0.5 * np.log(2 * np.pi * s2) - (y - x @ coef) ** 2 / (2 * s2)


def logit_fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    w = np.zeros(x.shape[1])
    for _ in range(50):
        p = 1 / (1 + np.exp(-(x @ w)))
        g = x.T @ (y - p) - 1e-6 * w
        h = (x * (p * (1 - p))[:, None]).T @ x + 1e-6 * np.eye(x.shape[1])
        step = np.linalg.solve(h, g)
        w += step
        if np.max(np.abs(step)) < 1e-9:
            break
    return w


def logit_ls(x, y, w):
    p = np.clip(1 / (1 + np.exp(-(x @ w))), 1e-12, 1 - 1e-12)
    return y * np.log(p) + (1 - y) * np.log(1 - p)


def model_scores(sim, dz, train, test):
    out = {}
    for k in MODELS:
        xd, ad = dz[k]
        coef, s2 = ols_fit(xd[train], sim["x"][train])
        w = logit_fit(ad[train], sim["a"][train])
        out[k] = gauss_ls(xd[test], sim["x"][test], coef, s2) + logit_ls(ad[test], sim["a"][test], w)
    return out


def t1_t2_t3(sim: dict, truth: str, b_observed: bool, rng: np.random.Generator) -> dict:
    n = sim["n"]
    dz = designs(sim, b_observed)
    frac = ~sim["held"]
    folds = rng.permutation(n) % 5
    pp = np.zeros((n, len(MODELS)))
    for f in range(5):
        test_p = folds == f
        train = frac & ~test_p[sim["person"]]
        test = frac & test_p[sim["person"]]
        scores = model_scores(sim, dz, train, test)
        for j, k in enumerate(MODELS):
            pp[:, j] += np.bincount(sim["person"][test], weights=scores[k], minlength=n)
    tot = pp.sum(axis=0)
    order = np.argsort(-tot)
    diff = pp[:, order[0]] - pp[:, order[1]]
    z = diff.mean() / (diff.std(ddof=1) / np.sqrt(n))
    selected = MODELS[order[0]]
    decided = bool(z >= 1.96)
    held_scores = model_scores(sim, dz, frac, sim["held"])
    held_best = MODELS[int(np.argmax([held_scores[k].sum() for k in MODELS]))]
    # T3: person slopes on the true mechanism's manipulation (fraction trials only)
    kf = KEY_FACTOR[truth]
    ck = sim["c"][:, kf]
    if int(sim["rep"].max()) >= 1:
        half = sim["rep"] % 2
    else:
        rank = np.zeros(len(ck), int)
        for lev in (-0.5, 0.5):
            cells_lev = sorted(set(sim["cell"][frac & (ck == lev)]))
            for i, cid in enumerate(cells_lev):
                rank[sim["cell"] == cid] = i % 2
        half = rank

    def slopes(mask):
        hi = np.bincount(sim["person"][mask & (ck > 0)], weights=sim["x"][mask & (ck > 0)], minlength=n) / np.maximum(
            np.bincount(sim["person"][mask & (ck > 0)], minlength=n), 1)
        lo = np.bincount(sim["person"][mask & (ck < 0)], weights=sim["x"][mask & (ck < 0)], minlength=n) / np.maximum(
            np.bincount(sim["person"][mask & (ck < 0)], minlength=n), 1)
        return hi - lo

    full = slopes(frac)
    s1, s2 = slopes(frac & (half == 0)), slopes(frac & (half == 1))
    rh = np.corrcoef(s1, s2)[0, 1]
    rel = 2 * rh / (1 + rh) if rh > -1 else np.nan
    t = full.mean() / (full.std(ddof=1) / np.sqrt(n))
    return {"selected": selected, "decided": decided, "z": float(z), "held_best": held_best,
            "t3_mean_detect": bool(2 * stats.t.sf(abs(t), n - 1) < 0.05), "t3_reliability": float(rel)}


def run_cell(idx: int, truth: str, d: float, tau: float, n: int, r: int, b_obs: bool) -> dict:
    rng = np.random.default_rng(SEED + idx)
    rows = []
    for _ in range(SIMS):
        sim = simulate(truth, d, tau, n, r, rng)
        rows.append(t1_t2_t3(sim, truth, b_obs, rng))
    df = pd.DataFrame(rows)
    out = {"truth": truth, "d": d, "tau": tau, "N": n, "r": r, "B_observed": b_obs, "sims": SIMS,
           "p_decided": float(df.decided.mean()), "t3_mean_detect": float(df.t3_mean_detect.mean()),
           "t3_reliability_mean": float(df.t3_reliability.mean())}
    if truth in MODELS:
        out["p_correct_decided"] = float((df.decided & (df.selected == truth)).mean())
        out["p_wrong_decided"] = float((df.decided & (df.selected != truth)).mean())
        out["p_undetermined"] = float((~df.decided).mean())
        out["p_heldout_true_best"] = float((df.held_best == truth).mean())
    for k in MODELS:
        out[f"sel_{k}"] = float((df.selected == k).mean())
        out[f"sel_decided_{k}"] = float((df.decided & (df.selected == k)).mean())
    return out


def t4(n: int, s: float, sims: int, rng: np.random.Generator) -> tuple[float, float]:
    hits, dr2 = 0, []
    for _ in range(sims):
        th, sf = rng.normal(size=(2, n))
        yq = 0.7 * th + rng.normal(0, np.sqrt(1 - 0.49), n)
        yb = 0.5 * th + s * sf + rng.normal(0, np.sqrt(1 - 0.25 - s * s), n)
        yt = 0.5 * th + s * sf + rng.normal(0, np.sqrt(1 - 0.25 - s * s), n)
        x1 = np.column_stack([np.ones(n), yq])
        x2 = np.column_stack([np.ones(n), yq, yt])
        rss1 = np.sum((yb - x1 @ np.linalg.lstsq(x1, yb, rcond=None)[0]) ** 2)
        rss2 = np.sum((yb - x2 @ np.linalg.lstsq(x2, yb, rcond=None)[0]) ** 2)
        f = (rss1 - rss2) / (rss2 / (n - 3))
        hits += stats.f.sf(f, 1, n - 3) < 0.05
        tss = np.sum((yb - yb.mean()) ** 2)
        dr2.append((rss1 - rss2) / tss)
    return hits / sims, float(np.mean(dr2))


def t4b(n: int, s: float, sims: int, rng: np.random.Generator) -> float:
    """A1: one-factor tetrad test with a fourth (other-report) indicator; bootstrap SE."""
    hits = 0
    for _ in range(sims):
        th, sf = rng.normal(size=(2, n))
        yq = 0.7 * th + rng.normal(0, np.sqrt(1 - 0.49), n)
        yo = 0.6 * th + rng.normal(0, np.sqrt(1 - 0.36), n)
        yb = 0.5 * th + s * sf + rng.normal(0, np.sqrt(1 - 0.25 - s * s), n)
        yt = 0.5 * th + s * sf + rng.normal(0, np.sqrt(1 - 0.25 - s * s), n)
        data = np.column_stack([yt, yb, yq, yo])

        def tetrad(m):
            r = np.corrcoef(m, rowvar=False)
            return r[0, 1] * r[2, 3] - r[0, 2] * r[1, 3]

        est = tetrad(data)
        boots = [tetrad(data[rng.integers(0, n, n)]) for _ in range(200)]
        hits += est / np.std(boots, ddof=1) >= 1.645
    return hits / sims


def icc21(ratings: np.ndarray) -> float:
    n, k = ratings.shape
    gm = ratings.mean()
    msr = k * np.sum((ratings.mean(axis=1) - gm) ** 2) / (n - 1)
    msc = n * np.sum((ratings.mean(axis=0) - gm) ** 2) / (k - 1)
    sse = np.sum((ratings - ratings.mean(axis=1, keepdims=True) - ratings.mean(axis=0, keepdims=True) + gm) ** 2)
    mse = sse / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))


def t5(n: int, icc: float, sims: int, rng: np.random.Generator) -> tuple[float, float]:
    est = []
    sb2 = 0.02
    se2 = (1 - icc) / icc - sb2  # sigma_T^2 = 1
    for _ in range(sims):
        t = rng.normal(size=n)
        bias = rng.normal(0, np.sqrt(sb2), 2)
        ratings = t[:, None] + bias[None, :] + rng.normal(0, np.sqrt(se2), (n, 2))
        est.append(icc21(ratings))
    return float(np.mean(est)), float(np.std(est, ddof=1))


def main() -> int:
    assert hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() == PROTOCOL_SHA
    assert hashlib.sha256(AMENDMENT.read_bytes()).hexdigest() == AMENDMENT_SHA
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    specs = []
    for truth in TRUTHS + ["H1+H2"]:
        for d in (0.2, 0.35, 0.5):
            for n in (150, 300, 450, 600):
                for r in (1, 2):
                    for b_obs in (True, False):
                        specs.append((truth, d, 0.25, n, r, b_obs))
    for truth in TRUTHS:
        for n in (150, 300, 450, 600):
            for r in (1, 2):
                specs.append((truth, 0.35, 0.1, n, r, True))
    rows = Parallel(n_jobs=2)(delayed(run_cell)(i, *spec) for i, spec in enumerate(specs))
    grid = pd.DataFrame(rows)
    grid.to_csv(OUT / "t1_t2_t3_grid.csv", index=False)
    rng = np.random.default_rng(SEED + 10_000)
    t4_rows = []
    for n in (150, 300, 450, 600):
        for s in (0.0, 0.3):
            power, dr2 = t4(n, s, 1000, rng)
            t4_rows.append({"N": n, "s": s, "reject_rate": power, "mean_delta_r2": dr2})
    for row in t4_rows:
        row["t4b_tetrad_reject_rate"] = t4b(row["N"], row["s"], 1000, rng)
    pd.DataFrame(t4_rows).to_csv(OUT / "t4_criterion.csv", index=False)
    t5_rows = []
    for n in (100, 200, 400):
        for icc in (0.7, 0.6):
            mean, sd = t5(n, icc, 1000, rng)
            t5_rows.append({"n_double_coded": n, "true_icc": icc, "mean_est": mean, "sd_est": sd})
    pd.DataFrame(t5_rows).to_csv(OUT / "t5_coding.csv", index=False)
    # frozen decision rules
    main_grid = grid[(grid.tau == 0.25) & grid.truth.isin(TRUTHS)]

    def smallest_n(d: float, r: int, b_obs: bool):
        for n in (150, 300, 450, 600):
            g = main_grid[(main_grid.d == d) & (main_grid.r == r) & (main_grid.B_observed == b_obs) & (main_grid.N == n)]
            if (g.p_correct_decided >= 0.80).all() and (g.p_wrong_decided <= 0.05).all():
                return n
        return None

    t4df = pd.DataFrame(t4_rows)
    crit_n = next((int(n) for n in (150, 300, 450, 600)
                   if float(t4df[(t4df.N == n) & (t4df.s == 0.3)].t4b_tetrad_reject_rate.iloc[0]) >= 0.80
                   and float(t4df[(t4df.N == n) & (t4df.s == 0.0)].t4b_tetrad_reject_rate.iloc[0]) <= 0.07), None)
    t1_n = smallest_n(0.35, 2, True)
    design_n = max(x for x in (t1_n, crit_n) if x is not None) if (t1_n or crit_n) else None
    t5df = pd.DataFrame(t5_rows)
    coding_n = {str(icc): next((int(n) for n in (100, 200, 400) if float(t5df[(t5df.n_double_coded == n) & (t5df.true_icc == icc)].sd_est.iloc[0]) <= 0.05), None) for icc in (0.7, 0.6)}
    at = lambda **kw: main_grid.loc[np.logical_and.reduce([main_grid[k] == v for k, v in kw.items()])]
    decision = {
        "t1_design_N_primary_r2_Bobs": t1_n,
        "t1_N_pessimistic_d.2_r2_Bobs": smallest_n(0.2, 2, True),
        "t1_N_primary_r1_Bobs": smallest_n(0.35, 1, True),
        "t1_N_primary_r2_Bunobserved": smallest_n(0.35, 2, False),
        "criterion_N_T4b_tetrad": crit_n,
        "design_N": design_n,
        "coding_n_double_for_sd_le_.05": coding_n,
        "elapsed_seconds": time.time() - t0,
        "protocol_sha256": PROTOCOL_SHA,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    if design_n:
        g = at(d=0.35, r=2, B_observed=True, N=design_n)
        gno = at(d=0.35, r=2, B_observed=False, N=design_n)
        decision["at_design"] = {
            "p_correct_decided_by_truth": dict(zip(g.truth, g.p_correct_decided.round(3))),
            "p_wrong_decided_by_truth": dict(zip(g.truth, g.p_wrong_decided.round(3))),
            "p_heldout_true_best_by_truth": dict(zip(g.truth, g.p_heldout_true_best.round(3))),
            "t3_reliability_by_truth": dict(zip(g.truth, g.t3_reliability_mean.round(3))),
            "t3_mean_detect_by_truth": dict(zip(g.truth, g.t3_mean_detect.round(3))),
            "B_unobserved_p_correct_decided_by_truth": dict(zip(gno.truth, gno.p_correct_decided.round(3))),
        }
    (OUT / "decision.json").write_text(json.dumps(decision, indent=1) + "\n")
    print(json.dumps(decision, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
