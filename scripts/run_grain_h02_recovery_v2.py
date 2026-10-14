#!/usr/bin/env python
"""H02 recovery V2: encompassing-model mechanism-set recovery (docs/GRAIN_H02_RECOVERY_PROTOCOL_V2.md)."""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

REPO = Path(__file__).resolve().parents[1]
PROTOCOL = REPO / "docs/GRAIN_H02_RECOVERY_PROTOCOL_V2.md"
PROTOCOL_SHA = "01e7351ac69dd29e54df41a50a3535f3b370093fc593cccc6b1a1a4ff6ba914c"
OUT = REPO / "results/grain_gptpro_foundation_v3/F17/h02_recovery_r2"
_spec = importlib.util.spec_from_file_location("h02v1", REPO / "scripts/run_grain_h02_recovery.py")
v1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v1)
SEED = 20260927
SIMS = 100
TRUTHS = {"H1": {"H1"}, "H2": {"H2"}, "H3": {"H3"}, "H4": {"H4"}, "H1+H2": {"H1", "H2"}, "H3+H4": {"H3", "H4"}}
MECHS = ["H1", "H2", "H3", "H4"]


def simulate(active: set[str], d: float, tau: float, n: int, r: int, rng: np.random.Generator) -> dict:
    """V1 world, generalized to any active mechanism set (each at d/len(active))."""
    base = v1.simulate("H1", 0.0, 0.0, n, r, rng)  # layout, B, V; mechanism rebuilt below
    person, c, b = base["person"], base["c"], base["b"]
    m = len(person)
    theta, xi = rng.normal(size=(2, n))
    v_person = base["v"][np.unique(person, return_index=True)[1]]
    a_i = 0.5 * (0.8 * v_person + 0.6 * xi) if "H4" in active else 0.5 * theta
    tau_i = rng.normal(0, tau, n)
    scale = 1.0 / len(active)
    eff = scale * (d + tau_i[person])
    mech = np.zeros(m)
    if "H1" in active:
        mech += -eff * b
    if "H2" in active:
        mech += eff * c[:, 1]
    if "H3" in active:
        mech += eff * c[:, 2]
    if "H4" in active:
        mech += -eff * c[:, 3]
    x_true = a_i[person] + mech + rng.normal(size=m)
    sigma_c2 = np.var(x_true) * (1 / v1.CODER_REL - 1)
    x_obs = x_true + rng.normal(0, np.sqrt(sigma_c2), size=(2, m)).mean(axis=0)
    a_lin = 0.8 * b - (eff * c[:, 1] if "H2" in active else 0.0)
    act = (rng.random(m) < 1 / (1 + np.exp(-a_lin))).astype(float)
    return {**base, "x": x_obs, "a": act}


def encompassing_designs(sim: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    one = np.ones(len(sim["x"]))
    b, c, v = sim["b"], sim["c"], sim["v"]
    cols = {"H1": [b], "H2": [c[:, 1]], "H3": [c[:, 2]], "H4": [c[:, 3], v]}
    out = {}
    for drop in [None] + MECHS:
        xcols = [one] + [col for k in MECHS if k != drop for col in cols[k]]
        acols = [one, b] + ([c[:, 1]] if drop != "H2" else [])
        out["E" if drop is None else f"E-{drop}"] = (np.column_stack(xcols), np.column_stack(acols))
    return out


def one_sim(sim: dict, truth: set[str], rng: np.random.Generator) -> dict:
    n = sim["n"]
    frac = ~sim["held"]
    dz = encompassing_designs(sim)
    folds = rng.permutation(n) % 5
    pp = {k: np.zeros(n) for k in dz}
    for f in range(5):
        test_p = folds == f
        train = frac & ~test_p[sim["person"]]
        test = frac & test_p[sim["person"]]
        for k, (xd, ad) in dz.items():
            coef, s2 = v1.ols_fit(xd[train], sim["x"][train])
            w = v1.logit_fit(ad[train], sim["a"][train])
            ls = v1.gauss_ls(xd[test], sim["x"][test], coef, s2) + v1.logit_ls(ad[test], sim["a"][test], w)
            pp[k] += np.bincount(sim["person"][test], weights=ls, minlength=n)
    z = {}
    for k in MECHS:
        diff = pp["E"] - pp[f"E-{k}"]
        z[k] = float(diff.mean() / (diff.std(ddof=1) / np.sqrt(n)))
    act_primary = {k for k in MECHS if z[k] >= 2.50}
    act_secondary = {k for k in MECHS if z[k] >= 1.96}
    v1_res = v1.t1_t2_t3(sim, "H1+H2" if len(truth) > 1 else next(iter(truth)), True, rng) if len(truth) > 1 else None
    return {"exact": act_primary == truth, "false_incl": bool(act_primary - truth), "false_excl": bool(truth - act_primary),
            "exact_196": act_secondary == truth, "false_incl_196": bool(act_secondary - truth),
            "v1_decided_single": (bool(v1_res["decided"]) if v1_res else None)}


def run_cell(idx: int, name: str, d: float, n: int) -> dict:
    rng = np.random.default_rng(SEED + idx)
    truth = TRUTHS[name]
    rows = [one_sim(simulate(truth, d, 0.25, n, 2, rng), truth, rng) for _ in range(SIMS)]
    df = pd.DataFrame(rows)
    out = {"truth": name, "d": d, "N": n, "sims": SIMS, "p_exact": float(df.exact.mean()), "p_false_inclusion": float(df.false_incl.mean()),
           "p_false_exclusion": float(df.false_excl.mean()), "p_exact_z1.96": float(df.exact_196.mean()),
           "p_false_inclusion_z1.96": float(df.false_incl_196.mean())}
    if len(truth) > 1:
        out["v1_rule_p_decided_single"] = float(df.v1_decided_single.astype(float).mean())
    return out


def main() -> int:
    assert hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() == PROTOCOL_SHA
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    specs = [(name, d, n) for name in TRUTHS for d in (0.2, 0.35, 0.5) for n in (150, 300, 450, 600)]
    rows = Parallel(n_jobs=2)(delayed(run_cell)(i, *s) for i, s in enumerate(specs))
    grid = pd.DataFrame(rows)
    grid.to_csv(OUT / "mechanism_set_recovery.csv", index=False)

    def smallest(d: float):
        for n in (150, 300, 450, 600):
            g = grid[(grid.d == d) & (grid.N == n)]
            if (g.p_exact >= 0.80).all() and (g.p_false_inclusion <= 0.10).all():
                return n
        return None

    decision = {"design_N_d.35": smallest(0.35), "N_pessimistic_d.2": smallest(0.2), "N_optimistic_d.5": smallest(0.5),
                "elapsed_seconds": time.time() - t0, "protocol_sha256": PROTOCOL_SHA,
                "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT / "decision.json").write_text(json.dumps(decision, indent=1) + "\n")
    print(json.dumps(decision, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
