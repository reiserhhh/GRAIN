#!/usr/bin/env python
"""R29 correlated measurement error and duplicate information (docs/GRAIN_R29_COVARIANCE_IDENTIFICATION_PROTOCOL_V1.md)."""
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
PROTOCOL = ROOT / "docs/GRAIN_R29_COVARIANCE_IDENTIFICATION_PROTOCOL_V1.md"
PROTOCOL_SHA = "76ecfe914dc338810b37825fbcec16d75846b113a45bfb2577411b3fc7ea43d7"
OUT = ROOT / "results/grain_gptpro_foundation_v3/F13/covariance_identification_r1"
SEED, N, R, B, DUP = 20260926, 300, 200, 200, 0.057


class NonIdentified(ValueError):
    pass


# ---------------------------------------------------------------- exact algebra
def params(world: str) -> dict:
    if world in ("W1", "W4"):
        return {"sa2": 1.0, "sb": [0.3, 0.3], "k": [0.0, 0.0], "k12": 0.0, "se": [0.5, 0.5], "se0": 0.5}
    if world == "W2":
        lam, nu = (0.4, 0.2), 0.3
        return {"sa2": 1.0, "sb": [lam[0] ** 2 + nu, lam[1] ** 2 + nu], "k": list(lam), "k12": lam[0] * lam[1], "se": [0.5, 0.5], "se0": 0.5}
    if world == "W3":
        return {"sa2": 1.0, "sb": [0.5, 0.5], "k": [0.0, 0.0], "k12": 0.3, "se": [0.5, 0.5], "se0": 0.5}
    raise ValueError(world)


def population_moments(th: dict, anchor: bool = True) -> dict:
    V = [th["sa2"] + 2 * th["k"][m] + th["sb"][m] for m in (0, 1)]
    mom = {"V1": V[0], "V2": V[1], "S12": th["sa2"] + th["k"][0] + th["k"][1] + th["k12"], "se1": th["se"][0], "se2": th["se"][1]}
    if anchor:
        mom.update({"V0": th["sa2"], "C01": th["sa2"] + th["k"][0], "C02": th["sa2"] + th["k"][1], "se0": th["se0"]})
    return mom


def blind_fit(mom: dict, assumption: str | None = None, request: tuple = ()) -> dict:
    """Identified quantities only; author variance only under an explicitly declared assumption."""
    out = {"V1": mom["V1"], "V2": mom["V2"], "S12": mom["S12"], "D": mom["V1"] + mom["V2"] - 2 * mom["S12"], "se1": mom["se1"], "se2": mom["se2"]}
    if any(q in request for q in ("sa2", "k1", "k2", "k12", "sb1", "sb2")):
        if assumption != "independent_effects":
            raise NonIdentified("Var(a), Cov(a,b), method variances are not identified from repeated occasions alone")
        out.update({"sa2_A0": mom["S12"], "sb1_A0": mom["V1"] - mom["S12"], "sb2_A0": mom["V2"] - mom["S12"], "assumption": "independent_effects (A0)"})
    return out


def anchor_fit(mom: dict) -> dict:
    if not all(k in mom for k in ("V0", "C01", "C02")):
        raise NonIdentified("anchor fit requires an observed independent anchor channel")
    sa2 = mom["V0"]
    k1, k2 = mom["C01"] - sa2, mom["C02"] - sa2
    return {"sa2": sa2, "k1": k1, "k2": k2, "sb1": mom["V1"] - sa2 - 2 * k1, "sb2": mom["V2"] - sa2 - 2 * k2, "k12": mom["S12"] - sa2 - k1 - k2}


# ---------------------------------------------------------------- simulation
def simulate(world: str, rng, n=N, conditions=False) -> dict:
    a = rng.normal(0, 1, n)
    if world in ("W1", "W4"):
        b = rng.normal(0, np.sqrt(0.3), (n, 2))
    elif world == "W2":
        b = np.column_stack([0.4 * a, 0.2 * a]) + rng.normal(0, np.sqrt(0.3), (n, 2))
    elif world == "W3":
        b = rng.normal(0, np.sqrt(0.3), n)[:, None] + rng.normal(0, np.sqrt(0.2), (n, 2))
    if not conditions:
        e = rng.normal(0, np.sqrt(0.5), (n, 2, 2))
        Y = a[:, None, None] + b[:, :, None] + e
        Y0 = a[:, None] + rng.normal(0, np.sqrt(0.5), (n, 2))
        return {"a": a, "b": b, "e": e, "Y": Y, "Y0": Y0}
    beta, d = np.array([0.0, 0.3]), np.array([[0.0, 0.4], [0.0, 0.0]])
    g = rng.normal(0, np.sqrt(0.15), (n, 2))
    e = rng.normal(0, np.sqrt(0.5), (n, 2, 2, 2))  # u, m, c, t
    Y = a[:, None, None, None] + beta[None, None, :, None] + g[:, None, :, None] + b[:, :, None, None] + d[None, :, :, None] + e
    Y0 = a[:, None, None] + beta[None, :, None] + g[:, :, None] + rng.normal(0, np.sqrt(0.5), (n, 2, 2))
    return {"a": a, "b": b, "g": g, "Y": Y, "Y0": Y0}


def person_stats(Y, Y0, dup=None):
    """Person-level sufficient statistics. dup (n,2,2) in {0,1}: extra exact copies of an event row (naive treatment)."""
    w = np.ones_like(Y) if dup is None else 1.0 + dup
    mu = (w * Y).sum(axis=(0, 2)) / w.sum(axis=(0, 2))           # per reader, row-weighted
    r = Y - mu[None, :, None]
    s1 = (w * r).sum(axis=2)
    s2 = (w * r * r).sum(axis=2)
    k = w.sum(axis=2)
    pairsum = s1 ** 2 - s2                                          # sum over distinct row pairs i != j
    npairs = k * (k - 1)
    ybar = (w * Y).sum(axis=2) / k
    within = (s2 - s1 ** 2 / k) / (k - 1)
    st = {"pairsum": pairsum, "npairs": npairs, "ybar": ybar, "within": within, "k": k}
    if Y0 is not None:
        r0 = Y0 - Y0.mean()
        st["a_pair"] = r0[:, 0] * r0[:, 1]
        st["a_bar"] = Y0.mean(axis=1)
    return st


def moments_from(st, idx=None):
    ix = slice(None) if idx is None else idx
    ps, npr, yb, wi = st["pairsum"][ix], st["npairs"][ix], st["ybar"][ix], st["within"][ix]
    V = ps.sum(axis=0) / npr.sum(axis=0)
    mom = {"V1": V[0], "V2": V[1], "S12": float(np.cov(yb[:, 0], yb[:, 1])[0, 1]), "se1": wi[:, 0].mean(), "se2": wi[:, 1].mean()}
    if "a_pair" in st:
        ab = st["a_bar"][ix]
        n = len(ab)
        mom["V0"] = st["a_pair"][ix].mean() * n / (n - 1)
        mom["C01"] = float(np.cov(ab, yb[:, 0])[0, 1])
        mom["C02"] = float(np.cov(ab, yb[:, 1])[0, 1])
    return mom


def estimates(mom, truth_sb1):
    bl = blind_fit(mom, assumption="independent_effects", request=("sa2",))
    an = anchor_fit(mom)
    return {"V1": bl["V1"], "V2": bl["V2"], "S12": bl["S12"], "D": bl["D"], "se1": bl["se1"], "se2": bl["se2"],
            "rel1": bl["V1"] / (bl["V1"] + bl["se1"]), "sa2_A0": bl["sa2_A0"],
            "sa2_anchor": an["sa2"], "k1_anchor": an["k1"], "k2_anchor": an["k2"], "sb1_anchor": an["sb1"], "sb2_anchor": an["sb2"], "k12_anchor": an["k12"],
            "ORACLE_sa2_mech": bl["V1"] - truth_sb1}


def truths(world):
    th = params(world)
    m = population_moments(th)
    return {"V1": m["V1"], "V2": m["V2"], "S12": m["S12"], "D": m["V1"] + m["V2"] - 2 * m["S12"], "se1": th["se"][0], "se2": th["se"][1],
            "rel1": m["V1"] / (m["V1"] + th["se"][0]), "sa2_A0": th["sa2"], "sa2_anchor": th["sa2"], "k1_anchor": th["k"][0], "k2_anchor": th["k"][1],
            "sb1_anchor": th["sb"][0], "sb2_anchor": th["sb"][1], "k12_anchor": th["k12"], "ORACLE_sa2_mech": th["sa2"]}


def run_world(world, seed_offset, dup_arm=False):
    rng = np.random.default_rng(SEED + seed_offset)
    tr = truths(world)
    rows = []
    for rep in range(R):
        sim = simulate(world, rng)
        dup = (rng.random(sim["Y"].shape) < DUP).astype(float) if dup_arm else None
        variants = [("event_aware", None)] + ([("naive_rows", dup)] if dup_arm else [])
        boot_idx = rng.integers(0, N, (B, N))
        for treat, dd in variants:
            st = person_stats(sim["Y"], sim["Y0"], dd)
            est = estimates(moments_from(st), params(world)["sb"][0])
            bs = pd.DataFrame([estimates(moments_from(st, boot_idx[b]), params(world)["sb"][0]) for b in range(B)])
            for k, v in est.items():
                lo, hi = np.percentile(bs[k], [2.5, 97.5])
                rows.append({"world": world, "rep": rep, "treatment": treat, "quantity": k, "estimate": v, "truth": tr[k], "ci_low": lo, "ci_high": hi,
                             "covered": bool(lo <= tr[k] <= hi), "rows": float(st["k"].sum()), "events": float(sim["Y"].size)})
    return rows


def condition_arm(seed_offset):
    rng = np.random.default_rng(SEED + seed_offset)
    out = []
    for rep in range(R):
        s = simulate("W4", rng, conditions=True)
        Y, Y0 = s["Y"], s["Y0"]
        eff_reader = Y[:, :, 1, :].mean(axis=(0, 2)) - Y[:, :, 0, :].mean(axis=(0, 2))
        E_R = eff_reader.mean()
        contrast_d = eff_reader[0] - eff_reader[1]
        E_0 = Y0[:, 1, :].mean() - Y0[:, 0, :].mean()
        se_hat = (np.var(Y[:, :, :, 0] - Y[:, :, :, 1], ddof=1, axis=0) / 2).mean()
        delta = Y[:, :, 1, :].mean(axis=(1, 2)) - Y[:, :, 0, :].mean(axis=(1, 2))
        sg2 = (np.var(delta, ddof=1) - 2 * se_hat / 4) / 2
        out.append({"rep": rep, "reader_average_condition_effect": E_R, "reader_condition_contrast": contrast_d, "anchor_condition_effect_beta": E_0,
                    "reader_average_bias_dbar": E_R - E_0, "person_by_condition_var": sg2, "mean_person_contrast_fixed_offset": delta.mean()})
    df = pd.DataFrame(out)
    truth = {"reader_average_condition_effect": 0.5, "reader_condition_contrast": 0.4, "anchor_condition_effect_beta": 0.3,
             "reader_average_bias_dbar": 0.2, "person_by_condition_var": 0.15, "mean_person_contrast_fixed_offset": 0.5}
    summ = {k: {"mean": float(df[k].mean()), "mc_se": float(df[k].std(ddof=1) / np.sqrt(R)), "truth": truth[k]} for k in truth}
    return df, summ


def counterexamples():
    rng = np.random.default_rng(SEED + 999)
    s = simulate("W2", rng)
    a, b, e = s["a"], s["b"], s["e"]
    c = 0.3 * b.mean(axis=1)
    Y1 = a[:, None, None] + b[:, :, None] + e
    Y2 = (a + c)[:, None, None] + (b - c[:, None])[:, :, None] + e
    m1, m2 = moments_from(person_stats(Y1, None)), moments_from(person_stats(Y2, None))
    samp = {"param1": {"var_a": float(np.var(a, ddof=1)), "cov_a_b1": float(np.cov(a, b[:, 0])[0, 1]), "cov_a_b2": float(np.cov(a, b[:, 1])[0, 1])},
            "param2": {"var_a": float(np.var(a + c, ddof=1)), "cov_a_b1": float(np.cov(a + c, b[:, 0] - c)[0, 1]), "cov_a_b2": float(np.cov(a + c, b[:, 1] - c)[0, 1])},
            "max_abs_Y_diff": float(np.abs(Y1 - Y2).max()), "max_abs_identified_moment_diff": float(max(abs(m1[k] - m2[k]) for k in m1)),
            "A0_estimate_both": [blind_fit(m1, "independent_effects", ("sa2",))["sa2_A0"], blind_fit(m2, "independent_effects", ("sa2",))["sa2_A0"]]}
    th1 = params("W1")
    th2 = {"sa2": 0.8, "sb": [0.3, 0.3], "k": [0.1, 0.1], "k12": 0.0, "se": [0.5, 0.5], "se0": 0.5}
    p1, p2 = population_moments(th1, anchor=False), population_moments(th2, anchor=False)
    pop = {"theta1": th1, "theta2": th2, "moments1": p1, "moments2": p2, "max_abs_moment_diff": max(abs(p1[k] - p2[k]) for k in p1),
           "var_a_differs": th1["sa2"] != th2["sa2"], "anchor_moments_distinguish": population_moments(th1)["V0"] != population_moments(th2)["V0"]}
    exact = {w: {"anchor_fit_minus_truth_max": max(abs(v - {"sa2": params(w)["sa2"], "k1": params(w)["k"][0], "k2": params(w)["k"][1], "sb1": params(w)["sb"][0],
                                                              "sb2": params(w)["sb"][1], "k12": params(w)["k12"]}[k]) for k, v in anchor_fit(population_moments(params(w))).items())}
             for w in ("W1", "W2", "W3")}
    # joint error identity Var(Y_m) = sa2 + Var(b+e) + 2Cov(a, b+e), per world, exact
    ident = {}
    for w in ("W1", "W2", "W3"):
        th = params(w)
        lhs = th["sa2"] + th["sb"][0] + 2 * th["k"][0] + th["se"][0]
        rhs = th["sa2"] + (th["sb"][0] + th["se"][0]) + 2 * th["k"][0]
        ident[w] = {"VarY1": lhs, "sa2": th["sa2"], "Var_b_plus_e": th["sb"][0] + th["se"][0], "2Cov_a_b": 2 * th["k"][0], "identity_error": abs(lhs - rhs)}
    return {"sample_counterexample_W2": samp, "population_counterexample": pop, "anchor_exact_recovery": exact, "joint_error_identity": ident}


def main() -> int:
    t0 = time.time()
    if hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() != PROTOCOL_SHA:
        raise RuntimeError("protocol sha mismatch")
    OUT.mkdir(parents=True, exist_ok=True)
    ce = counterexamples()
    (OUT / "identification_counterexamples.json").write_text(json.dumps(ce, indent=1, default=float) + "\n")
    from joblib import Parallel, delayed
    jobs = [("W1", 1, True), ("W2", 2, True), ("W3", 3, False)]
    res = Parallel(n_jobs=2)(delayed(run_world)(w, o, d) for w, o, d in jobs)
    df = pd.DataFrame([r for rr in res for r in rr])
    df.to_csv(OUT / "per_draw_estimates.csv", index=False)
    summ = df.groupby(["world", "treatment", "quantity"]).agg(truth=("truth", "first"), mean=("estimate", "mean"), sd=("estimate", "std"),
                                                             coverage=("covered", "mean"), rows=("rows", "mean"), events=("events", "mean")).reset_index()
    summ["bias"] = summ["mean"] - summ["truth"]
    summ["bias_mc_se"] = summ["sd"] / np.sqrt(R)
    summ["coverage_mc_se"] = np.sqrt(summ["coverage"] * (1 - summ["coverage"]) / R)
    summ.to_csv(OUT / "estimator_risk.csv", index=False)
    cdf, csumm = condition_arm(4)
    cdf.to_csv(OUT / "condition_arm_per_draw.csv", index=False)
    (OUT / "condition_arm.json").write_text(json.dumps(csumm, indent=1) + "\n")
    (OUT / "runtime.json").write_text(json.dumps({"seconds": time.time() - t0, "neural_calls": 0, "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}) + "\n")
    print(json.dumps({"seconds": time.time() - t0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
