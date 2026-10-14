#!/usr/bin/env python3
"""F07-05: MAP recovery of (a,b) under measured reader emissions (not only information).

Symbols (missing, positive, negative) are generated through a measured emission
matrix; estimators: source-frozen measured channel, naive ignorable (drop missing,
read symbols as responses), known application channel (reference), oracle
complete responses (reference). Prior a~N(0,1), b~N(.5,.5^2) (F07 protocol);
32 opportunities alternating z=-1,+1; 600 application authors per seed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from grain_research.gptpro_foundation_v3.f07_reader_channel import fit_emission

ROOT = Path(__file__).resolve().parents[1]
F08 = ROOT / "results/grain_gptpro_foundation_v3/F08"
READERS = {
    "lexical": F08 / "reader_recovery_r1/reader_scores.csv",
    "nli": F08 / "nli_reader_r1/reader_scores.csv",
}
SEEDS = (20260915, 20260916, 20260917)
PRIOR_MU, PRIOR_SD = np.array([0.0, 0.5]), np.array([1.0, 0.5])
Z = np.tile([-1.0, 1.0], 16)


def emissions(path):
    """Source-frozen and held-out application mixed-panel emission matrices."""
    with Path(path).open() as f:
        rows = list(csv.DictReader(f))
    source = fit_emission([r for r in rows if r["role"] == "source"], "mixed")
    app = fit_emission(
        [dict(r, role="source") for r in rows if r["role"] == "application"], "mixed"
    )
    return source, app, hashlib.sha256(Path(path).read_bytes()).hexdigest()


def neg_log_post(theta, s, y, estimator, e):
    """Negative log posterior for one author under an estimator's likelihood."""
    p = expit(theta[0] + theta[1] * Z)
    if estimator == "oracle_complete":
        ll = np.sum(y * np.log(p) + (1 - y) * np.log1p(-p))
    elif estimator == "naive_ignorable":
        keep = s > 0
        yy = (s[keep] == 1).astype(float)
        pk = p[keep]
        ll = np.sum(yy * np.log(pk) + (1 - yy) * np.log1p(-pk))
    else:
        probs = (1 - p)[:, None] * e[0][None] + p[:, None] * e[1][None]
        ll = np.sum(np.log(np.maximum(probs[np.arange(32), s], 1e-300)))
    return -ll + 0.5 * np.sum(((theta - PRIOR_MU) / PRIOR_SD) ** 2)


def fit(s, y, estimator, e):
    """MAP by BFGS (gtol 1e-8, maxiter 200) with a numerical Laplace variance for b."""
    res = minimize(
        neg_log_post,
        PRIOR_MU.copy(),
        args=(s, y, estimator, e),
        method="BFGS",
        options={"gtol": 1e-8, "maxiter": 200},
    )
    h = 1e-4
    hess = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            di, dj = np.eye(2)[i] * h, np.eye(2)[j] * h
            hess[i, j] = (
                neg_log_post(res.x + di + dj, s, y, estimator, e)
                - neg_log_post(res.x + di - dj, s, y, estimator, e)
                - neg_log_post(res.x - di + dj, s, y, estimator, e)
                + neg_log_post(res.x - di - dj, s, y, estimator, e)
            ) / (4 * h * h)
    cov = np.linalg.inv(hess)
    return res.x, cov, bool(res.success)


def main():
    """Run all readers x generating channels x seeds; paired contrasts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows, meta = [], {}
    for reader, path in READERS.items():
        e_src, e_app, digest = emissions(path)
        meta[reader] = {
            "reader_scores_sha256": digest,
            "source_emission": e_src.tolist(),
            "application_emission": e_app.tolist(),
            "max_difference": float(np.abs(e_src - e_app).max()),
        }
        for gen_name, e_gen in (
            ("source_channel", e_src),
            ("application_channel", e_app),
        ):
            for seed in SEEDS:
                rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [
                            seed,
                            7,
                            5,
                            list(READERS).index(reader),
                            gen_name == "application_channel",
                        ]
                    )
                )
                for i in range(600):
                    theta = PRIOR_MU + PRIOR_SD * rng.standard_normal(2)
                    y = (rng.random(32) < expit(theta[0] + theta[1] * Z)).astype(float)
                    s = np.array([rng.choice(3, p=e_gen[int(v)]) for v in y])
                    for est, e in (
                        ("measured_source_channel", e_src),
                        ("naive_ignorable", None),
                        ("known_application_channel", e_app),
                        ("oracle_complete", None),
                    ):
                        x, cov, ok = fit(s, y, est, e)
                        sd = float(np.sqrt(max(cov[1, 1], 0)))
                        rows.append(
                            {
                                "reader": reader,
                                "generating_channel": gen_name,
                                "seed": seed,
                                "author": i,
                                "estimator": est,
                                "a_true": theta[0],
                                "b_true": theta[1],
                                "a_hat": float(x[0]),
                                "b_hat": float(x[1]),
                                "b_sd": sd,
                                "b_covered": bool(abs(x[1] - theta[1]) <= 1.96 * sd),
                                "converged": ok,
                                "observed_symbols": int(np.sum(s > 0)),
                            }
                        )
            print(json.dumps({"reader": reader, "generating": gen_name}), flush=True)
    with (args.output / "fits.csv").open("x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    import pandas as pd

    df = pd.DataFrame(rows)
    df["se_b"] = (df.b_hat - df.b_true) ** 2
    df["se_a"] = (df.a_hat - df.a_true) ** 2
    summary = (
        df.groupby(["reader", "generating_channel", "estimator"])
        .agg(
            rmse_b=("se_b", lambda v: float(np.sqrt(v.mean()))),
            bias_b=("b_hat", "mean"),
            true_b=("b_true", "mean"),
            rmse_a=("se_a", lambda v: float(np.sqrt(v.mean()))),
            coverage_b=("b_covered", "mean"),
            converged=("converged", "mean"),
            observed=("observed_symbols", "mean"),
        )
        .reset_index()
    )
    summary["bias_b"] = summary.bias_b - summary.true_b
    summary.to_csv(args.output / "summary.csv", index=False)
    rng = np.random.default_rng(20260915)
    contrasts = []
    for (reader, gen), g in df.groupby(["reader", "generating_channel"]):
        piv = g.pivot_table(
            index=["seed", "author"], columns="estimator", values="se_b"
        )
        for a, b in (
            ("measured_source_channel", "naive_ignorable"),
            ("measured_source_channel", "oracle_complete"),
            ("measured_source_channel", "known_application_channel"),
        ):
            diffs = [(piv.loc[s][a] - piv.loc[s][b]).to_numpy() for s in SEEDS]
            rep = np.zeros(2000)
            for d in diffs:
                rep += d[rng.integers(len(d), size=(2000, len(d)))].mean(1) / len(diffs)
            lo, hi = float(np.quantile(rep, 0.025)), float(np.quantile(rep, 0.975))
            contrasts.append(
                {
                    "reader": reader,
                    "generating_channel": gen,
                    "contrast": f"{a}_minus_{b}",
                    "delta_mse_b": float(np.mean([d.mean() for d in diffs])),
                    "lower": lo,
                    "upper": hi,
                    "decision": "SUPPORTED_LOWER_LOSS"
                    if hi < 0
                    else "HIGHER_LOSS"
                    if lo > 0
                    else "INCONCLUSIVE",
                }
            )
    pd.DataFrame(contrasts).to_csv(args.output / "paired_contrasts.csv", index=False)
    (args.output / "channels.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(summary.to_string())
    print(pd.DataFrame(contrasts).to_string())


if __name__ == "__main__":
    main()
