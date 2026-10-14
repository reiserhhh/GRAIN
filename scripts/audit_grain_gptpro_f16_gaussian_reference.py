#!/usr/bin/env python3
"""Dense-Gaussian replay of reference integration and corrected comparison tables."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss
from scipy.special import logsumexp

from grain_research.gptpro_foundation_v3 import f16_dynamic_text as text
from grain_research.gptpro_foundation_v3.f14_metadata import sha256_file
from grain_research.gptpro_foundation_v3.f16_dynamic_gaussian_reference import (
    conditional_filter,
    joint_future,
)
from scripts.audit_grain_gptpro_f16_dynamic_text import metric_record, projected_targets
from scripts.run_grain_gptpro_f16_t1_text import save


def dense(model, eta, arrays, author):
    """Condition complete observation/target covariance, not sequential recursions."""
    z, y, precision, mask = arrays
    visible = mask[author] & (precision[author] > 0)
    times = np.flatnonzero(visible)
    target_time = np.arange(z.shape[1] - 1, z.shape[1] + 4)
    mu, sig = np.asarray(model["prior"]["mu"]), np.asarray(model["prior"]["Sigma"])
    ab_mu = mu[:2] + sig[:2, 2] * (eta - mu[2]) / sig[2, 2]
    ab_cov = sig[:2, :2] - np.outer(sig[:2, 2], sig[2, :2]) / sig[2, 2]
    rho = 0.99 * np.tanh(eta)
    q, r = np.exp(2 * np.asarray(model["log_sigma"]))
    stationary = q / (1 - rho * rho)
    h = np.column_stack([np.ones(len(times)), z[author, times]])
    g = np.vstack([[0, 0], np.column_stack([np.ones(4), text.PLAN])])
    v = (
        h @ ab_cov @ h.T
        + stationary * rho ** abs(times[:, None] - times[None, :])
        + np.diag(r / precision[author, times])
    )
    cross = g @ ab_cov @ h.T + stationary * rho ** (
        target_time[:, None] - times[None, :]
    )
    total = (
        g @ ab_cov @ g.T
        + stationary * rho ** abs(target_time[:, None] - target_time[None, :])
        + np.diag([0, r, r, r, r])
    )
    residual = y[author, times] - h @ ab_mu
    solved = np.linalg.solve(v, np.column_stack([residual, cross.T]))
    mean = g @ ab_mu + cross @ solved[:, 0]
    covariance = total - cross @ solved[:, 1:]
    chol = np.linalg.cholesky(v)
    ll = -0.5 * (
        len(times) * np.log(2 * np.pi)
        + 2 * np.log(np.diag(chol)).sum()
        + residual @ solved[:, 0]
    )
    if (
        not np.isfinite(mean).all()
        or not np.isfinite(covariance).all()
        or not np.isfinite(ll)
    ):
        raise ValueError("nonfinite dense reference cannot pass a maximum-error check")
    return mean, covariance, ll


def main():
    """Report exact audited scope rather than claiming a full second pipeline."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("original", "refined", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    proto = json.loads((args.refined / "protocol.json").read_text())
    for p, h in proto["code_hashes"].items():
        assert sha256_file(p) == h
    assert (
        sha256_file(args.original / "output_manifest.json")
        == proto["original_manifest_hash"]
    )
    for p, h in json.loads((args.refined / "manifest.json").read_text()).items():
        assert sha256_file(args.refined / p) == h
    full = pd.read_csv(args.refined / "scores.csv")
    old = pd.read_csv(args.original / "scores.csv")
    categorical = ~full.arm.isin(["naive_gaussian", "eligible_continuous"])
    keys = ["seed", "author_id", "arm", "target"]
    assert (
        full[categorical]
        .sort_values(keys)
        .reset_index(drop=True)
        .equals(
            old[~old.arm.isin(["naive_gaussian", "eligible_continuous"])]
            .sort_values(keys)
            .reset_index(drop=True)
        )
    )
    error, comparisons, objects = 0.0, 0, 0
    for seed in (20260915, 20260916, 20260917):
        folder = args.original / str(seed)
        bundle = json.loads((folder / "bundle.json").read_text())
        inputs = [
            json.loads(x)
            for x in (folder / "application_inputs.jsonl").read_text().splitlines()
        ]
        ev = {
            (r["author_id"], r["event_id"]): r
            for r in json.loads(
                (folder / "application_rendering_evaluator.json").read_text()
            )
        }
        for arm in ("naive_gaussian", "eligible_continuous"):
            arrays, _ = text.prepare([inputs[0], inputs[-1]], bundle, arm, ev)
            for b in (0, 100, 199):
                model = bundle["source_models"][b]
                x, w = hermgauss(64)
                w /= np.sqrt(np.pi)
                eta = (
                    model["prior"]["mu"][2]
                    + np.sqrt(2 * model["prior"]["Sigma"][2][2]) * x
                )
                m, c, ll = conditional_filter(model, eta, arrays)
                means, within, parameters = joint_future(m, c, eta, model["log_sigma"])
                for i in range(2):
                    alternative = [dense(model, e, arrays, i) for e in eta]
                    dm = np.stack([v[0] for v in alternative])
                    dc = np.stack([v[1] for v in alternative])
                    dll = np.array([v[2] for v in alternative])
                    if not all(
                        np.isfinite(value).all()
                        for value in (dm, dc, dll, means, within, parameters, ll)
                    ):
                        raise ValueError("nonfinite conditional comparison")
                    weights = np.exp(dll + np.log(w) - logsumexp(dll + np.log(w)))
                    avg = weights @ dm
                    cv = np.einsum("n,nij->ij", weights, dc) + np.einsum(
                        "n,ni,nj->ij", weights, dm - avg, dm - avg
                    )
                    weights2 = np.exp(
                        ll[:, i] + np.log(w) - logsumexp(ll[:, i] + np.log(w))
                    )
                    avg2 = weights2 @ means[:, i]
                    cv2 = np.einsum(
                        "n,nij->ij", weights2, (within + parameters)[:, i]
                    ) + np.einsum(
                        "n,ni,nj->ij", weights2, means[:, i] - avg2, means[:, i] - avg2
                    )
                    error = max(
                        error,
                        float(abs(dm - means[:, i]).max()),
                        float(abs(dc - (within + parameters)[:, i]).max()),
                        float(abs(dll - ll[:, i]).max()),
                        float(abs(avg - avg2).max()),
                        float(abs(cv - cv2).max()),
                    )
                    comparisons += len(eta)
            records = json.loads(
                (args.refined / str(seed) / f"{arm}_scores.json").read_text()
            )
            for i, score in enumerate(records):
                for t, (m, v) in projected_targets(score).items():
                    s = score["targets"][t]
                    row = full[
                        (full.seed == seed)
                        & (full.arm == arm)
                        & (full.target == t)
                        & (full.author_id == inputs[i]["author_id"])
                    ].iloc[0]
                    error = max(
                        error,
                        abs(m - row.score),
                        abs(v - row.variance),
                        abs(v - sum(s["components"].values())),
                    )
                objects += 1
    table = pd.read_csv(args.refined / "metrics_by_seed.csv")
    for row in table.to_dict("records"):
        sub = full[
            (full.seed == row["seed"])
            & (full.arm == row["arm"])
            & (full.target == row["target"])
        ]
        error = max(error, max(abs(row[k] - v) for k, v in metric_record(sub).items()))
    pairs = pd.read_csv(args.refined / "paired_contrasts.csv")
    for row in pairs.to_dict("records"):
        sub = full[full.target == row["target"]]
        p = sub[sub.arm == "hybrid_joint"].merge(
            sub[sub.arm == row["reference"]],
            on=["seed", "author_id"],
            suffixes=("_p", "_r"),
            validate="one_to_one",
        )
        d = (p.score_p - p.truth_p) ** 2 - (p.score_r - p.truth_r) ** 2
        rng, draws = np.random.default_rng(165), np.zeros(2000)
        for seed in (20260915, 20260916, 20260917):
            x = d[p.seed == seed].to_numpy()
            draws += x[rng.integers(0, len(x), (2000, len(x)))].mean(1) / 3
        lo, hi = np.quantile(draws, [0.025, 0.975])
        error = max(
            error,
            abs(d.mean() - row["delta_mse"]),
            abs(lo - row["ci_low"]),
            abs(hi - row["ci_high"]),
        )
    if error > 1e-10:
        raise ValueError(f"independent reference audit mismatch: {error}")
    save(
        args.output / "summary.json",
        {
            "status": "DENSE_CONDITIONAL_AND_CORRECTED_TABLE_REPLAY_PASS",
            "dense_parameter_author_cases": comparisons,
            "corrected_objects": objects,
            "metrics": len(table),
            "paired_intervals": len(pairs),
            "max_error": error,
            "primary_categorical_rows_unchanged": True,
            "scope": "dense replay first/last author and3sourcefits perseed/arm;all1200saved reference objects/table arithmetic;notall-fits second implementation",
            "seconds": time.monotonic() - start,
        },
    )


if __name__ == "__main__":
    main()
