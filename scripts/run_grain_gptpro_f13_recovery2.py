#!/usr/bin/env python3
"""F13 recovery 2: dynamic missingness arms, variance components, MDC95 (T1)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from numpy.polynomial.hermite import hermgauss

from grain_research.gptpro_foundation_v3 import f13_dynamic_missing as dm
from grain_research.gptpro_foundation_v3 import f16_t1_source as t1

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/grain_gptpro_foundation_v3"
SEEDS = (20260915, 20260916, 20260917)
MASKS = ("complete", "mcar_20", "condition_20_60", "mnar_y")
PLAN = np.array([1.0, -1.0, 1.0, -1.0])


def sha(path):
    """File hash."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def groups(path, role):
    """Author -> time-ordered rows."""
    out = {}
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        if r["role"] != role:
            raise ValueError("wrong role")
        out.setdefault(r["author_id"], []).append(r)
    return {a: sorted(v, key=lambda r: r["t"]) for a, v in out.items()}


def score_cell(args):
    """Score all application authors of one seed x mask without truth."""
    output, seed, mask = args
    started = time.monotonic()
    models = [
        x["model"]
        for x in json.loads(
            (BASE / f"F16/t2_cold_full_r1/{seed}/bundle.json").read_text()
        )["source_models"]
    ]
    apps = groups(
        BASE / f"F05/source_full_r2/core_{seed}/application.jsonl", "application"
    )
    rows = []
    for index, author in enumerate(sorted(apps)):
        r = apps[author][:48]
        if [x["t"] for x in r] != list(range(48)):
            raise ValueError("prefix clock")
        z = np.array([x["z"] for x in r])
        y = np.array([x["y"] for x in r])
        rng = np.random.default_rng(
            np.random.SeedSequence([seed, 13, 7, MASKS.index(mask), index])
        )
        observed = rng.random(48) >= dm.mask_probability(mask, z, y)
        mean, cov, within, source = dm.score_author(models, z, y, observed)
        rows.append(
            {
                "author_id": author,
                "observed": int(observed.sum()),
                "mean": mean.tolist(),
                "cov": cov.tolist(),
                "within": within.tolist(),
                "source": source.tolist(),
            }
        )
    path = Path(output) / "scores" / f"{seed}_{mask}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "seed": seed,
                "mask": mask,
                "rows": rows,
                "seconds": time.monotonic() - started,
            }
        )
        + "\n"
    )
    return str(path)


def wilson(p, n):
    """Wilson95 interval."""
    v = 1.96**2
    c = (p + v / (2 * n)) / (1 + v / n)
    h = 1.96 * np.sqrt(p * (1 - p) / n + v / (4 * n * n)) / (1 + v / n)
    return c - h, c + h


def evaluate(output):
    """Open truth only after all scoring cells exist; metrics and paired CIs."""
    preds = []
    for seed in SEEDS:
        truth = {
            r["author_id"]: r
            for r in json.loads(
                (BASE / f"F05/source_full_r2/core_{seed}/truth.json").read_text()
            )
            if r["role"] == "application"
        }
        apps = groups(
            BASE / f"F05/source_full_r2/core_{seed}/application.jsonl", "application"
        )
        for mask in MASKS:
            cell = json.loads((output / "scores" / f"{seed}_{mask}.json").read_text())
            for row in cell["rows"]:
                a = row["author_id"]
                fut = apps[a][48:52]
                if [x["t"] for x in fut] != [48, 49, 50, 51]:
                    raise ValueError("future clock")
                t2 = truth[a]["states"][47]
                t3 = float(
                    np.mean(
                        [
                            x["y"] + truth[a]["b"] * (zp - x["z"])
                            for zp, x in zip(PLAN, fut, strict=True)
                        ]
                    )
                )
                for k, (name, value) in enumerate((("T2", t2), ("T3", t3))):
                    m, v = row["mean"][k], row["cov"][k][k]
                    preds.append(
                        {
                            "seed": seed,
                            "mask": mask,
                            "author_id": a,
                            "target": name,
                            "observed": row["observed"],
                            "truth": value,
                            "score": m,
                            "variance": v,
                            "lower": m - 1.96 * np.sqrt(v),
                            "upper": m + 1.96 * np.sqrt(v),
                            "source_share": row["source"][k][k] / v,
                        }
                    )
    metrics = []
    for seed in SEEDS:
        for mask in MASKS:
            for target in ("T2", "T3"):
                g = [
                    p
                    for p in preds
                    if (p["seed"], p["mask"], p["target"]) == (seed, mask, target)
                ]
                y = np.array([p["truth"] for p in g])
                s = np.array([p["score"] for p in g])
                cov = float(
                    np.mean([(p["lower"] <= p["truth"] <= p["upper"]) for p in g])
                )
                lo, hi = wilson(cov, len(g))
                metrics.append(
                    {
                        "seed": seed,
                        "mask": mask,
                        "target": target,
                        "n": len(g),
                        "mean_observed_events": float(
                            np.mean([p["observed"] for p in g])
                        ),
                        "mse": float(np.mean((s - y) ** 2)),
                        "bias": float(np.mean(s - y)),
                        "coverage": cov,
                        "wilson_low": lo,
                        "wilson_high": hi,
                        "gate": bool(lo <= 0.95 <= hi),
                        "width": float(np.mean([p["upper"] - p["lower"] for p in g])),
                        "source_share": float(np.mean([p["source_share"] for p in g])),
                    }
                )
    rng = np.random.default_rng(20260915)
    contrasts = []
    for target in ("T2", "T3"):
        for mask in MASKS[1:]:
            diffs = []
            for seed in SEEDS:
                base = {
                    p["author_id"]: (p["score"] - p["truth"]) ** 2
                    for p in preds
                    if (p["seed"], p["mask"], p["target"]) == (seed, "complete", target)
                }
                cur = {
                    p["author_id"]: (p["score"] - p["truth"]) ** 2
                    for p in preds
                    if (p["seed"], p["mask"], p["target"]) == (seed, mask, target)
                }
                diffs.append(np.array([cur[a] - base[a] for a in sorted(base)]))
            rep = np.zeros(2000)
            for d in diffs:
                rep += d[rng.integers(len(d), size=(2000, len(d)))].mean(1) / len(diffs)
            mean = float(np.mean([d.mean() for d in diffs]))
            lo, hi = float(np.quantile(rep, 0.025)), float(np.quantile(rep, 0.975))
            contrasts.append(
                {
                    "target": target,
                    "contrast": f"{mask}_minus_complete",
                    "delta_mse": mean,
                    "lower": lo,
                    "upper": hi,
                    "decision": "HIGHER_LOSS"
                    if lo > 0
                    else "SUPPORTED_LOWER_LOSS"
                    if hi < 0
                    else "INCONCLUSIVE",
                }
            )
    return preds, metrics, contrasts


def variance_components():
    """Source-estimated facets with declared truths; covariances kept."""
    rows = []
    for seed in SEEDS:
        b = json.loads((BASE / f"F16/t1_source_full_r1/{seed}/bundle.json").read_text())
        z = np.array(t1.CONDITIONS)
        basis = np.stack(
            [
                np.outer(np.ones(4), np.ones(4)).ravel(),
                np.outer(z, z).ravel(),
                (np.outer(np.ones(4), z) + np.outer(z, np.ones(4))).ravel(),
            ],
            axis=1,
        )
        est = []
        for m in b["source_models"]:
            sig = np.asarray(m["Sigma"])
            coef, *_ = np.linalg.lstsq(basis, sig.ravel(), rcond=None)
            remainder = sig.ravel() - basis @ coef
            est.append(
                [
                    coef[0],
                    coef[1],
                    coef[2],
                    m["sigma2"],
                    float(np.sqrt(np.mean(remainder**2))),
                ]
            )
        est = np.array(est)
        for i, (facet, true) in enumerate(
            (
                ("person (intercept a)", 1.0),
                ("person:condition (slope b variance)", 0.25),
                ("cov(person, person:condition)", 0.0),
                ("residual", 0.25),
                ("person:condition nonlinear remainder RMS", 0.0),
            )
        ):
            rows.append(
                {
                    "world": "F04_L",
                    "seed": seed,
                    "facet": facet,
                    "declared_true": true,
                    "source_estimate_mean": float(est[:, i].mean()),
                    "q025": float(np.quantile(est[:, i], 0.025)),
                    "q975": float(np.quantile(est[:, i], 0.975)),
                    "note": "four-cell source moment model; Sigma projected on [1,z] person structure",
                }
            )
        d = json.loads((BASE / f"F16/t2_cold_full_r1/{seed}/bundle.json").read_text())
        hx, hw = hermgauss(64)
        est = []
        for x in d["source_models"]:
            m = x["model"]
            mu, sig = np.asarray(m["prior"]["mu"]), np.asarray(m["prior"]["Sigma"])
            q, r = np.exp(2 * np.asarray(m["log_sigma"]))
            eta = mu[2] + np.sqrt(2 * sig[2, 2]) * hx
            rho = 0.99 * np.tanh(eta)
            state = float((hw / np.sqrt(np.pi)) @ (q / (1 - rho**2)))
            est.append([sig[0, 0], sig[1, 1], sig[0, 1], state, q, r])
        est = np.array(est)
        true_state = 0.25 / 0.6 * (np.arctanh(0.8) - np.arctanh(0.2))
        for i, (facet, true) in enumerate(
            (
                ("person (level a)", 1.0),
                ("person:condition (slope b)", 0.25),
                ("cov(person, person:condition)", 0.0),
                ("state (stationary, averaged over rho)", true_state),
                ("state innovation q", 0.25),
                ("residual", 0.25),
            )
        ):
            rows.append(
                {
                    "world": "F05_core",
                    "seed": seed,
                    "facet": facet,
                    "declared_true": true,
                    "source_estimate_mean": float(est[:, i].mean()),
                    "q025": float(np.quantile(est[:, i], 0.025)),
                    "q975": float(np.quantile(est[:, i], 0.975)),
                    "note": "F05 source prior over (a,b,eta) and log_sigma; state variance integrated over the eta prior",
                }
            )
    for facet in ("reader (text channel)", "material"):
        rows.append(
            {
                "world": "text_controlled_bank",
                "seed": "all",
                "facet": facet,
                "declared_true": "",
                "source_estimate_mean": "",
                "q025": "",
                "q975": "",
                "note": "NOT_IDENTIFIED separately: text bundles pair source b with channel draw b; source and reader spread form one component; material resampling not performed",
            }
        )
    return rows


def mdc_t1():
    """Two-occasion MDC95 on actual F04 L application runs (true change 0)."""
    out = []
    for seed in SEEDS:
        b = json.loads((BASE / f"F16/t1_source_full_r1/{seed}/bundle.json").read_text())
        apps = groups(
            BASE / f"F04/source_full_r1/L_{seed}/application.jsonl", "application"
        )
        d_all, se_noise, se_naive = [], [], []
        for author in sorted(apps):
            rows = apps[author][:32]
            halves = [
                [r for r in rows if r["t"] < 16],
                [r for r in rows if r["t"] >= 16],
            ]
            parts, weights, sig2, post_var = [], [], [], []
            for m in b["source_models"]:
                scores = []
                for h in halves:
                    counts, sums, _ = t1.sufficient(h)
                    mean, cov = t1.posterior(counts, sums, m)
                    scores.append(
                        (
                            float(t1.CONTRAST @ mean),
                            float(t1.CONTRAST @ cov @ t1.CONTRAST),
                        )
                    )
                counts = t1.sufficient(halves[0])[0]
                sig = np.asarray(m["Sigma"])
                k = sig + np.diag(m["sigma2"] / counts)
                w = np.linalg.solve(k, sig @ t1.CONTRAST)
                parts.append(scores)
                weights.append(w)
                sig2.append(m["sigma2"])
                post_var.append([scores[0][1], scores[1][1]])
            parts = np.array(parts)  # source, half, (mean,var)
            s1, s2 = parts[:, 0, 0], parts[:, 1, 0]
            d = float(s2.mean() - s1.mean())
            wbar = np.mean(weights, axis=0)
            noise_var = 2 * float(np.sum(wbar**2 / counts)) * float(np.mean(sig2))
            v1 = parts[:, 0, 1].mean() + s1.var()
            v2 = parts[:, 1, 1].mean() + s2.var()
            c12 = float(np.mean((s1 - s1.mean()) * (s2 - s2.mean())))
            d_all.append(d)
            se_noise.append(np.sqrt(noise_var))
            se_naive.append(np.sqrt(v1 + v2 - 2 * c12))
        d_all, se_noise, se_naive = map(np.array, (d_all, se_noise, se_naive))
        for label, se in (
            ("noise_predictive", se_noise),
            ("posterior_variance_formula", se_naive),
        ):
            inside = float(np.mean(np.abs(d_all) <= 1.96 * se))
            lo, hi = wilson(inside, len(d_all))
            out.append(
                {
                    "seed": seed,
                    "method": label,
                    "n_authors": len(d_all),
                    "mean_mdc95": float(np.mean(1.96 * se)),
                    "mdc95_in_reference_sd": float(
                        np.mean(1.96 * se) / b["reference_sigma"]
                    ),
                    "fraction_within_mdc": inside,
                    "wilson_low": lo,
                    "wilson_high": hi,
                    "gate_95": bool(lo <= 0.95 <= hi),
                    "sd_observed_change": float(d_all.std(ddof=1)),
                }
            )
    return out


def write_csv(path, rows):
    """CSV writer."""
    with Path(path).open("x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    """Score (parallel cells), then evaluate, variance components, MDC."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = {}
    for seed in SEEDS:
        for p in (
            BASE / f"F16/t2_cold_full_r1/{seed}/bundle.json",
            BASE / f"F05/source_full_r2/core_{seed}/application.jsonl",
            BASE / f"F16/t1_source_full_r1/{seed}/bundle.json",
            BASE / f"F04/source_full_r1/L_{seed}/application.jsonl",
        ):
            inputs[str(p.relative_to(ROOT))] = sha(p)
    (args.output / "protocol_before_run.json").write_text(
        json.dumps(
            {
                "protocol": "docs/GRAIN_F13_RECOVERY2_PROTOCOL_V1.md",
                "inputs": inputs,
                "code": {
                    p: sha(ROOT / p)
                    for p in (
                        "grain_research/gptpro_foundation_v3/f13_dynamic_missing.py",
                        "scripts/run_grain_gptpro_f13_recovery2.py",
                    )
                },
                "masks": MASKS,
                "seeds": SEEDS,
            },
            indent=1,
        )
        + "\n"
    )
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for path in pool.map(
            score_cell, [(str(args.output), s, m) for s in SEEDS for m in MASKS]
        ):
            print(
                json.dumps(
                    {"scored": path, "seconds": round(time.monotonic() - started, 1)}
                ),
                flush=True,
            )
    preds, metrics, contrasts = evaluate(args.output)
    write_csv(args.output / "predictions.csv", preds)
    write_csv(args.output / "missing_dynamic_metrics.csv", metrics)
    write_csv(args.output / "missing_dynamic_contrasts.csv", contrasts)
    write_csv(args.output / "variance_components.csv", variance_components())
    write_csv(args.output / "minimum_change.csv", mdc_t1())
    manifest = {
        str(p.relative_to(args.output)): sha(p)
        for p in sorted(args.output.rglob("*"))
        if p.is_file() and not p.name.startswith(".")
    }
    (args.output / "output_manifest.json").write_text(
        json.dumps(manifest, indent=1) + "\n"
    )
    print(json.dumps({"done": True, "seconds": round(time.monotonic() - started, 1)}))


if __name__ == "__main__":
    main()
