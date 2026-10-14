#!/usr/bin/env python3
"""Apply learned source ensembles to frozen masked data with joint person draws."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

from grain_research.gptpro_foundation_v3.f13_masked_conditioning import (
    condition_source_fits,
)


def main():
    """Never tune against application truth; checkpoint whole masked cells."""
    parser = argparse.ArgumentParser()
    for key in ("source", "application", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    seeds = [20260915] if args.smoke else [20260915, 20260916, 20260917]
    worlds = ["iid", "ar06", "prior_shift"]
    masks = ["complete", "MCAR", "MAR", "MNAR"]
    files = [
        Path(__file__),
        Path("grain_research/gptpro_foundation_v3/f13_masked_conditioning.py"),
    ]
    files.extend(p for p in args.source.rglob("*.json") if not p.name.startswith("._"))
    files.extend(
        args.application / f"{w}_{m}_{s}.npz"
        for w in worlds
        for m in masks
        for s in seeds
    )
    protocol = {
        "seeds": seeds,
        "worlds": worlds,
        "masks": masks,
        "outer": 200,
        "inner": 200,
        "smoke": args.smoke,
        "source": "independent complete source authors; prior_shift uses old iid source",
        "rho": "fixed declared0/.6, not estimated",
        "rng": "[seed,13,4,world_index,mask_index,rho_index,author_index,2]",
        "scope": "source moment ensemble; Gaussian working posterior ignores MNAR selection",
        "files": [
            {
                "path": str(p.resolve()),
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            }
            for p in files
        ],
    }
    (args.output / "protocol_before_run.json").write_text(
        json.dumps(protocol, indent=2) + "\n"
    )
    started = time.monotonic()
    rows = []
    cells = 0
    for seed in seeds:
        for wi, world in enumerate(worlds):
            source_world = "iid" if world == "prior_shift" else world
            for mi, missing in enumerate(masks):
                data = np.load(args.application / f"{world}_{missing}_{seed}.npz")
                x = np.column_stack([np.ones(len(data["z"])), data["z"]])
                n = min(6, len(data["y"])) if args.smoke else len(data["y"])
                for ri, rho in enumerate((0.0, 0.6)):
                    root = args.source / f"{source_world}_{seed}_assumed{rho}"
                    fits = [
                        json.loads((root / f"fit_{b:03d}.json").read_text())["fit"]
                        for b in range(200)
                    ]
                    plugin = json.loads((root / "plugin.json").read_text())
                    if any(f["status"] != "SUPPORTED" for f in fits + [plugin]):
                        raise ValueError(
                            "invalid source fit retained; no silent reweighting"
                        )
                    lag = abs(np.arange(len(x))[:, None] - np.arange(len(x)))
                    kernel = rho**lag
                    means_all, covs_all, mcmeans, mccovs = [], [], [], []
                    for i in range(n):
                        keep = data["observed"][i]
                        means, covs = condition_source_fits(
                            fits,
                            x[keep],
                            data["y"][i, keep],
                            kernel[np.ix_(keep, keep)],
                        )
                        pm, pv = condition_source_fits(
                            [plugin],
                            x[keep],
                            data["y"][i, keep],
                            kernel[np.ix_(keep, keep)],
                        )
                        center = means.mean(axis=0)
                        deviations = means - center
                        within = covs.mean(axis=0)
                        between = deviations.T @ deviations / 200
                        recipe = [seed, 13, 4, wi, mi, ri, i, 2]
                        rng = np.random.default_rng(np.random.SeedSequence(recipe))
                        draws = means[:, None, :] + np.einsum(
                            "bji,bki->bjk",
                            rng.normal(size=(200, 200, 2)),
                            np.linalg.cholesky(covs),
                        )
                        flat = draws.reshape(-1, 2)
                        mc = flat.mean(axis=0)
                        dv = flat - mc
                        mcv = dv.T @ dv / len(flat)
                        means_all.append(means)
                        covs_all.append(covs)
                        mcmeans.append(mc)
                        mccovs.append(mcv)
                        for method, mu, cov in (
                            ("plugin", pm[0], pv[0]),
                            ("source_mixture", center, within + between),
                            ("mc_200x200", mc, mcv),
                        ):
                            for target, c in (
                                ("level", np.array([1.0, 0.0])),
                                ("delta", np.array([0.0, 2.0])),
                            ):
                                score = float(c @ mu)
                                variance = float(c @ cov @ c)
                                truth = float(c @ data["theta"][i])
                                half = 1.96 * np.sqrt(variance)
                                rows.append(
                                    {
                                        "world": world,
                                        "missingness": missing,
                                        "seed": seed,
                                        "rho_assumed": rho,
                                        "author_id": "u_"
                                        + hashlib.sha256(
                                            f"{seed}/13/application/{i}".encode()
                                        ).hexdigest()[:16],
                                        "author_index": i,
                                        "method": method,
                                        "target": target,
                                        "score": score,
                                        "variance": variance,
                                        "truth": truth,
                                        "covered": int(abs(score - truth) <= half),
                                        "width": 2 * half,
                                        "squared_error": (score - truth) ** 2,
                                        "within_source_variance": float(c @ within @ c),
                                        "between_source_variance": float(
                                            c @ between @ c
                                        ),
                                        "observed": int(keep.sum()),
                                    }
                                )
                    name = f"{world}_{missing}_{seed}_rho{rho}"
                    np.savez_compressed(
                        args.output / f"{name}.npz",
                        means=means_all,
                        covariances=covs_all,
                        mc_means=mcmeans,
                        mc_covariances=mccovs,
                    )
                    cells += 1
                    (args.output / "checkpoint.json").write_text(
                        json.dumps(
                            {
                                "cells": cells,
                                "rows": len(rows),
                                "elapsed": time.monotonic() - started,
                            }
                        )
                        + "\n"
                    )
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "predictions.csv", index=False)
    keys = ["world", "missingness", "seed", "rho_assumed", "method", "target"]
    metrics = []
    for key, g in frame.groupby(keys):
        low, high = proportion_confint(g.covered.sum(), len(g), method="wilson")
        metrics.append(
            dict(zip(keys, key, strict=True))
            | {
                "n": len(g),
                "coverage": g.covered.mean(),
                "wilson_low": low,
                "wilson_high": high,
                "mean_width": g.width.mean(),
                "mse": g.squared_error.mean(),
                "decision": "PASS" if low <= 0.95 <= high else "FAIL",
            }
        )
    pd.DataFrame(metrics).to_csv(args.output / "metrics.csv", index=False)
    summary = {
        "status": "MASKED_SOURCE_MIXTURE_COMPLETE",
        "cells": cells,
        "prediction_rows": len(rows),
        "metric_rows": len(metrics),
        "parameter_draws": cells * n * 200 * 200,
        "elapsed": time.monotonic() - started,
        "scope": protocol["scope"],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
