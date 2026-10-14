#!/usr/bin/env python3
"""Paired observationally equivalent worlds with independent source anchors."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grain_research.gptpro_foundation_v3.f11_anchor import (
    covariance_basis,
    fit_anchor,
    text_only_blup,
)
from grain_research.gptpro_foundation_v3.f11_mtmm import designs, moment_fit


def main():
    """Fit both mechanisms on source data and evaluate shared application text."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    seeds = [20260915] if args.smoke else [20260915, 20260916, 20260917]
    counts = [30, 6, 6] if args.smoke else [300, 100, 200]
    protocol = {
        "seeds": seeds,
        "counts": counts,
        "anchor_noise_variance": 0.25,
        "anchor_loading": 1,
        "anchor_independent": True,
        "source_anchor_only": True,
        "world_A": "f plus shared author method c",
        "world_B": "g=f+c and no shared method",
        "scope": "paired exact observational-equivalence construction, not two independent populations",
        "smoke_source_override": "30 needed for 21-variable covariance"
        if args.smoke
        else None,
    }
    (args.output / "protocol_before_run.json").write_text(
        json.dumps(protocol, indent=2) + "\n"
    )
    sigma = np.full((3, 3), 0.3)
    np.fill_diagonal(sigma, 1.0)
    fdesign, mdesign, _ = designs()
    metrics, fits, predictions = [], [], []
    for seed in seeds:
        data = {}
        for role_id, (role, n) in enumerate(
            zip(("source", "validation", "application"), counts, strict=True)
        ):
            streams = [
                np.random.default_rng(x)
                for x in np.random.SeedSequence([seed, 11, 3, role_id]).spawn(5)
            ]
            f = streams[0].multivariate_normal(np.zeros(3), sigma, size=n)
            c = streams[1].normal(scale=0.5, size=(n, 1))
            method = streams[2].normal(scale=0.5, size=(n, 3))
            error = streams[3].normal(scale=0.5, size=(n, 18))
            anchor_error = streams[4].normal(scale=0.5, size=(n, 3))
            g = f + c
            text = g @ fdesign.T + method @ mdesign.T + error
            np.savez(
                args.output / f"observations_{seed}_{role}.npz",
                text=text,
                anchor_A=f + anchor_error,
                anchor_B=g + anchor_error,
            )
            np.savez(
                args.output / f"truth_{seed}_{role}.npz",
                f=f,
                g=g,
                c=c,
                method=method,
                error=error,
                anchor_error=anchor_error,
            )
            data[role] = text, f, g, anchor_error
        y, source_f, source_g, noise = data["source"]
        naive = moment_fit(y, include_event=False)
        for world in ("A", "B"):
            target_cov = sigma if world == "A" else sigma + 0.25 * np.ones((3, 3))
            target_actor = 0.25 if world == "A" else 0.0
            source_anchor = (source_f if world == "A" else source_g) + noise
            models = {
                "unanchored_effective": naive,
                "source_anchored": fit_anchor(y, source_anchor),
                "oracle": {
                    "sigma": target_cov,
                    "method": 0.25,
                    "error": 0.25,
                    "actor": target_actor,
                    "mean": np.zeros(18),
                },
            }
            for name, model in models.items():
                (args.output / f"fit_{seed}_{world}_{name}.json").write_text(
                    json.dumps(
                        {
                            k: v.tolist() if isinstance(v, np.ndarray) else v
                            for k, v in model.items()
                        },
                        indent=2,
                    )
                    + "\n"
                )
                fits.append(
                    {
                        "seed": seed,
                        "world": world,
                        "method": name,
                        "actor_variance": model.get("actor", 0.0),
                        "true_actor_variance": target_actor,
                        "sigma_rmse": float(
                            np.sqrt(np.mean((model["sigma"] - target_cov) ** 2))
                        ),
                    }
                )
                for role in ("validation", "application"):
                    text, f, g, _ = data[role]
                    truth = f if world == "A" else g
                    p, v = text_only_blup(model, text)
                    se = np.sqrt(np.diag(v))
                    metrics.append(
                        {
                            "seed": seed,
                            "world": world,
                            "role": role,
                            "method": name,
                            "n": len(text),
                            "mse": float(np.mean((p - truth) ** 2)),
                            "coverage": float(np.mean(np.abs(p - truth) <= 1.96 * se)),
                        }
                    )
                    for i in range(len(text)):
                        for k in range(3):
                            predictions.append(
                                {
                                    "seed": seed,
                                    "world": world,
                                    "role": role,
                                    "method": name,
                                    "index": i,
                                    "trait": k,
                                    "prediction": p[i, k],
                                    "truth": truth[i, k],
                                    "variance": v[k, k],
                                }
                            )
    for filename, rows in (
        ("fits.csv", fits),
        ("metrics.csv", metrics),
        ("predictions.csv", predictions),
    ):
        with (args.output / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "status": "SOURCE_ANCHOR_WORLDS_COMPUTED_AUDIT_PENDING",
        "source_rank_without_anchor": int(
            np.linalg.matrix_rank(covariance_basis(False))
        ),
        "source_rank_with_anchor": int(np.linalg.matrix_rank(covariance_basis(True))),
        "covariance_components": 9,
        "unique_simulated_authors": len(seeds) * sum(counts),
        "prediction_rows": len(predictions),
        "application_anchor_used": False,
        "independence_of_anchor_is_assumed": True,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
