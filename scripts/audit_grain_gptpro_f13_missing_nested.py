#!/usr/bin/env python3
"""Replay all nested draws using explicit two-dimensional Gaussian formulas."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    """Check saved mixture moments and every MC draw stream, without refitting."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((args.input / "protocol_before_run.json").read_text())
    for file in protocol["files"]:
        if (
            hashlib.sha256(Path(file["path"]).read_bytes()).hexdigest()
            != file["sha256"]
        ):
            raise ValueError("frozen input changed")
    frame = pd.read_csv(args.input / "predictions.csv")
    keys = [
        "world",
        "missingness",
        "seed",
        "rho_assumed",
        "author_index",
        "method",
        "target",
    ]
    assert not frame.duplicated(keys).any()
    indexed = frame.set_index(keys)
    error = 0.0
    prefixes = 0
    for seed in protocol["seeds"]:
        for wi, world in enumerate(protocol["worlds"]):
            for mi, mask in enumerate(protocol["masks"]):
                for ri, rho in enumerate((0.0, 0.6)):
                    bundle = np.load(args.input / f"{world}_{mask}_{seed}_rho{rho}.npz")
                    for i, (means, covs) in enumerate(
                        zip(bundle["means"], bundle["covariances"], strict=True)
                    ):
                        center = means.mean(axis=0)
                        raw = (
                            covs.mean(axis=0)
                            + means.T @ means / len(means)
                            - np.outer(center, center)
                        )
                        rng = np.random.default_rng(
                            np.random.SeedSequence([seed, 13, 4, wi, mi, ri, i, 2])
                        )
                        normals = rng.normal(size=(200, 200, 2))
                        a = np.sqrt(covs[:, 0, 0])
                        b = covs[:, 1, 0] / a
                        c = np.sqrt(covs[:, 1, 1] - b * b)
                        samples = np.stack(
                            [
                                means[:, None, 0] + a[:, None] * normals[:, :, 0],
                                means[:, None, 1]
                                + b[:, None] * normals[:, :, 0]
                                + c[:, None] * normals[:, :, 1],
                            ],
                            axis=-1,
                        ).reshape(-1, 2)
                        mc = samples.mean(axis=0)
                        mcv = samples.T @ samples / len(samples) - np.outer(mc, mc)
                        error = max(
                            error,
                            float(np.max(abs(mc - bundle["mc_means"][i]))),
                            float(np.max(abs(mcv - bundle["mc_covariances"][i]))),
                        )
                        for method, mu, v in (
                            ("source_mixture", center, raw),
                            ("mc_200x200", mc, mcv),
                        ):
                            for target, k, scale in (("level", 0, 1), ("delta", 1, 2)):
                                row = indexed.loc[
                                    (world, mask, seed, rho, i, method, target)
                                ]
                                error = max(
                                    error,
                                    abs(scale * mu[k] - row.score),
                                    abs(scale * scale * v[k, k] - row.variance),
                                )
                        prefixes += 1
    if error > 1e-10:
        raise ValueError("nested replay mismatch")
    result = {
        "status": "NESTED_DRAW_AND_MIXTURE_REPLAY_PASS",
        "prefixes": prefixes,
        "parameter_draws": prefixes * 200 * 200,
        "max_error": error,
        "scope": "all saved Gaussian draws, mixture moments and mixture/MC target rows; source-conditioned posterior fit correctness separately tested, plugin/table coverage audit not included",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
