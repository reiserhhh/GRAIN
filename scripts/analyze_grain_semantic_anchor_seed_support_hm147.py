#!/usr/bin/env python3
"""Signed source-seed support for the fixed HM147 witness; no rescoring/refit."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_semantic_anchor_hm147 as H


def run(out):
    """Decompose empirical mean energy into cross-seed agreement and dispersion.

    Shared estimated source units preclude treating the cross term as an
    unbiased population signal estimate. The signed finite identity is exact.
    """
    H.check(out)
    destination = out / "04_seed_support"
    destination.mkdir(exist_ok=False)
    rows, _, z, _, _, _, _ = H.source_objects(out)
    seeds = sorted({r["seed"] for r in rows})
    if seeds != [11, 23, 37]:
        raise ValueError("Original three source seeds required")
    phi = H.M.tensor_features(z)
    features = H.M.feature_sectors()
    results, totals, max_error = [], [], 0.0
    for target in H.TARGETS:
        weights = []
        for seed in seeds:
            weights.append(
                H.M.anchor_weights(
                    [r["axis"] if r["seed"] == seed else "" for r in rows],
                    [r["pole"] for r in rows],
                    (target,),
                )[:, 0]
            )
        weights = np.stack(weights)
        witnesses = weights @ phi
        mean = witnesses.mean(axis=0)
        for mask in range(16):
            part = witnesses[:, features == mask]
            gram = part @ part.T
            energy = float(mean[features == mask] @ mean[features == mask])
            cross = float((gram.sum() - np.trace(gram)) / 6)
            dispersion = float(np.sum((part - part.mean(axis=0)) ** 2) / 2)
            kernel = np.ones((len(z), len(z)))
            for v in range(4):
                if mask & (1 << v):
                    kernel *= z[:, v] @ z[:, v].T
            gram_alternate = weights @ kernel @ weights.T
            error = max(
                abs(energy - cross - dispersion / 3),
                float(abs(gram - gram_alternate).max()),
            )
            max_error = max(max_error, error)
            results.append(
                {
                    "target": target,
                    "sector": mask,
                    "order": mask.bit_count(),
                    "mean_energy": energy,
                    "cross_seed_energy": cross,
                    "dispersion_over3": dispersion / 3,
                    "identity_and_kernel_error": error,
                }
            )
        for order in range(5):
            selected = [
                r for r in results if r["target"] == target and r["order"] == order
            ]
            totals.append(
                {
                    "target": target,
                    "order": order,
                    **{
                        key: sum(r[key] for r in selected)
                        for key in (
                            "mean_energy",
                            "cross_seed_energy",
                            "dispersion_over3",
                        )
                    },
                }
            )
    if max_error > 1e-10:
        raise ValueError("Alternate source-seed identity failed")
    H.write_csv(destination / "source_sector_support.csv", results)
    H.write_csv(destination / "support_by_order.csv", totals)
    H.write_json(
        destination / "summary.json",
        {
            "seeds": seeds,
            "maximum_error": max_error,
            "support_by_order": totals,
            "interpretation": "Signed finite cross-seed agreement, not a noise-free human-trait fraction or unbiased population estimator; all source units and primary scores unchanged",
        },
    )
    H.write_json(
        destination / "seal.json",
        {
            name: H.digest(destination / name)
            for name in (
                "source_sector_support.csv",
                "support_by_order.csv",
                "summary.json",
            )
        }
        | {"code": H.digest(Path(__file__))},
    )
    print((destination / "support_by_order.csv").read_text())
    print("maximum_error=" + str(max_error))


if __name__ == "__main__":
    run(H.DEFAULT)
