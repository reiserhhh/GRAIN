#!/usr/bin/env python3
"""Trace the frozen FA compression through its exterior-square action.

Companion to the fixed HM97 banks; no fitting and no parameter selection.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_grain_cycle_capacity_hm97 import check, settings
from scripts.run_grain_observer_parity_hm96 import load, seal, sha, verify, write


def run():
    """Authenticate every official point and exact induced area transformation."""
    spec, out = settings()
    check(out)
    verify(out, load(out / "02_seal.json"))
    verify(out, load(out / "03_seal.json"))
    if load(out / "verification.json")["status"] != "VERIFIED_FINITE_HM97_NOT_GLOBAL":
        raise ValueError("Primary independent arithmetic required first")
    dest = out / "05_compression_companion"
    dest.mkdir()
    bundle = load(ROOT / spec["bundle"])
    if bundle["feature_names"] != [f"mean::svd_{i:03d}" for i in range(24)]:
        raise ValueError("Only the declared complete affine instrument is covered")
    w = np.asarray(bundle["factor_loadings"])
    psi = np.maximum(bundle["noise_variance"], 1e-12)
    posterior_map = np.linalg.solve(w.T @ w + np.diag(psi), w.T)
    norm = np.asarray(bundle["factor_signs"]) / np.asarray(bundle["norm_scale"])
    linear = posterior_map * norm / np.asarray(bundle["feature_scale"])[:, None]
    offset = -(
        np.asarray(bundle["feature_center"]) / np.asarray(bundle["feature_scale"])
        + np.asarray(bundle["fa_mean"])
    ) @ posterior_map * norm - np.asarray(bundle["norm_mean"]) / np.asarray(
        bundle["norm_scale"]
    )
    raw_pairs, factor_pairs = (
        list(itertools.combinations(range(24), 2)),
        list(itertools.combinations(range(4), 2)),
    )
    exterior = np.array(
        [
            [
                linear[i, p] * linear[j, q] - linear[i, q] * linear[j, p]
                for p, q in factor_pairs
            ]
            for i, j in raw_pairs
        ]
    )
    nodes = np.load(out / "02/official_coordinates.npz")
    measured = np.load(out / "03/measured_capacity.npz")
    point_error = float(
        np.max(np.abs(nodes["svd"] @ linear + offset - nodes["factors"]))
    )
    errors, scope_ranks = [], []
    for context in range(4):
        actual = measured[f"area_cycle_{context}_0"]
        rebuilt = measured[f"area_cycle_{context}_1"] @ exterior
        errors.append(float(np.max(np.abs(actual - rebuilt))))
        scope_ranks.append(
            {
                "context": context,
                "raw_cycle_rank": int(
                    np.linalg.matrix_rank(
                        measured[f"area_cycle_{context}_1"].reshape(324, 276), tol=1e-10
                    )
                ),
                "factor_cycle_rank": int(
                    np.linalg.matrix_rank(actual.reshape(324, 6), tol=1e-10)
                ),
            }
        )
    if max(point_error, *errors) > spec["arithmetic_atol"]:
        raise ValueError("Actual factor compression does not match its exterior lift")
    np.savez_compressed(
        dest / "frozen_compression.npz", linear=linear, offset=offset, exterior=exterior
    )
    summary = {
        "status": "VERIFIED_AFFINE_COMPRESSION_COMPANION",
        "official_point_error": point_error,
        "area_pushforward_errors": errors,
        "linear_rank": int(np.linalg.matrix_rank(linear)),
        "exterior_rank": int(np.linalg.matrix_rank(exterior)),
        "scope_ranks": scope_ranks,
        "fit_calls": 0,
        "scope_complete": False,
        "not_a_new_exterior_algebra_theorem": True,
        "companion_code": {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))},
    }
    write(dest / "summary.json", summary)
    write(out / "05_seal.json", seal(out, dest))
    print(summary, flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run()
