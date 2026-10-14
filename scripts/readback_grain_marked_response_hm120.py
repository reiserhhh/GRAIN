#!/usr/bin/env python3
"""Alternate numerical readback of all marked response moments and comparisons."""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_author_evidence_merge_hm102 as H


def audit(out):
    """Check retained bytes and recompute tags/actions with explicit matrix loops."""
    m, summary = H.load(out / "00/manifest.json"), H.load(out / "02/summary.json")
    H.verify_hashes(ROOT, m["code"] | m["source"])
    H.verify_hashes(out, m["inputs"])
    H.verify_hashes(out, H.load(out / "01/seal.json"))
    with np.load(out / "02/moments.npz", allow_pickle=False) as z:
        z = {k: z[k] for k in z.files}
    a, g, mask = z["attribution"], z["marks"], z["mask"]
    count = len(a)
    explicit = {
        name: np.zeros_like(z[name])
        for name in (
            "mass",
            "body_mass",
            "untagged_mass",
            "first",
            "connected",
            "misaligned_connected",
            "total_variation",
        )
    }
    for p in range(count):
        positions = np.flatnonzero(mask[p])
        average = np.mean(g[p, positions], axis=0)
        shifted = np.roll(g[p, positions], len(positions) // 2, axis=0)
        for channel in range(3):
            for mode in range(2):
                j = channel * 2 + mode
                for q in range(6):
                    values = a[p, channel, mode, q]
                    explicit["mass"][p, j, q] = sum(values)
                    explicit["body_mass"][p, j, q] = sum(values[positions])
                    explicit["untagged_mass"][p, j, q] = sum(values[~mask[p]])
                    explicit["total_variation"][p, j, q] = sum(abs(values))
                    explicit["first"][p, j, q] = values[positions] @ g[p, positions]
                    explicit["connected"][p, j, q] = values[positions] @ (
                        g[p, positions] - average
                    )
                    explicit["misaligned_connected"][p, j, q] = values[positions] @ (
                        shifted - average
                    )
    errors = {
        name: float(np.max(abs(value - z[name]))) for name, value in explicit.items()
    }
    end = z["endpoints"]
    # Replay the parent float32 grouping; reassociation changes its rounding.
    direct = (end[1, :, 0] - end[1, :, 1]) - (end[0, :, 0] - end[0, :, 1])
    alternate64 = (
        end.astype(float)[1, :, 0]
        - end.astype(float)[1, :, 1]
        - end.astype(float)[0, :, 0]
        + end.astype(float)[0, :, 1]
    )
    float32_rounding = float(np.max(abs(direct - alternate64)))
    errors["endpoint_difference"] = float(abs(direct - z["truth"]).max())
    residual = np.stack([a[p].sum((0, 3)) - direct[p] for p in range(count)])
    errors["quadrature_residual"] = float(abs(residual - z["residual"]).max())
    # The same arithmetic can be exact while integration is approximate. Never
    # turn an alternate readback PASS into a quadrature/native compatibility PASS.
    errors["recorded_completeness_RMS"] = abs(
        float(np.sqrt(np.mean(residual**2))) - summary["completeness_RMS"]
    )
    pairs = H.load(out / "00/pairs.json")
    matching_error = 0.0
    permutations = list(itertools.permutations(range(6)))
    if count == len(pairs):
        maps = {"scalar_mass": a.sum((1, 4))}
        for name in ("first", "connected", "misaligned_connected"):
            maps[name] = explicit[name].reshape(count, 3, 2, 6, 4).sum(1)
        for row in summary["comparisons"]:
            field = np.zeros((3, 2, 6, 2, 2, *maps[row["map"]].shape[1:]))
            for pair, value in zip(pairs, maps[row["map"]], strict=True):
                v, w, f, c, o = pair["key"]
                field[v, w, f, (c + 1) // 2, (o + 1) // 2] = value
            x = field[:, :, :, :, 1]
            if row["component"] == "actor_odd":
                x = (x - field[:, :, :, :, 0]) / 2
            s = x[0, 0, :, :, row["mode"]].reshape(6, 2, -1)
            t = x[row["variant"], row["wording"], :, :, row["mode"]].reshape(6, 2, -1)
            if row["status"] != "OBSERVED_DIRECTION":
                if (
                    min(
                        np.linalg.norm(s, axis=-1).min(),
                        np.linalg.norm(t, axis=-1).min(),
                    )
                    > 1e-12
                ):
                    raise AssertionError("Unavailability is not numerically supported")
                continue
            values = []
            for perm in permutations:
                values.append(
                    np.mean(
                        [
                            np.dot(s[f, c], t[perm[f], c])
                            / (np.linalg.norm(s[f, c]) * np.linalg.norm(t[perm[f], c]))
                            for f, c in itertools.product(range(6), range(2))
                        ]
                    )
                )
            matching_error = max(
                matching_error,
                float(np.max(abs(np.array(values) - row["permutation_values"]))),
            )
    errors["all_finite_matching_values"] = matching_error
    if (
        max(errors.values()) > 1e-10
        or "torch" in sys.modules
        or "transformers" in sys.modules
    ):
        raise AssertionError(errors)
    result = {
        "status": "NUMERIC_READBACK_PASSED_NOT_SCIENTIFIC_COMPLETION",
        "pairs": count,
        "errors": errors,
        "native_HM119_compatibility": "FAILED_RETAINED",
        "integration_RMS_fraction": summary["completeness_RMS_fraction"],
        "separate_float32_endpoint_rounding_max": float32_rounding,
        "failed_first_readback_preserved": "readback_first_failure.json and readback_first_attempt.py.txt",
        "same_agent_not_third_party": True,
        "new_model_calls_or_fits": [0, 0],
        "torch_or_transformers_loaded": False,
        "audit_code_sha256": H.sha(__file__),
    }
    H.write(out / "02/readback.json", result)
    H.write(out / "02/seal.json", H.seals(out, out / "02"))
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    audit(parser.parse_args().output)
