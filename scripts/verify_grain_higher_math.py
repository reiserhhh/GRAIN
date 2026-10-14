#!/usr/bin/env python3
"""Extend HM-01 replay to product scores, local routing and observed atlas paths."""

import json
from itertools import product

import numpy as np
import pandas as pd
from run_grain_math_local import DEFAULT_OUT as LOCAL
from run_grain_math_local import TARGETS, load_program
from run_grain_math_response import ROOT, sha, write_json


def main():
    """Check saved arithmetic and hashes without neural calls or parameter fitting."""
    maximum, checked = 0.0, 0
    bank = [
        json.loads(s)
        for s in (LOCAL / "00/material_bank.jsonl").read_text().splitlines()
    ]
    for reader, target in product(("e5", "bge"), TARGETS):
        with np.load(LOCAL / f"03/cold_scores_{reader}_{target}.npz") as data:
            z = data["factors"]
            predicted = (
                z[:, 1]
                * np.clip((z[:, 0] + 1) / 2, 0, 1)
                * np.clip((z[:, 2] + 1) / 2, 0, 1)
                * np.clip((z[:, 3] + 1) / 2, 0, 1)
            )
            maximum = max(
                maximum, float(np.max(abs(predicted - data["declared_product"])))
            )
            checked += len(predicted)
            rows = pd.read_csv(LOCAL / "03/compound_effects.csv")
            rows = rows[(rows.reader == reader) & (rows.target == target)]
            for r in rows.to_dict("records"):
                ids = [int(r["base"]), int(r["added"])]
                if r["same_event"]:
                    ids = ids[:1]
                q = [i for i in ids if all(z[i, j] > 0 for j in (0, 2, 3))]
                base = int(r["base"])
                base_q = all(z[base, j] > 0 for j in (0, 2, 3))
                delta = (
                    float(np.mean(data["joint"][q]) - data["joint"][base])
                    if q and base_q
                    else None
                )
                if delta is None:
                    if not pd.isna(r["local_delta"]):
                        raise AssertionError(
                            "missing local evidence was numerically filled"
                        )
                else:
                    maximum = max(maximum, abs(delta - r["local_delta"]))
                checked += 1
        # Half-design metrics were saved rather than half-design score arrays.
        half = load_program(LOCAL / f"02/bundles/{reader}_{target}_half.npz")
        mask = np.array(
            [
                r["target"] == target and r["kind"] == "single" and r["role"] == "test"
                for r in bank
            ]
        )
        x = np.load(LOCAL / f"01B/embeddings/{reader}.npy")[mask].astype(float)
        factors = half["factors"]
        z = x @ factors["w"] + factors["bias"] - factors["mu"] @ factors["w"]
        phi = np.column_stack(
            [
                np.prod(z[:, [j for j in range(4) if m & (1 << j)]], axis=1)
                for m in range(1, 16)
            ]
        )
        model = half["joint"]
        y = (phi @ model["w"] + model["bias"] - model["mu"] @ model["w"])[:, 0]
        bits = np.array([r["bits"] for r, keep in zip(bank, mask, strict=True) if keep])
        true = np.all(bits[:, [0, 2, 3]] == 1, axis=1) * bits[:, 1]
        expected = pd.read_csv(LOCAL / "03/metrics.csv")
        row = expected[
            (expected.reader == reader)
            & (expected.target == target)
            & (expected.arm == "half_design_joint")
        ].iloc[0]
        maximum = max(
            maximum, abs(float(np.sqrt(np.mean((y - true) ** 2))) - row.unit_rmse)
        )
        checked += len(y)
    atlas = ROOT / "results/grain_math_response_v1/atlas_r1"
    for target, degree in product(TARGETS, (1, 2)):
        za, zc = [
            np.load(LOCAL / f"03/cold_scores_{reader}_{target}.npz")["factors"]
            for reader in ("e5", "bge")
        ]
        lookup = {
            (r["family"], tuple(r["bits"])): i
            for i, r in enumerate(bank)
            if r["target"] == target and r["kind"] == "single"
        }
        indices = [
            i
            for i, r in enumerate(bank)
            if r["target"] == target
            and r["kind"] == "single"
            and r["role"] == "test"
            and r["bits"][4] == 1
        ]
        informal = [
            lookup[(bank[i]["family"], tuple(bank[i]["bits"][:4] + [-1]))]
            for i in indices
        ]
        charts = [za[indices], za[informal], zc[indices]]
        maps = load_program(atlas / f"02/{target}_maps_degree{degree}.npz")

        def apply(key, x, maps=maps, degree=degree):
            """Rebuild a frozen map using explicit polynomial and affine algebra."""
            m = maps[key]
            phi = (
                x
                if degree == 1
                else np.column_stack(
                    [x]
                    + [
                        x[:, i] * x[:, j]
                        for i in range(x.shape[1])
                        for j in range(i + 1)
                    ]
                )
            )
            return phi @ m["w"] + m["bias"] - m["mu"] @ m["w"]

        with np.load(atlas / f"02/{target}_predictions_degree{degree}.npz") as expected:
            for i, j in product(range(3), repeat=2):
                if i != j:
                    key = f"{i}_to_{j}"
                    maximum = max(
                        maximum,
                        float(np.max(abs(apply(key, charts[i]) - expected[key]))),
                    )
                    checked += len(charts[i])
            for i, j, k in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (2, 1, 0)):
                key = f"{i}_to_{j}_to_{k}"
                actual = apply(f"{j}_to_{k}", apply(f"{i}_to_{j}", charts[i]))
                cycle = apply(f"{k}_to_{i}", actual)
                maximum = max(
                    maximum,
                    float(np.max(abs(actual - expected[key]))),
                    float(np.max(abs(cycle - expected[key + "_cycle"]))),
                )
                checked += 2 * len(actual)
    if maximum > 1e-10:
        raise AssertionError(maximum)
    hashes = 0
    for out, manifest in (
        (LOCAL, "04/artifact_hashes.json"),
        (atlas, "03/artifact_hashes.json"),
    ):
        for name, expected in json.loads((out / manifest).read_text()).items():
            if sha(out / name) != expected:
                raise AssertionError("artifact changed: " + name)
            hashes += 1
    hard = ROOT / "results/grain_math_response_v1/local_hard_objects_r2"
    old = ROOT / "results/grain_math_response_v1/local_hard_objects_r1"
    for reader in ("e5", "bge"):
        if sha(hard / f"01B/embeddings/{reader}.npy") != sha(
            old / f"01B/embeddings/{reader}.npy"
        ):
            raise AssertionError("hard-control repair changed embeddings")
    old_metrics, new_metrics = [pd.read_csv(p / "02/metrics.csv") for p in (old, hard)]
    for column in (
        "unit_rmse",
        "relevance_auc",
        "qualification_fp",
        "irrelevant_polarity_gain",
    ):
        np.testing.assert_array_equal(old_metrics[column], new_metrics[column])
    dest = ROOT / "results/grain_math_response_v1/higher_math_verification_r1"
    write_json(
        dest / "verification.json",
        {
            "status": "PASS",
            "max_formula_error": maximum,
            "checked_vectors_or_events": checked,
            "verified_artifact_hashes": hashes,
            "hard_repair_changed_scores": False,
            "scope": "extended product/half/local/observed-atlas replay; same agent, not human-independent",
        },
    )
    print(
        f"extended verification: {checked} vectors/events, {hashes} hashes, max error {maximum:.3g}"
    )


if __name__ == "__main__":
    main()
