#!/usr/bin/env python3
"""Probe geometry on finite worlds and the frozen HM-01 observed coordinates."""

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
from run_grain_math_local import DEFAULT_OUT as INPUT
from run_grain_math_local import TARGETS
from run_grain_math_response import ROOT, sha, write_csv, write_json

sys.path.insert(0, str(ROOT))
from grain_research.construction import math_atlas as A


def save_maps(path, models):
    """Archive source-fitted transitions as object-free arrays."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **{
            name + "__" + k: v
            for name, model in models.items()
            for k, v in model.items()
        },
    )


def run(out, source):
    """Run fixed chart comparisons without selecting a degree on the test set."""
    inputs = [source / "00/material_bank.jsonl", source / "02/model_freeze.json"]
    inputs += [
        source / f"03/cold_scores_{r}_{t}.npz"
        for r, t in product(("e5", "bge"), TARGETS)
    ]
    files = [Path(__file__), ROOT / "grain_research/construction/math_atlas.py"]
    signature = {
        "inputs": {str(p): sha(p) for p in inputs},
        "code": {str(p.relative_to(ROOT)): sha(p) for p in files},
    }
    contract = out / "00/protocol_before_run.json"
    if contract.exists():
        if json.loads(contract.read_text())["signature"] != signature:
            raise ValueError("changed atlas protocol: use new output, preserve history")
    else:
        write_json(
            contract,
            {
                "signature": signature,
                "seeds": [71, 173, 379],
                "worlds": ["affine", "curved", "collision"],
                "roles": "source240 / application240 per world/seed",
                "degrees": [1, 2],
                "new_encodings": 0,
                "api": 0,
                "real_charts": ["E5 formal", "E5 conversational", "BGE formal"],
                "real_fit": "10 source families; score 8 held families",
                "not_claimed": "smooth embedding manifold, psychometric curvature, sheaf cohomology",
            },
        )
    all_rows, scalar_rows, real_rows = [], [], []
    for world, seed in product(("affine", "curved", "collision"), (71, 173, 379)):
        src, app, evaluator = A.simulated_chart_world(seed, world)
        arrays = {"source" + str(i): v for i, v in enumerate(src)}
        arrays.update({"application" + str(i): v for i, v in enumerate(app)})
        arrays.update({"truth_" + k: v for k, v in evaluator.items()})
        path = out / f"01/{world}_{seed}"
        path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path / "observations_and_evaluator.npz", **arrays)
        for degree in (1, 2):
            rows, models, predictions = A.atlas_statistics(src, app, degree)
            save_maps(path / f"maps_degree{degree}.npz", models)
            np.savez_compressed(path / f"predictions_degree{degree}.npz", **predictions)
            all_rows.extend({"world": world, "seed": seed, **r} for r in rows)
            back = predictions["2_to_0"]
            scalar_rows.append(
                {
                    "world": world,
                    "seed": seed,
                    "degree": degree,
                    "signed_state_rmse": float(
                        np.sqrt(
                            np.mean((back[:, 0] - evaluator["application"][:, 0]) ** 2)
                        )
                    ),
                    "second_state_rmse": float(
                        np.sqrt(
                            np.mean((back[:, 1] - evaluator["application"][:, 1]) ** 2)
                        )
                    ),
                }
            )
    bank = [json.loads(s) for s in inputs[0].read_text().splitlines()]
    for target in TARGETS:
        za, zc = [
            np.load(source / f"03/cold_scores_{reader}_{target}.npz")["factors"]
            for reader in ("e5", "bge")
        ]
        lookup = {
            (r["target"], r["family"], tuple(r["bits"])): i
            for i, r in enumerate(bank)
            if r["kind"] == "single"
        }
        indices = {
            role: [
                i
                for i, r in enumerate(bank)
                if r["target"] == target
                and r["kind"] == "single"
                and r["role"] == role
                and r["bits"][4] == 1
            ]
            for role in ("source", "test")
        }
        groups = []
        for role in ("source", "test"):
            formal = indices[role]
            informal = [
                lookup[(target, bank[i]["family"], tuple(bank[i]["bits"][:4] + [-1]))]
                for i in formal
            ]
            groups.append([za[formal], za[informal], zc[formal]])
        for degree in (1, 2):
            rows, models, predictions = A.atlas_statistics(*groups, degree)
            real_rows.extend({"target": target, **r} for r in rows)
            save_maps(out / f"02/{target}_maps_degree{degree}.npz", models)
            np.savez_compressed(
                out / f"02/{target}_predictions_degree{degree}.npz", **predictions
            )
    write_csv(out / "01/chart_metrics.csv", all_rows)
    write_csv(out / "01/signed_state_recovery.csv", scalar_rows)
    write_csv(out / "02/observed_chart_metrics.csv", real_rows)
    write_json(out / "01/collision_witness.json", A.collision_witness())
    # Alternate explicit affine-polynomial replay, without calling transition().
    maximum = 0.0
    for world, seed in product(("affine", "curved", "collision"), (71, 173, 379)):
        path = out / f"01/{world}_{seed}"
        with np.load(path / "observations_and_evaluator.npz") as observations:
            for degree in (1, 2):
                with (
                    np.load(path / f"maps_degree{degree}.npz") as maps,
                    np.load(path / f"predictions_degree{degree}.npz") as predictions,
                ):
                    for i in range(3):
                        for j in range(3):
                            if i == j:
                                continue
                            key = f"{i}_to_{j}"
                            x = observations[f"application{i}"]
                            phi = (
                                x
                                if degree == 1
                                else np.c_[
                                    x, x[:, 0] ** 2, x[:, 1] * x[:, 0], x[:, 1] ** 2
                                ]
                            )
                            w, mu, bias = (
                                maps[key + "__" + k] for k in ("w", "mu", "bias")
                            )
                            actual = phi @ w + bias - mu @ w
                            maximum = max(
                                maximum, float(np.max(abs(actual - predictions[key])))
                            )
    if maximum > 1e-10:
        raise AssertionError(maximum)
    write_json(
        out / "03/verification.json",
        {
            "status": "PASS",
            "max_formula_error": maximum,
            "scope": "108 simulated direct transitions, alternate algebra; not independent human review",
        },
    )
    paths = sorted(
        p
        for p in out.rglob("*")
        if p.is_file()
        and not p.name.startswith("._")
        and p.name != "artifact_hashes.json"
    )
    write_json(
        out / "03/artifact_hashes.json",
        {str(p.relative_to(out)): sha(p) for p in paths},
    )
    print(f"atlas: 9 worlds, 2 targets, no new encodings, replay error={maximum:.3g}")


def main():
    """Execute a bounded, deterministic offline atlas diagnostic."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=INPUT)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "results/grain_math_response_v1/atlas_r1"
    )
    args = parser.parse_args()
    run(args.out, args.source)


if __name__ == "__main__":
    main()
