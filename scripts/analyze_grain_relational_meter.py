#!/usr/bin/env python3
"""Post-result algebraic explanation of semantic meter bias, without score tuning."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.relational_evidence import marginalize
from grain_research.relational_meter import (
    controlled_operator,
    enumerate_expectation,
    expected_joint,
    fractional_map,
)
from scripts.run_grain_query_moment_operator import sha, write


def run(out):
    """Derive from public controls only; do not open simulated/human outcome files."""
    folder = out / "meter_analysis_v1"
    if folder.exists():
        raise FileExistsError("Do not overwrite a completed analysis")
    folder.mkdir()
    bank = pd.read_csv(out / "public_bank.csv")
    with np.load(out / "readers/observer_tensors.npz", allow_pickle=False) as data:
        tensors = {name: data[name] for name in ("literal", "nli")}
    write(
        folder / "contract.json",
        {
            "stage": "Post-result explanatory derivation; not preregistered confirmation",
            "source_files": {
                str(p): sha(p)
                for p in (
                    out / "public_bank.csv",
                    out / "readers/observer_tensors.npz",
                    ROOT / "grain_research/relational_meter.py",
                    Path(__file__),
                )
            },
            "author_profiles_or_truth_opened": False,
            "new_scoring_arm": False,
            "new_training_or_neural_calls": False,
        },
    )
    marginal_map = np.array(
        [np.concatenate(marginalize(v.reshape(2, 2, 2, 2))) for v in np.eye(16)]
    ).T
    rng = np.random.default_rng(91600)
    operators, maps, errors, ranks = {}, [], [], []
    for z in (-1, 1):
        for reader, tensor in tensors.items():
            controls = np.zeros((4, 2, 2, 16))
            for row in bank[bank.condition == z].itertuples():
                controls[
                    2 * row.actor + row.channel,
                    row.orientation,
                    int(row.target_sign > 0),
                ] = tensor[row.case_id].ravel()
            a, b = controlled_operator(controls)
            operators[f"{z}:{reader}"] = {"offset": a, "operator": b}
            singular = np.linalg.svd(b, compute_uv=False)
            marginal_singular = np.linalg.svd(marginal_map @ b, compute_uv=False)
            ranks.append(
                {
                    "condition": z,
                    "reader": reader,
                    "joint_rank_at_1e_10": int(sum(singular > 1e-10)),
                    "marginal_rank_at_1e_10": int(sum(marginal_singular > 1e-10)),
                    "joint_singular_values": singular.tolist(),
                    "marginal_singular_values": marginal_singular.tolist(),
                }
            )
            for channel in range(2):
                mapping = fractional_map(a, b, channel=channel)
                maps.append(
                    {"condition": z, "reader": reader, "channel": channel, **mapping}
                )
            for _ in range(64):
                mu = rng.uniform(-0.6, 0.6, 4)
                error = np.max(
                    abs(expected_joint(a, b, mu) - enumerate_expectation(controls, mu))
                )
                for channel in range(2):
                    result = fractional_map(a, b, channel=channel, at=mu)
                    for group in range(4):
                        h = np.eye(4)[group] * 1e-5
                        plus = fractional_map(a, b, channel=channel, at=mu + h)["score"]
                        minus = fractional_map(a, b, channel=channel, at=mu - h)[
                            "score"
                        ]
                        error = max(
                            error,
                            abs((plus - minus) / 2e-5 - result["derivative"][group]),
                        )
                errors.append(error)
    if max(errors) > 1e-8:
        raise AssertionError("Affine expectation or rational derivative failed")
    write(folder / "operators.json", operators)
    write(folder / "fractional_maps.json", maps)
    write(folder / "ranks.json", ranks)
    rows = []
    for row in maps:
        rows.append(
            {
                "condition": row["condition"],
                "reader": row["reader"],
                "channel": row["channel"],
                "zero_profile_score": row["score"],
                "membership_mass": row["denominator_offset"],
                **{f"derivative_group_{i}": d for i, d in enumerate(row["derivative"])},
            }
        )
    pd.DataFrame(rows).to_csv(folder / "zero_profile_jacobian.csv", index=False)
    result = {
        "status": "PASS_EXPLANATORY_DERIVATION",
        "random_parameter_points": len(errors),
        "maximum_enumeration_or_derivative_error": max(errors),
        "new_neural_calls": 0,
        "scores_retuned": False,
    }
    write(folder / "summary.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
