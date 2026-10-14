#!/usr/bin/env python3
"""Exploratory complete-label comparison of conditional behavior response directions."""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_author_evidence_merge_hm102 as H
from scripts import run_grain_conditional_chart_hm114 as R


def run(out):
    """Compare all six fixed correspondences against all720 family permutations."""
    manifest = R.check(out)
    H.verify_hashes(out, H.load(out / "02/instrument_seal.json"))
    H.verify_hashes(out, H.load(out / "03/output_seal.json"))
    if (
        H.load(out / "verification.json")["status"]
        != "VERIFIED_FINITE_HM114_NOT_GLOBAL"
    ):
        raise ValueError("Finite primary experiment not verified")
    folder = out / "05_direction_diagnostic"
    folder.mkdir()
    with np.load(out / "02/fields.npz", allow_pickle=False) as z:
        fields = z["fields"]
    gains = (fields[:, :, :, 1, 1] - fields[:, :, :, 0, 1]) / 2
    # [bank,family,condition,unit,target]; one permutation acts on both conditions.
    permutations = list(itertools.permutations(range(6)))
    all_rows, null_rows, summaries = [], [], []
    alternate_error = 0.0
    for unit in range(2):
        source, target = gains[0, :, :, unit], gains[1, :, :, unit]
        sn, tn = np.linalg.norm(source, axis=2), np.linalg.norm(target, axis=2)
        if np.any(sn == 0) or np.any(tn == 0):
            raise ValueError("Zero effect direction has no angle;do not fabricate")
        cross = np.einsum(
            "fcq,gcq->cfg", source / sn[:, :, None], target / tn[:, :, None]
        )
        actual = float(np.mean(np.stack([cross[c].diagonal() for c in range(2)])))
        values = []
        for i, permutation in enumerate(permutations):
            value = float(np.mean(cross[:, np.arange(6), np.asarray(permutation)]))
            values.append(value)
            null_rows.append(
                {
                    "unit": unit,
                    "permutation_id": i,
                    "permutation": "-".join(map(str, permutation)),
                    "mean_direction_cosine": value,
                }
            )
        for f, c in itertools.product(range(6), range(2)):
            cosine = float(cross[c, f, f])
            direct = float(np.dot(source[f, c], target[f, c]) / (sn[f, c] * tn[f, c]))
            alternate_error = max(alternate_error, abs(cosine - direct))
            all_rows.append(
                {
                    "unit": unit,
                    "family": manifest["spec"]["targets"][f],
                    "condition": -1 if c == 0 else 1,
                    "source_gain_norm": float(sn[f, c]),
                    "target_gain_norm": float(tn[f, c]),
                    "direction_cosine": cosine,
                }
            )
        values = np.asarray(values)
        summaries.append(
            {
                "unit": unit,
                "matched_mean_cosine": actual,
                "permutation_mean": float(values.mean()),
                "permutation_min_max": [float(values.min()), float(values.max())],
                "count_at_least_matched": int(np.sum(values >= actual - 1e-12)),
                "exact_label_permutation_tail": float(
                    np.mean(values >= actual - 1e-12)
                ),
                "matched_minus_permutation_mean": actual - float(values.mean()),
                "not_personality_correlation_or_preregistered_significance": True,
            }
        )
    if alternate_error > manifest["spec"]["identity_atol"]:
        raise ValueError("Independent direction arithmetic differs")
    H.table(folder / "matched_directions.csv", all_rows)
    H.table(folder / "all_label_permutations.csv", null_rows)
    H.write(
        folder / "summary.json",
        {
            "status": "RESULT_AFTER_EXPLORATORY_DIRECTION_DIAGNOSTIC",
            "units": summaries,
            "alternate_max_error": alternate_error,
            "permutations_per_unit": 720,
            "same_behavior_clauses_and_current_carrier": True,
            "only_opportunity_wording_changes": True,
            "new_model_calls": 0,
            "fits": 0,
        },
    )
    H.write(
        folder / "seal.json",
        {
            "primary_verification_sha256": H.sha(out / "verification.json"),
            "diagnostic_code_sha256": H.sha(Path(__file__)),
            "outputs": H.seals(out, folder),
        },
    )
    print("DIRECTION_DIAGNOSTIC", summaries, "alternate error", alternate_error)


def main():
    """Write a separate result-after account without changing the primary battery."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output.resolve())


if __name__ == "__main__":
    main()
