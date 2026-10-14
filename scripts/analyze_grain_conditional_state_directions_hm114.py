#!/usr/bin/env python3
"""Compare captured pre-read state effects with the already reported readout effects."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_author_evidence_merge_hm102 as H
from scripts import run_grain_conditional_chart_hm114 as R


def run(out):
    """Use common token positions/native coordinates; do not learn any alignment."""
    manifest = R.check(out)
    H.verify_hashes(out, H.load(out / "01/raw_seal.json"))
    H.verify_hashes(out, H.load(out / "02/instrument_seal.json"))
    if (
        H.load(out / "verification.json")["status"]
        != "VERIFIED_FINITE_HM114_NOT_GLOBAL"
    ):
        raise ValueError("Primary finite experiment not verified")
    folder = out / "06_state_direction_diagnostic"
    folder.mkdir()
    bundle = H.load(out / "00/bundle.json")
    directions = np.asarray(bundle["directions"], float)
    with np.load(out / "02/fields.npz", allow_pickle=False) as z:
        fields = z["fields"]
    states = {}
    for row in H.load(out / "00/rows.json"):
        if row["owner"] == 1:
            if row["current"] != manifest["current_positions"]:
                raise ValueError("Different positions cannot be silently aligned")
            states[row["bank"], row["family"], row["condition"], row["polarity"]] = (
                R.raw_unit(out, row["id"])
            )
    rows, summaries, projection_error = [], [], 0.0
    for name in ("k", "v", "mean_v"):
        cosines = []
        ratios = []
        for family in range(6):
            for condition in (-1, 1):
                gains, offsets = [], []
                for bank in range(2):
                    low = states[bank, family, condition, -1][
                        "v" if name == "mean_v" else name
                    ]
                    high = states[bank, family, condition, 1][
                        "v" if name == "mean_v" else name
                    ]
                    low, high = np.asarray(low, float), np.asarray(high, float)
                    if name == "mean_v":
                        low, high = low.mean(0), high.mean(0)
                    gain, offset = (high - low) / 2, (high + low) / 2
                    gains.append(gain.ravel())
                    offsets.append(offset.ravel())
                    if name == "mean_v":
                        ratio = directions.shape[1] // gain.shape[0]
                        projected = np.einsum(
                            "qhd,hd->q", directions, np.repeat(gain, ratio, axis=0)
                        )
                        c = (condition + 1) // 2
                        actual = (
                            fields[bank, family, c, 1, 1, 0]
                            - fields[bank, family, c, 0, 1, 0]
                        ) / 2
                        projection_error = max(
                            projection_error, float(np.max(abs(projected - actual)))
                        )
                norms = [np.linalg.norm(gain) for gain in gains]
                if min(norms) == 0:
                    raise ValueError("Unobserved direction,not a zero similarity")
                cosine = float(np.dot(gains[0], gains[1]) / np.prod(norms))
                shift = float(np.linalg.norm(offsets[1] - offsets[0]))
                relative = shift / float(np.mean(norms))
                cosines.append(cosine)
                ratios.append(relative)
                rows.append(
                    {
                        "state": name,
                        "family": manifest["spec"]["targets"][family],
                        "condition": condition,
                        "source_gain_norm": float(norms[0]),
                        "target_gain_norm": float(norms[1]),
                        "effect_direction_cosine": cosine,
                        "offset_shift_norm": shift,
                        "offset_shift_over_mean_gain_norm": relative,
                    }
                )
        summaries.append(
            {
                "state": name,
                "matched_mean_cosine": float(np.mean(cosines)),
                "matched_cosine_min_max": [min(cosines), max(cosines)],
                "mean_offset_shift_over_mean_gain_norm": float(np.mean(ratios)),
                "not_information_or_personality_accuracy": True,
            }
        )
    if projection_error > manifest["spec"]["identity_atol"]:
        raise ValueError("Uniform readout difference not linked to actual mean V")
    H.table(folder / "state_directions.csv", rows)
    H.write(
        folder / "summary.json",
        {
            "status": "RESULT_AFTER_EXPLORATORY_PRE_READ_DIRECTION_ACCOUNT",
            "states": summaries,
            "uniform_readout_projection_max_error": projection_error,
            "actual_captures_reused": 48,
            "new_model_calls": 0,
            "fits": 0,
            "fixed_current_tokens_and_positions": True,
            "pretrained_state_still_required": True,
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
    print("PRE_READ_DIRECTIONS", summaries, "projection error", projection_error)


def main():
    """Keep this result-after diagnosis separate from primary acceptance."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output.resolve())


if __name__ == "__main__":
    main()
