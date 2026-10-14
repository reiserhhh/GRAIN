#!/usr/bin/env python3
"""Keep HM109 score/log-odds quantities separate; never interpret mixed-unit RMS."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_author_evidence_merge_hm102 as H


def rms(values):
    """Return a unit-specific RMS, or an absent result for absent captures."""
    return float(np.sqrt(np.mean(np.square(values)))) if len(values) else None


def compute(source):
    """Reconstruct all unit-specific author comparisons from the sealed full table."""
    with (source / "profile_accounts.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    result, transport = [], []
    cohorts = sorted({r["cohort"] for r in rows})
    for cohort in cohorts:
        for bank in range(3):
            for unit, name in enumerate(("score", "log_odds")):
                picked = [
                    r
                    for r in rows
                    if r["cohort"] == cohort
                    and int(r["bank"]) == bank
                    and r["unit_index"] == str(unit)
                ]
                values = {}
                for key in (
                    "native_difference",
                    "additive_difference",
                    "lift_difference",
                    "directed_response_difference",
                    "random_response_difference",
                ):
                    values[key] = [float(r[key]) for r in picked if r[key] != ""]
                result.append(
                    {
                        "cohort": cohort,
                        "bank": bank,
                        "unit": name,
                        "profile_pairs": 12,
                        "target_readings": len(picked),
                        **{key + "_RMS": rms(value) for key, value in values.items()},
                        "paired_cancellation_error": max(
                            abs(
                                float(r["native_difference"])
                                - float(r["additive_difference"])
                                - float(r["lift_difference"])
                            )
                            for r in picked
                        ),
                    }
                )
        for unit, name in enumerate(("score", "log_odds")):
            source_rows = {
                (int(r["direction"]), int(r["target_index"])): r
                for r in rows
                if r["cohort"] == cohort
                and r["bank"] == "0"
                and r["unit_index"] == str(unit)
            }
            known_rows = {
                (int(r["direction"]), int(r["target_index"])): r
                for r in rows
                if r["cohort"] == cohort
                and r["bank"] == "1"
                and r["unit_index"] == str(unit)
            }
            if not all(
                r["directed_response_difference"] != "" for r in known_rows.values()
            ):
                transport.append(
                    {
                        "cohort": cohort,
                        "unit": name,
                        "response_cosine": None,
                        "reason": "known_bank_incomplete_capture",
                        "same_direction_count": None,
                    }
                )
                continue
            keys = sorted(source_rows)
            a = np.array(
                [float(source_rows[k]["directed_response_difference"]) for k in keys]
            )
            b = np.array(
                [float(known_rows[k]["directed_response_difference"]) for k in keys]
            )
            diagonal = [k for k in keys if k[0] // 2 == k[1] and k[0] % 2 == unit]
            transport.append(
                {
                    "cohort": cohort,
                    "unit": name,
                    "response_cosine": float(
                        a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
                    ),
                    "same_direction_count": sum(
                        float(source_rows[k]["directed_response_difference"])
                        * float(known_rows[k]["directed_response_difference"])
                        > 0
                        for k in diagonal
                    ),
                    "diagonal_comparisons": len(diagonal),
                    "reason": "source_selected_witnesses_existing_known_bank_not_blind_replication",
                }
            )
    return {
        "unit_accounts": result,
        "known_response_transport": transport,
        "primary_mixed_unit_RMS": "diagnostic only; do not interpret as an effect size",
        "new_forwards": 0,
        "new_fits": 0,
        "original_scope_complete": False,
    }


def main():
    """Freeze an immutable companion without rewriting the primary packet."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("analyze", "verify"))
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT
        / "results/grain_math_response_v1/author_response_observability_r1",
    )
    args = parser.parse_args()
    source, out = args.source.resolve(), args.source.resolve() / "units_companion"
    H.verify_hashes(source, H.load(source / "output_seal.json"))
    if args.phase == "analyze":
        out.mkdir()
        H.write(
            out / "manifest.json",
            {
                "code": {str(Path(__file__).relative_to(ROOT)): H.sha(Path(__file__))},
                "source_table": H.sha(source / "profile_accounts.csv"),
            },
        )
        H.write(out / "summary.json", compute(source))
        H.write(out / "output_seal.json", H.seals(out, out))
        print("HM109 unit accounts separated; primary retained", flush=True)
    else:
        manifest = H.load(out / "manifest.json")
        H.verify_hashes(ROOT, manifest["code"])
        H.verify_hashes(out, H.load(out / "output_seal.json"))
        if manifest["source_table"] != H.sha(
            source / "profile_accounts.csv"
        ) or compute(source) != H.load(out / "summary.json"):
            raise ValueError("Unit-specific full table replay failed")
        H.write(
            out / "verification.json",
            {"status": "VERIFIED_UNIT_COMPANION_NOT_GLOBAL", "same_agent": True},
        )
        print("HM109 full unit-account table replay PASS", flush=True)


if __name__ == "__main__":
    main()
