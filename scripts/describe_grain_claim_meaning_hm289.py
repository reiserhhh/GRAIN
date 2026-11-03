#!/usr/bin/env python3
"""Post-outcome descriptive accounts only;no additional rank,fit or native calls."""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_observed_family_hm170 as R
from scripts.score_grain_author_belief_cold import load_bundle

PARENT = ROOT / "results/grain_math_response_v1/claim_meaning_r1"
OUT = PARENT / "postoutcome_description_r1"
TARGETS = ("E", "N", "A", "C", "O")


def main():
    """Describe complete fixed contacts under both new claims and the old frame."""
    start = time.process_time()
    R.check(
        PARENT, ("collected_seal.json", "analyzed_seal.json", "verification_seal.json")
    )
    if OUT.exists():
        raise FileExistsError("No repeated or outcome-selected descriptor")
    data, summary = (
        load_bundle(PARENT / "readings.npz"),
        R.read(PARENT / "summary.json"),
    )
    bank = R.read(PARENT / "bank.json")
    rows, account = [], {}
    for sp, space in enumerate(("E5", "GRAIN15")):
        for ref, reference in enumerate(("bank0", "bank1", "pooled")):
            odd = data["semantic_direction/global_common"][sp, ref, 0, 2, 3]
            even = data["affirmation_direction/global_common"][sp, ref, 0, 2, 3]
            shift = data["positive_frame_change_direction/global_common"][
                sp, ref, 0, 2, 3
            ]
            for q, target in enumerate(TARGETS):
                rows.append(
                    {
                        "space": space,
                        "reference": reference,
                        "target": target,
                        "odd": float(odd[q]),
                        "even": float(even[q]),
                        "new_positive": float(odd[q] + even[q]),
                        "new_negative": float(even[q] - odd[q]),
                        "positive_frame_change": float(shift[q]),
                        "old_positive": float(odd[q] + even[q] - shift[q]),
                    }
                )
            for channel in (
                "semantic_direction",
                "affirmation_direction",
                "availability_even",
                "availability_odd",
            ):
                for readout in ("balanced_key", "common_support"):
                    matrix = data[channel + "/" + readout][sp, ref, 0, 2, 3]
                    account[f"{space}/{reference}/{channel}/{readout}"] = (
                        matrix.tolist()
                    )
    # A whole-claim involution does not establish that individual item content matters.
    # This finite no-content counterworld uses no actual outcome or selected coefficient.
    t = np.array([-1.0, 1.0])
    world_plus, world_minus = t, -t
    no_content_odd = (world_plus - world_minus) / 2
    alias = {
        "authors": 2,
        "items_have_identical_responses": True,
        "source_theta": t.tolist(),
        "positive": world_plus.tolist(),
        "negative": world_minus.tolist(),
        "odd_source_contact": float(np.mean(t * no_content_odd)),
        "not_an_explanation_of_actual_Qwen_outputs": True,
        "not_a_new_nonidentifiability_theorem": True,
    }
    OUT.mkdir()
    R.table(OUT / "level_and_frame_accounts.csv", rows)
    R.write(
        OUT / "description.json",
        {
            "postoutcome_not_preregistered_not_independent_replication": True,
            "new_fits_significance_draws_native_calls": 0,
            "all_source_query_contacts": account,
            "no_content_counterworld": alias,
            "primary_results_unchanged": summary["primary"],
            "runtime_CPU_seconds": time.process_time() - start,
            "item_addresses_preserved": [b["item"] for b in bank],
        },
    )
    R.write(
        OUT / "provenance.json",
        {
            "descriptor_code_sha256": R.digest(Path(__file__).resolve()),
            "inputs": {
                p: R.digest(PARENT / p)
                for p in (
                    "readings.npz",
                    "summary.json",
                    "bank.json",
                    "verification_seal.json",
                )
            },
            "parent_scientific_code_or_sealed_data_changed": False,
        },
    )
    R.seal(
        OUT,
        "description_seal.json",
        ("level_and_frame_accounts.csv", "description.json", "provenance.json"),
    )
    R.check(
        PARENT, ("collected_seal.json", "analyzed_seal.json", "verification_seal.json")
    )
    print(
        json.dumps(
            {
                "GRAIN15": [r for r in rows if r["space"] == "GRAIN15"],
                "C_and_A_query_contacts": {
                    k: {"C": v[3], "A": v[2]}
                    for k, v in account.items()
                    if k.startswith("GRAIN15/")
                    and ("semantic_direction/" in k or "affirmation_direction/" in k)
                },
                "no_content_counterworld": alias,
            },
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
