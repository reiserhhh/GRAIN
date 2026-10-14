#!/usr/bin/env python3
"""Separate condition/address audit; never alter HM243's sealed observations."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_observed_family_hm170 as R

PRIMARY = ROOT / "results/grain_math_response_v1/companion_observation_r1"
PARENT = ROOT / "results/grain_math_response_v1/relational_diagnostic_update_r1"
OUTPUT = ROOT / "results/grain_math_response_v1/companion_metadata_audit_r1"


def main():
    """Bind inherited condition metadata to SAME prepared evidence addresses."""
    R.check(PRIMARY)
    R.check(PARENT, ("built_seal.json", "analyzed_seal.json", "verified_seal.json"))
    with np.load(PARENT / "observations.npz", allow_pickle=False) as z:
        old = {
            k: z[k]
            for k in (
                "author_ids",
                "selected",
                "control",
                "panel_conditions",
                "panel_vectors",
            )
        }
    with np.load(PRIMARY / "observations.npz", allow_pickle=False) as z:
        new = {
            k: z[k]
            for k in (
                "author_ids",
                "selected",
                "control",
                "panel_vectors",
                "permutation",
            )
        }
    if any(
        not np.array_equal(old[k], new[k])
        for k in ("author_ids", "selected", "control", "panel_vectors")
    ):
        raise ValueError(
            "Condition metadata does not share the native vector addresses"
        )
    c = old["panel_conditions"]
    rows = []
    for u, v in np.ndindex(24, 2):
        other = int(new["permutation"][u])
        selected, control = old["selected"][u, v], old["control"][u, v]
        ns, nc = int(c[u, v, selected].sum()), int(c[u, v, control].sum())
        own, foreign = int(c[u, 1 - v].sum()), int(c[other, 1 - v].sum())
        rows.append(
            {
                "author": u,
                "view": v,
                "selected_condition1": ns,
                "control_condition1": nc,
                "focal_edit_condition_count_equal": ns == nc,
                "own_companion_condition1": own,
                "foreign_companion_condition1": foreign,
                "companion_condition_count_equal": own == foreign,
                "foreign_author": other,
                "same_event_vector_addresses_verified": True,
            }
        )
    OUTPUT.mkdir(parents=True, exist_ok=False)
    R.write(
        OUTPUT / "manifest.json",
        {
            "code": {str(Path(__file__).relative_to(ROOT)): R.digest(Path(__file__))},
            "inputs": {
                str(p.relative_to(ROOT)): R.digest(p)
                for p in (
                    PRIMARY / "observations.npz",
                    PRIMARY / "event_contracts.json",
                    PARENT / "observations.npz",
                    PARENT / "verified_seal.json",
                )
            },
            "separate_from_sealed_primary": True,
            "native_outcomes_read": False,
            "new_neural_forwards": 0,
        },
    )
    R.table(OUTPUT / "condition_accounts.csv", rows)
    summary = {
        "views": 48,
        "authors": 24,
        "focal_edit_count_equal": sum(
            r["focal_edit_condition_count_equal"] for r in rows
        ),
        "companion_count_equal": sum(
            r["companion_condition_count_equal"] for r in rows
        ),
        "both_equal": sum(
            r["focal_edit_condition_count_equal"]
            and r["companion_condition_count_equal"]
            for r in rows
        ),
        "condition_is_inherited_top20_venue_bit_not_full_situation": True,
        "equal_four_event_count_not_full_condition_matching": True,
        "no_selection_rescue": True,
        "primary_formula_or_outputs_changed": False,
        "original_scope_complete": False,
    }
    R.write(OUTPUT / "summary.json", summary)
    R.seal(
        OUTPUT,
        "audit_seal.json",
        ("manifest.json", "condition_accounts.csv", "summary.json"),
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
