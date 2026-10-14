#!/usr/bin/env python3
"""Result-after signed author accounts; preserve HM91's primary seal unchanged."""

from __future__ import annotations

import csv
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_grain_hierarchical_query_hm88 import load, sha, table, verify, write

OUT = ROOT / "results/grain_math_response_v1/relation_scope_r1"
TERMS = (
    "selection",
    "subject_gate",
    "value_content",
    "selection_gate",
    "selection_value",
    "gate_value",
    "triple",
)


def read_table(name):
    """Read immutable public aggregate accounts, not private natural text."""
    with (OUT / "02" / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def run():
    """Integrate the exact seven-term identity under the fixed relation laws."""
    verified = load(OUT / "verification.json")
    if verified["status"] != "VERIFIED_FINITE_HM91_NOT_GLOBAL":
        raise ValueError("Verified primary required")
    verify(OUT, load(OUT / "02/output_seal.json"))
    manifest, material = load(OUT / "00/manifest.json"), load(OUT / "00/rows.json")
    frames, targets = manifest["spec"]["frames"], manifest["spec"]["targets"]
    lookup = {
        (r["frame"], r["mode"], r["target"], tuple(r["coordinates"])): r["unit"]
        for r in material
    }
    account = {
        (int(r["unit"]), r["readout"]): r
        for r in read_table("core_product_accounts.csv")
    }
    authors = {
        (r["frame"], r["target"], r["field"], r["readout"]): float(
            r["joint_law_contrast"]
        )
        for r in read_table("author_laws.csv")
    }
    records, error = [], 0.0
    for frame, target, readout in itertools.product(frames, targets, targets):
        integrated = {term: 0.0 for term in TERMS}
        for point in itertools.product((-1, 1), repeat=4):
            inner = account[lookup[frame, "negated_event", target, point], readout]
            outer = account[lookup[frame, "negated_report", target, point], readout]
            x, y, g, h = point
            weight = manifest["spec"]["epsilon"] * (x * g + y * h) / 96
            for term in TERMS:
                integrated[term] += weight * (float(inner[term]) - float(outer[term]))
        core = authors[frame, target, "paired_core", readout]
        full = authors[frame, target, "paired_raw", readout]
        identity_error = abs(core - sum(integrated.values()))
        error = max(error, identity_error)
        records.append(
            {
                "frame": frame,
                "target": target,
                "readout": readout,
                "full_contrast": full,
                "matched_behavior_contrast": core,
                "remaining_token_contrast": full - core,
                **integrated,
                "integrated_identity_error": identity_error,
            }
        )
    geometry = read_table("scalar_action.csv")
    raw = [
        float(r["non_scalar_energy_fraction"])
        for r in geometry
        if r["mode"] != "asserted_event" and r["field"] == "paired_raw"
    ]
    core_geometry = [
        float(r["non_scalar_energy_fraction"])
        for r in geometry
        if r["mode"] != "asserted_event" and r["field"] == "paired_core"
    ]
    matched = [r for r in records if r["target"] == r["readout"]]
    fulls, cores = (
        np.array([r["full_contrast"] for r in matched]),
        np.array([r["matched_behavior_contrast"] for r in matched]),
    )
    dest = OUT / "03_interpretation"
    dest.mkdir()
    table(dest / "author_relation_accounts.csv", records)
    write(
        dest / "summary.json",
        {
            "result_after_not_preregistered_replication": True,
            "code_sha256": sha(Path(__file__)),
            "primary_output_seal_sha256": sha(OUT / "02/output_seal.json"),
            "verification_sha256": sha(OUT / "verification.json"),
            "non_scalar_raw_range": [min(raw), max(raw)],
            "non_scalar_behavior_range": [min(core_geometry), max(core_geometry)],
            "matched_full_rms": float(np.sqrt(np.mean(fulls**2))),
            "matched_behavior_rms": float(np.sqrt(np.mean(cores**2))),
            "matched_remaining_rms": float(np.sqrt(np.mean((fulls - cores) ** 2))),
            "full_core_opposite_sign_conditions": int(
                np.count_nonzero(fulls * cores < 0)
            ),
            "matched_full_absolute_range": [
                float(np.min(np.abs(fulls))),
                float(np.max(np.abs(fulls))),
            ],
            "integrated_identity_error": error,
            "matched_accounts": matched,
            "no_new_model_calls": True,
            "scientific_scope": "A signed algebraic account,not a causal activation patch or psychological information percentage",
        },
    )
    if error > 1e-10:
        raise ValueError("Preserve failed companion; no changed tolerance")
    print(
        "HM91 INTERPRETED",
        "identity",
        error,
        "RMS full/core/rest",
        float(np.sqrt(np.mean(fulls**2))),
        float(np.sqrt(np.mean(cores**2))),
        float(np.sqrt(np.mean((fulls - cores) ** 2))),
        flush=True,
    )


if __name__ == "__main__":
    run()
