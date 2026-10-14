#!/usr/bin/env python3
"""Save fixed-arm mechanism summaries without a winning-head selection."""

from __future__ import annotations

import csv
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_grain_hierarchical_query_hm88 import load, sha, table, verify, write

OUT = ROOT / "results/grain_math_response_v1/channel_boundary_transport_r1"


def records(path):
    """Read immutable primary tables with explicitly named evidence units."""
    with path.open() as stream:
        return list(csv.DictReader(stream))


def run():
    """Integrate the conditional identity over the declared author evidence laws."""
    verify(OUT, load(OUT / "02/output_seal.json"))
    verification = load(OUT / "verification.json")
    if verification["status"] != "VERIFIED_FINITE_HM90_NOT_GLOBAL":
        raise ValueError("Primary numerical verification required")
    manifest, rows = load(OUT / "00/manifest.json"), load(OUT / "00/rows.json")
    contrasts = records(OUT / "02/author_law_contrasts.csv")
    accounts = records(OUT / "02/conditioning_accounts.csv")
    corner_weight = {}
    for row in rows:
        x, y, g, h = row["coordinates"]
        corner_weight[row["unit"]] = manifest["spec"]["epsilon"] * (x * g + y * h) / 16
    summaries = []
    for frame, arm in itertools.product(
        [*manifest["spec"]["frames"], "ALL"], manifest["arms"]
    ):
        selected = [
            r
            for r in contrasts
            if r["is_target"] == "True"
            and r["arm"] == arm
            and (frame == "ALL" or r["frame"] == frame)
        ]
        c = np.array([float(r["author_conditional_contrast"]) for r in selected])
        raw = np.array([float(r["author_raw_contrast"]) for r in selected])
        summaries.append(
            {
                "frame": frame,
                "arm": arm,
                "conditions": len(c),
                "positive": int((c > 0).sum()),
                "conditional_mean": float(c.mean()),
                "raw_mean": float(raw.mean()),
            }
        )
    decompositions, errors = [], []
    for frame, order, target, readout, kind in itertools.product(
        manifest["spec"]["frames"],
        (0, 1),
        manifest["spec"]["targets"],
        manifest["spec"]["targets"],
        ("direct", "harmonic"),
    ):
        rs = [
            r
            for r in accounts
            if (r["frame"], int(r["order"]), r["target"], r["readout"], r["kind"])
            == (frame, order, target, readout, kind)
        ]
        if len(rs) != 16:
            raise ValueError("Incomplete author-law cells")
        terms = {
            key: sum(corner_weight[int(r["unit"])] * float(r[key]) for r in rs)
            for key in ("difference", "numerator_term", "normalization_term")
        }
        raw_effect = sum(
            corner_weight[int(r["unit"])]
            * (float(r["paired_raw_action"]) - float(r["mean_raw_action"]))
            for r in rs
        )
        error = abs(
            terms["difference"] - terms["numerator_term"] - terms["normalization_term"]
        )
        errors.append(error)
        decompositions.append(
            {
                "frame": frame,
                "order": order,
                "target": target,
                "readout": readout,
                "kind": kind,
                **terms,
                "raw_pairing_contrast": raw_effect,
                "identity_error": error,
                "is_target": target == readout,
            }
        )
    if max(errors) > 1e-10:
        raise ValueError("Author-law integrated identity failed")
    out = OUT / "03_interpretation"
    out.mkdir()
    table(out / "fixed_arm_summary.csv", summaries)
    table(out / "author_pairing_decomposition.csv", decompositions)
    result = {
        "status": "RESULT_AFTER_INTERPRETATION_NOT_NEW_EXPERIMENT",
        "code_sha256": sha(Path(__file__)),
        "new_model_calls": 0,
        "primary_seal": sha(OUT / "02/output_seal.json"),
        "verification_sha256": sha(OUT / "verification.json"),
        "author_integrated_identity_error": max(errors),
        "all_shifts_preserved_no_winning_arm_selection": True,
        "synchronous_permutation_is_coordinate_relabeling_gate_only_is_not": True,
        "semantic_role_uniqueness_not_inferred_from_positive_contrast": True,
        "original_scope_complete": False,
    }
    write(out / "summary.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    run()
