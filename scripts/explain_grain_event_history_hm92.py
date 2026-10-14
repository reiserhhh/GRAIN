#!/usr/bin/env python3
"""Sealed result-after HM92 mechanism summaries; no selection or new calls."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_grain_hierarchical_query_hm88 import load, sha, table, verify, write
from scripts.run_grain_query_evidence_hm86 import seal

OUT = ROOT / "results/grain_math_response_v1/event_history_observation_r1"
TERMS = (
    "selection",
    "subject_gate",
    "value_content",
    "selection_gate",
    "selection_value",
    "gate_value",
    "triple",
)


def rows(name):
    """Read complete frozen tables without filtering signs or readers."""
    with (OUT / "02" / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def run():
    """Integrate signed product accounts under the same fixed author law."""
    manifest = load(OUT / "00/manifest.json")
    verify(ROOT, manifest["code"] | manifest["source"])
    verify(OUT, manifest["inputs"])
    verify(OUT, load(OUT / "01/raw_seal.json"))
    verify(OUT, load(OUT / "02/output_seal.json"))
    if load(OUT / "verification.json")["status"] != "VERIFIED_FINITE_HM92_NOT_GLOBAL":
        raise ValueError("Verified primary required")
    dest = OUT / "03_interpretation"
    if dest.exists():
        raise FileExistsError("Preserve earlier interpretations")
    dest.mkdir()
    material = {r["unit"]: r for r in load(OUT / "00/rows.json")}
    laws = rows("author_laws.csv")
    products = {}
    for r in rows("K2_scope_product_accounts.csv"):
        point = material[int(r["unit"])]["coordinates"]
        weight = manifest["spec"]["epsilon"] * point[1] * point[3] / 32
        key = r["frame"], r["order"], r["family"], r["readout"]
        if key not in products:
            products[key] = dict.fromkeys((*TERMS, "delta"), 0.0)
        for name in (*TERMS, "delta"):
            products[key][name] += weight * float(r[name])
    accounts = []
    for law in laws:
        if law["field"] != "K2_local_paired":
            continue
        key = tuple(law[name] for name in ("frame", "order", "family", "readout"))
        terms = products[key]
        accounts.append(
            {
                **dict(zip(("frame", "order", "family", "readout"), key, strict=True)),
                **terms,
                "author_contrast": float(law["contrast"]),
                "identity_error": abs(
                    sum(terms[n] for n in TERMS) - float(law["contrast"])
                ),
            }
        )
    table(dest / "K2_author_product_accounts.csv", accounts)
    matched = [r for r in laws if r["family"] == r["readout"]]
    table(dest / "matched_author_contrasts.csv", matched)
    controls = rows("past_future_controls.csv")
    past = [r for r in controls if r["changed_other_scope_is_future"] == "False"]
    future = [r for r in controls if r["changed_other_scope_is_future"] == "True"]
    composition = rows("composition.csv")
    ranges = {
        field: [
            min(
                float(r["cross_event_fraction"])
                for r in composition
                if r["field"] == field
            ),
            max(
                float(r["cross_event_fraction"])
                for r in composition
                if r["field"] == field
            ),
        ]
        for field in manifest["fields"]
    }
    contrasts = {}
    for field in manifest["fields"]:
        for order in ("0", "1"):
            values = [
                float(r["contrast"])
                for r in matched
                if r["field"] == field and r["order"] == order
            ]
            contrasts[field + ":order" + order] = {
                "absolute_range": [
                    min(abs(v) for v in values),
                    max(abs(v) for v in values),
                ],
                "rms": float(np.sqrt(np.mean(np.asarray(values) ** 2))),
                "signed_values": values,
            }
    reader_fractions = []
    for r in rows("reader_accounts.csv"):
        if r["order"] == "0" and r["field"] in (
            "full_paired_raw",
            "K2_local_paired",
            "K2_uniform_value",
        ):
            reader_fractions.append(
                {
                    "frame": r["frame"],
                    "field": r["field"],
                    "reader_specific_energy_fraction": float(
                        r["reader_specific_energy"]
                    )
                    / float(r["full_energy"]),
                }
            )
    write(
        dest / "summary.json",
        {
            "status": "RESULT_AFTER_INTERPRETATION_NOT_NEW_REPLICATION",
            "code_sha256": sha(Path(__file__)),
            "new_model_calls": 0,
            "past_condition_count": len(past),
            "future_condition_count": len(future),
            "past_raw_value_rms_range": [
                min(float(r["value_rms"]) for r in past),
                max(float(r["value_rms"]) for r in past),
            ],
            "past_paired_rms_range": [
                min(float(r["local_paired_rms"]) for r in past),
                max(float(r["local_paired_rms"]) for r in past),
            ],
            "future_raw_value_max": max(float(r["value_max_abs"]) for r in future),
            "future_local_paired_rms_max": max(
                float(r["local_paired_rms"]) for r in future
            ),
            "cross_event_fraction_ranges": ranges,
            "matched_contrasts": contrasts,
            "reader_specific_fractions": reader_fractions,
            "integrated_product_max_error": max(a["identity_error"] for a in accounts),
            "scope": "Exact finite observer-history mechanism;not author trait dynamics or personality accuracy",
            "original_scope_complete": False,
        },
    )
    write(dest / "output_seal.json", seal(OUT, dest))
    print("EXPLAINED HM92", len(accounts), "signed author accounts", flush=True)


if __name__ == "__main__":
    run()
