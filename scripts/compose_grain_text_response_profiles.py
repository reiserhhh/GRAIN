#!/usr/bin/env python3
"""Compose fixed same-marginal evidence profiles from actual HM104 text corners."""

from __future__ import annotations

import csv
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction.math_author_evidence_merge import compose
from scripts import run_grain_author_evidence_merge_hm102 as H
from scripts.run_grain_text_response_hm104 import check, group_data


def run(out):
    """Retain every signed target contrast; never fit or select profile weights."""
    manifest = check(out)
    if (
        H.load(out / "verification.json")["status"]
        != "VERIFIED_FINITE_HM104_NOT_GLOBAL"
    ):
        raise ValueError("Primary finite verification required")
    H.verify_hashes(out, H.load(out / "output_seal.json"))
    dest = out / "03_author_profiles"
    dest.mkdir()
    signed = np.array([1, -1, -1, 1], float)
    pplus, pminus = (1 + 0.5 * signed) / 4, (1 - 0.5 * signed) / 4
    H.write(
        dest / "manifest.json",
        {
            "profile_order": ["high_A", "low_A", "high_B", "low_B"],
            "positive_law": pplus.tolist(),
            "negative_law": pminus.tolist(),
            "code_sha256": H.sha(Path(__file__)),
            "theory_sha256": H.sha(
                ROOT / "docs/GRAIN_TEXT_QUALIFIED_RESPONSE_THEORY_V1.md"
            ),
            "primary_manifest_sha256": H.sha(out / "00/manifest.json"),
            "primary_verification_sha256": H.sha(out / "verification.json"),
            "counts_are_design_profiles_not_human_authors": True,
        },
    )
    targets = manifest["spec"]["targets"]
    all_rows, reader = H.load(out / "00/rows.json"), H.load(out / "00/reader.json")
    with (out / "02/actual_text_interactions.csv").open() as stream:
        actual_table = {
            (
                int(r["bank"]),
                int(r["bridge"]),
                r["current_family"],
                r["content_axis"],
                r["target"],
            ): float(r["actual_mixed"])
            for r in csv.DictReader(stream)
        }
    rows, error, marginal_error = [], 0.0, 0.0
    for p in (pplus, pminus):
        marginal_error = max(
            marginal_error, abs(p[[0, 2]].sum() - 0.5), abs(p[[0, 1]].sum() - 0.5)
        )
    for bank, bridge, family in itertools.product(range(2), range(2), range(6)):
        selected = [
            r
            for r in all_rows
            if (r["bank"], r["bridge"], r["family"]) == (bank, bridge, family)
        ]
        _, _, scores, _, _, _, _ = group_data(out, selected, reader)
        for axis in range(6):
            f = np.stack(
                [
                    scores[axis, 2, 2],
                    scores[axis, 0, 2],
                    scores[axis, 2, 0],
                    scores[axis, 0, 0],
                ]
            )
            plus = compose(f, np.zeros_like(f), pplus)[0]
            minus = compose(f, np.zeros_like(f), pminus)[0]
            alt = (pplus - pminus) @ f
            for q, target in enumerate(targets):
                primary = (
                    0.25
                    * actual_table[bank, bridge, targets[family], targets[axis], target]
                )
                contrast = float(plus[q] - minus[q])
                error = max(error, abs(contrast - primary), abs(contrast - alt[q]))
                rows.append(
                    {
                        "bank": bank,
                        "bridge": bridge,
                        "current_family": targets[family],
                        "content_axis": targets[axis],
                        "target": target,
                        "positive_profile": float(plus[q]),
                        "negative_profile": float(minus[q]),
                        "contrast": contrast,
                    }
                )
    H.table(dest / "author_profile_contrasts.csv", rows)
    summary = {
        "status": "COMPOSED_AND_REPLAYED_FINITE",
        "paired_design_profiles": 144,
        "contrast_rows": len(rows),
        "replay_error_max": error,
        "marginal_error_max": marginal_error,
        "new_model_calls": 0,
        "new_fits": 0,
    }
    for bank in range(2):
        a = np.array([r["contrast"] for r in rows if r["bank"] == bank])
        summary[f"bank{bank}_contrast_rms"] = float(np.sqrt(np.mean(a**2)))
    if max(error, marginal_error) > manifest["spec"]["numeric_atol"]:
        H.write(dest / "failed_summary.json", summary)
        raise ValueError("Author expectation bridge failed")
    H.write(dest / "summary.json", summary)
    H.write(out / "author_profile_seal.json", H.seals(out, dest))
    print("COMPOSED HM104", summary, flush=True)


if __name__ == "__main__":
    run(ROOT / "results/grain_math_response_v1/text_supported_response_r1")
