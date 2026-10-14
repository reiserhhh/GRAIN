#!/usr/bin/env python3
"""After-result, zero-forward diagnostics of every qualified composition channel."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_qualified_composition as M
from scripts import run_grain_author_evidence_merge_hm102 as H
from scripts import run_grain_qualified_composition_hm110 as P


def cosine(a, b):
    """Avoid naming near-zero additive-law numerical residue a transport pattern."""
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.sum(a * b) / denominator) if denominator > 1e-18 else None


def main():
    """Save all fixed channels/targets; no choice, refit, model call or promotion."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    out = args.source
    P.check(out)
    H.verify_hashes(out, H.load(out / "output_seal.json"))
    if (
        H.load(out / "verification.json")["status"]
        != "VERIFIED_FINITE_HM110_NOT_GLOBAL"
    ):
        raise ValueError("Verified finite packet required")
    contrasts = []
    for bank in range(2):
        with np.load(out / f"03/cold_output_{bank}.npz", allow_pickle=False) as z:
            contrasts.append(z["scores"][:, 0] - z["scores"][:, 1])
    contrasts = np.array(contrasts)
    targets = H.load(out / "00/reader.json")["targets"]
    records = []
    for bank in range(2):
        for target, name in enumerate(targets):
            records.append(
                {
                    "bank": bank,
                    "target": name,
                    **{
                        f + "_RMS": M.rms(contrasts[bank, :, field, target])
                        for field, f in enumerate(P.FIELDS)
                    },
                    "encoding_remainder_RMS": M.rms(
                        contrasts[bank, :, 0, target] - contrasts[bank, :, 2, target]
                    ),
                }
            )
    transport = []
    for field, f in enumerate(P.FIELDS[:3]):
        a, b = contrasts[:, :, field]
        transport.append(
            {
                "field": f,
                "cosine": cosine(a, b),
                "same_sign": int(np.sum(a * b > 0)),
                "cells": int(a.size),
            }
        )
    dest = out / "diagnostics_companion"
    dest.mkdir()
    H.table(dest / "author_targets.csv", records)
    H.write(
        dest / "summary.json",
        {
            "classification": "RESULT_AFTER_DIAGNOSTIC_NOT_REGISTERED_REPLICATION",
            "source_application_author_patterns": transport,
            "bank_differences": "Behavior situations and wording both change;not a paraphrase-only test",
            "nonzero_actual_cells_over_1e_minus9": (abs(contrasts[:, :, 0]) > 1e-9)
            .sum(axis=1)
            .tolist(),
            "new_forwards": 0,
            "new_fits": 0,
        },
    )
    H.write(
        dest / "manifest.json",
        {
            "code": {str(Path(__file__).relative_to(ROOT)): H.sha(Path(__file__))},
            "primary_output_seal": H.sha(out / "output_seal.json"),
            "verification": H.sha(out / "verification.json"),
            "outputs": H.seals(out, dest),
        },
    )
    print("HM110 diagnostics saved: all targets/channels,not new prediction accuracy")


if __name__ == "__main__":
    main()
