#!/usr/bin/env python3
"""HM77 result-after display identity repair; never change frozen model/results."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grain_research.construction.math_heterogeneous_evidence_cube import TARGETS
from scripts.run_grain_context_anchored_operation import (
    OUT,
    check,
    digest,
    files,
    models,
    read,
    seal,
    write,
)


def main():
    """Restore overwritten display labels using the sealed source loop order."""
    check()
    fitted = models()
    if set(fitted) != {(q, m) for q in TARGETS for m in ("orthogonal", "oblique")}:
        raise ValueError("Frozen source model filename identities differ")
    original = OUT / "01/diagnostics.json"
    diagnostics = read(original)
    if len(diagnostics) != len(TARGETS):
        raise ValueError("Source diagnostics do not match fixed source loop order")
    out = OUT / "04_display_identity"
    if out.exists():
        raise FileExistsError("Identity companion already exists; no overwrite")
    out.mkdir()
    repaired = []
    for query, row in zip(TARGETS, diagnostics, strict=True):
        if not isinstance(row["query"], dict):
            raise TypeError("Unexpected original query display schema")
        model = fitted[query, "oblique"]
        if row["anchor_rank"] != model["anchor_left"].shape[1]:
            raise ValueError("Source diagnostic/model dimension mismatch")
        repaired.append(
            {
                "query": query,
                "query_geometry": row["query"],
                **{k: v for k, v in row.items() if k != "query"},
            }
        )
    write(out / "source_diagnostics_labeled.json", repaired)
    write(
        out / "manifest.json",
        {
            "status": "RESULT_AFTER_DISPLAY_IDENTITY_ONLY",
            "original_diagnostics_sha256": digest(original),
            "source_plan_sha256": digest(
                OUT / "00/code/scripts/run_grain_context_anchored_operation.py"
            ),
            "origin": "calibrate loop saves in TARGETS order; numeric info.query overwrote its display label",
            "label_source": "sealed loop order AND exact frozen model filename set",
            "all_original_numeric_values_preserved": all(
                row["query"] == companion["query_geometry"]
                for row, companion in zip(diagnostics, repaired, strict=True)
            ),
            "new_fits": 0,
            "new_model_calls": 0,
            "primary_changed": False,
        },
    )
    if not read(out / "manifest.json")["all_original_numeric_values_preserved"]:
        raise ValueError("Display companion changed numeric geometry")
    write(out / "output_seal.json", seal(OUT, files(out)))
    print("HM77 display-label companion exported; no scientific result/model changed")


if __name__ == "__main__":
    main()
