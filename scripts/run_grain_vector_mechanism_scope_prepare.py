#!/usr/bin/env python3
"""Open only first current comments of 48 calibration authors for source-only coding."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.cohort import validate_multicomment_bundle
from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.inventory import archive_source, code_fingerprint
from grain_research.vector_mechanisms.scope import DEFINITIONS, build_scope_controls
from scripts.run_grain_vector_mechanism_controls import write_json


def main():
    """No psychological labels, no holdout author access and no adaptive sampling."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--views", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    bundles = json.loads(a.views.read_text(encoding="utf-8"))
    if len(bundles) != 48 or any(b["split"] != "calibration" for b in bundles):
        raise ValueError(
            "This protocol opens exactly the 48 preselected calibration authors"
        )
    rows = []
    for i, b in enumerate(bundles):
        validate_multicomment_bundle(b, units=12)
        c = b["views"][2][0]
        rows.append(
            {
                "case_id": f"c{i:02d}",
                "author_id": b["author_id"],
                "split": "calibration",
                "current": c["body"],
                "comment_id": c["comment_id"],
                "word_count": c["word_count"],
                "leakage_flags": c["leakage_flags"],
            }
        )
    a.output_dir.mkdir(parents=True, exist_ok=True)
    if any(a.output_dir.iterdir()):
        raise ValueError("Use a new output directory")
    archive_source(ROOT, a.output_dir / "source_snapshot.zip")
    write_json(a.output_dir / "calibration_sources.json", rows)
    write_json(a.output_dir / "synthetic_controls.json", build_scope_controls())
    manifest = {
        "protocol": "GRAIN_VM03_EVIDENCE_SCOPE_V1",
        "source_sha256": material_hash(bundles),
        "calibration_sha256": material_hash(rows),
        "authors": 48,
        "comments": 48,
        "selection": "First comment of third chronological view for every calibration author",
        "definitions": DEFINITIONS,
        "external_labels_used": False,
        "first_calibration_access": "source coding only, before new model readouts",
        "code_sha256": code_fingerprint(ROOT),
    }
    manifest["identity"] = material_hash(manifest)
    write_json(a.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
