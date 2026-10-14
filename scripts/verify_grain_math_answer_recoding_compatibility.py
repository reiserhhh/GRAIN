#!/usr/bin/env python3
"""Preserve the frozen verifier failure; resolve HM50's old capture metadata schema."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts import verify_grain_math_answer_recoding_mechanism as V
from scripts.run_grain_math_answer_recoding_mechanism import OUT, write


def verify(out):
    """Bind missing source identity fields to HM50's frozen design, not HM53 scores."""
    manifest = V.check(out)
    hm52 = ROOT / manifest["spec"]["parent"]
    hm51 = ROOT / read_json(hm52 / "00/manifest.json")["spec"]["parent"]
    hm50 = ROOT / read_json(hm51 / "00/manifest.json")["spec"]["parent"]
    original = {
        hm50 / f"01/units/unit_{e['unit']:04d}.json": e
        for e in read_json(hm50 / "00/design.json")
        if e["kind"] == "capture" and e["role"] == "source"
    }
    old_rows = {r["unit"]: r for r in read_json(hm50 / "00/receivers.json")}
    for r in read_json(out / "00/rows.json"):
        if r["role"] == "source" and r["reused"]:
            entry = original[ROOT / r["reused_path"]]
            old = old_rows[entry["receiver"]]
            if (
                r["input_ids"] != old["input_ids"]
                or r["rendered"] != old["rendered"]
                or any(r[k] != entry[k] for k in ("content", "person", "direction"))
            ):
                raise ValueError("Historical source binding mismatch")
    actual_loader = V.load_arrays

    def schema_loader(path):
        """Add only declared absent identity fields in memory; do not modify old files."""
        meta, arrays = actual_loader(path)
        if path in original:
            entry = original[path]
            if any(
                meta[k] != entry[k]
                for k in ("unit", "receiver", "donor", "role", "kind", "arm")
            ):
                raise ValueError(
                    "Original source capture no longer matches original design"
                )
            for k in ("content", "person", "direction"):
                if k in meta and meta[k] != entry[k]:
                    raise ValueError("Cannot override an existing scientific identity")
            meta = meta | {k: entry[k] for k in ("content", "person", "direction")}
        return meta, arrays

    try:
        V.load_arrays = schema_loader
        V.verify(out)
    finally:
        V.load_arrays = actual_loader
    result = read_json(out / "verification.json")
    result.update(
        compatibility_companion_sha256=digest(Path(__file__)),
        original_frozen_verifier_error="KeyError: content in unannotated HM50 capture metadata; retained unchanged",
        identity_source="HM50 frozen design/receiver/input binding, not current application answers",
        amended_scientific_arrays_or_scores=False,
    )
    write(out / "verification.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    verify(parser.parse_args().output)
