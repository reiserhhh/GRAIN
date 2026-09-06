#!/usr/bin/env python3
"""Validate public atlas provenance links and evidence-status schema, not science."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def validate(root=ROOT):
    """Reject missing reports, ambiguous IDs and accidental completion promotion."""
    directory = root / "reports/grain_vector_mechanisms"
    data = json.loads((directory / "mechanism_evidence.json").read_text())
    statuses = {"observed", "unsupported", "underresolved", "not_tested", "refused"}
    keys = {
        "instrument",
        "function",
        "semantic_code_fidelity",
        "psychological_validity",
        "external_association",
        "author_correspondence",
        "full_neural_comparison",
    }
    ids = set()
    for entry in data["entries"]:
        assert entry["id"] not in ids
        ids.add(entry["id"])
        path = (directory / entry["source_report"]).resolve()
        assert path.is_relative_to(directory.resolve()) and path.is_file()
        assert (
            "psychological_validity" in entry
            and "instrument" in entry
            and entry["not_claimed"]
        )
        assert all(entry[k] in statuses for k in keys & entry.keys())
    if data["remaining"]:
        assert (
            data["program_complete"] is False and data["status"] == "WORKING_SYNTHESIS"
        )
    assert (directory / "MECHANISM_ATLAS.md").exists()
    assert (root / "docs/GRAIN_VECTOR_OBSERVER_SPEC.md").exists()
    return {
        "status": "PASS_PUBLIC_SCHEMA_AND_LINKS_ONLY",
        "entries": len(ids),
        "program_complete": data["program_complete"],
        "scientific_validity_independently_proven": False,
    }


if __name__ == "__main__":
    print(json.dumps(validate()))
