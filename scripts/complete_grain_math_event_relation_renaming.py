#!/usr/bin/env python3
"""Transparent HM58 serialization completion and narrow verification amendment.

Frozen scientific code, CSV tables, inputs and raw observations are unchanged.
The legacy verifier's alias check is amended from the first record to the
explicit parent_id, preserving every alternate numerical replay procedure.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_grain_math_event_relation_renaming import check

OUT = ROOT / "results/grain_math_response_v1/event_relation_renaming_r1"
VERIFIER = ROOT / "scripts/verify_grain_math_event_relation_renaming.py"


def read(path):
    """Read a saved artifact, never infer or regenerate a missing observation."""
    return json.loads(Path(path).read_text())


def sha(path):
    """Bind code/results to the separately recorded completion procedure."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    """Write generated metadata using only plain JSON number types."""
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def amend_alias_parent(tree):
    """Replace exactly one legacy first-record lookup with the explicit parent."""
    original = ast.dump(ast.parse('own[0]["event_id"]', mode="eval").body)
    replacement = 'next(r for r in own if r["record_id"] == x["parent_id"])["event_id"]'

    class Lookup(ast.NodeTransformer):
        """Change the known metadata-check expression and nothing numerical."""

        def __init__(self):
            self.count = 0

        def visit_Subscript(self, node):
            if ast.dump(node) == original:
                self.count += 1
                return ast.copy_location(ast.parse(replacement, mode="eval").body, node)
            return self.generic_visit(node)

    transform = Lookup()
    repaired = transform.visit(tree)
    if transform.count != 1:
        raise ValueError(
            "Only the exact single frozen alias-parent expression may change"
        )
    return ast.fix_missing_locations(repaired)


def complete(out):
    """Complete the already written CSV summary after int64 JSON serialization failed."""
    check(out)
    state = read(out / "run_status.json")
    if (
        state["status"] != "COLLECTED"
        or state["completed_units"] != 648
        or state["in_flight"] is not None
    ):
        raise ValueError("Complete, terminal acquisition required")
    if (out / "03/summary.json").exists() or (out / "03/artifact_hashes.json").exists():
        raise FileExistsError("Do not overwrite an existing scientific summary")
    names = (
        "native",
        "pool_coordinates",
        "paired_changes",
        "orbits",
        "collisions",
        "orbit_summary",
        "collision_summary",
    )
    counts, total = {}, []
    for name in names:
        with (out / f"03/{name}.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        counts[name] = len(rows)
        if name == "orbit_summary":
            for row in rows:
                if row["content"] != "ALL":
                    continue
                total.append(
                    {
                        k: None
                        if value == ""
                        else value
                        if k in ("content", "contrast")
                        else int(value)
                        if k in ("pairs", "all_codes_same_nonzero_sign")
                        else float(value)
                        for k, value in row.items()
                    }
                )
        if name == "orbits":
            maximum = max(float(r["identity_error"]) for r in rows)
    if list(counts.values()) != [864, 30720, 960, 240, 4800, 49, 56] or len(total) != 7:
        raise ValueError("No missing tables may be filled in")
    summary = {
        "packet": "HM58",
        "status": "ANALYZED_FINITE_NOT_GLOBAL",
        "counts": counts,
        "actual_new_calls": state["forward_calls"],
        "reused_calls": state["reused_units"],
        "model_seconds": state["model_seconds"],
        "maximum_orbit_identity_error": maximum,
        "results": total,
        "no_fit_or_patch": True,
        "is_personality_accuracy": False,
        "group_average_invariance_is_tautological": True,
        "group_is_C4_not_full_S4": True,
        "original_scope_complete": False,
    }
    write(out / "03/summary.json", summary)
    write(
        out / "03/artifact_hashes.json",
        {
            str(p.relative_to(out)): sha(p)
            for p in sorted((out / "03").iterdir())
            if p.name != "artifact_hashes.json" and not p.name.startswith("._")
        },
    )
    write(
        out / "completion_manifest.json",
        {
            "status": "SUMMARY_SERIALIZATION_COMPLETED_NOT_YET_VERIFIED",
            "analysis_failure": "TypeError:Object of type int64 is not JSON serializable at results/all_codes_same_nonzero_sign",
            "frozen_scientific_files_changed": False,
            "CSV_values_changed": False,
            "new_model_calls": 0,
            "completion_code_sha256": sha(Path(__file__)),
            "summary_source": "existing full orbit_summary.csv and existing orbit identities",
            "summary_sha256": sha(out / "03/summary.json"),
            "verifier_amendment": "own[0].event_id -> record explicitly addressed by alias parent_id;one AST expression only",
            "frozen_verifier_sha256": sha(VERIFIER),
        },
    )
    print(summary)


def verify(out):
    """Run the frozen alternate replay with only the recorded metadata-check repair."""
    check(out)
    amendment = read(out / "completion_manifest.json")
    if (
        sha(Path(__file__)) != amendment["completion_code_sha256"]
        or sha(VERIFIER) != amendment["frozen_verifier_sha256"]
    ):
        raise ValueError("Completion/amendment code changed")
    tree = amend_alias_parent(ast.parse(VERIFIER.read_text(), filename=str(VERIFIER)))
    scope = {
        "__name__": "grain_hm58_verification_with_declared_parent_amendment",
        "__file__": str(VERIFIER),
    }
    # Execute only hash-checked local verifier code with one declared AST repair.
    exec(compile(tree, str(VERIFIER), "exec"), scope)  # noqa: S102
    scope["verify"](out)
    result = read(out / "verification.json")
    result["metadata_parent_check_amended"] = True
    result["frozen_verifier_sha256"] = amendment["frozen_verifier_sha256"]
    result["amendment_code_sha256"] = amendment["completion_code_sha256"]
    write(out / "verification.json", result)
    amendment["status"] = "VERIFIED_FINITE_COMPLETION_WITH_EXPLICIT_PARENT_CHECK"
    amendment["verification_sha256"] = sha(out / "verification.json")
    write(out / "completion_manifest.json", amendment)


def main():
    """Expose CPU-only summary completion and separate verification; never collect."""
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("complete", "verify"))
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    {"complete": complete, "verify": verify}[args.phase](args.output)


if __name__ == "__main__":
    main()
