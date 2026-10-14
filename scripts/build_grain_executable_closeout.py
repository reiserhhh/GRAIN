#!/usr/bin/env python3
"""Validate the bounded follow-up outputs and save a non-release manifest."""

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/grain_vector_mechanisms/all_plan_logic_v1"


def digest(path):
    """Stream a local file into a SHA256 digest without exporting its contents."""
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    summaries = [
        "open_condition_r4_corrected_control/summary.json",
        "open_condition_r6_source_mixture_finite/summary.json",
        "open_condition_r7_full_moment/summary.json",
        "metadata_conditions_r2_with_revision_inventory/summary.json",
        "reader_scope_closure_r2_final/summary.json",
    ]
    data = {}
    for name in summaries:
        data[name] = json.loads(
            (BASE / name).read_text(),
            parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)),
        )
    reader = data[summaries[-1]]
    assert reader["contrast_pass"] == reader["contrast_total"] == 12
    assert reader["natural_self_scored"]["guarded"] == 73
    metadata = data[summaries[-2]]
    assert metadata["author_overlap"] == metadata["parent_overlap"] == 0
    tests = ET.parse(BASE / "codex_closure_tests.xml").getroot()
    cases = tests.findall(".//testcase")
    assert len(cases) >= 66
    assert not tests.findall(".//failure") and not tests.findall(".//error")
    files = [BASE / p for p in summaries] + [BASE / "codex_closure_tests.xml"]
    files += [
        ROOT / p
        for p in (
            "grain_research/open_condition.py",
            "grain_research/dialogue_ledger_generic.py",
            "scripts/run_grain_open_condition_assignment.py",
            "scripts/run_grain_metadata_conditions.py",
            "scripts/run_grain_reader_scope_closure.py",
            "scripts/build_grain_executable_closeout.py",
            "tests/test_grain_all_plan_closure.py",
            "docs/GRAIN_EXECUTABLE_RESEARCH_PLAN_V2.md",
            "docs/GRAIN_ALL_PLAN_REQUIREMENT_CROSSWALK_V1.md",
            "docs/GRAIN_CURRENT_SYSTEM_LOGIC_V1.md",
            "reports/grain_measurement_foundation/CODEX_EXECUTABLE_GAP_CLOSEOUT_20260914.md",
            "docs/paper/GRAIN_MECHANISM_THEORY_PACKAGE_V1.md",
            "docs/paper/GRAIN_MECHANISM_SUPPLEMENT_EN_V1.md",
            "docs/paper/GRAIN_MECHANISM_SUPPLEMENT_JA_V1.md",
            "docs/paper/GRAIN_MECHANISM_SUPPLEMENT_ZH_V1.md",
        )
    ]
    prefix = "results/grain_vector_mechanisms/all_plan_logic_v1"
    manifest = {
        "status": "bounded_V2_followups_executed_not_universal_theory_closure",
        "tests_passed": len(cases),
        "files_sha256": {str(p.relative_to(ROOT)): digest(p) for p in files},
        "input_sha256": {
            "private_antecedent_pairs": digest(
                BASE / "natural_antecedent_scan_r1/private_antecedent_pairs.csv"
            )
        },
        "private_data": "Input and parse caches stay gitignored and local; manifest hashes do not authorize publication.",
        "commands": [
            f"python scripts/run_grain_open_condition_assignment.py --output {prefix}/REPRO_control --clause-correlated",
            f"python scripts/run_grain_open_condition_assignment.py --output {prefix}/REPRO_epsilon --clause-correlated --source-error-mixture",
            f"python scripts/run_grain_open_condition_assignment.py --output {prefix}/REPRO_moment --clause-correlated --full-assignment-moment",
            f"python scripts/run_grain_metadata_conditions.py --pairs {prefix}/natural_antecedent_scan_r1/private_antecedent_pairs.csv --output {prefix}/REPRO_metadata",
            f"python scripts/run_grain_reader_scope_closure.py --input-dir {prefix}/natural_antecedent_coverage_r3_polar400 --output {prefix}/REPRO_scope",
            "python -m pytest tests/test_grain_all_plan_closure.py tests/test_grain_open_condition.py tests/test_grain_dialogue_ledger_generic.py tests/test_grain_proposition_response.py tests/test_grain_dialogue_claim_ledger.py tests/test_grain_dialogue_ledger_input.py tests/test_grain_contextual_instrument.py tests/test_grain_pilot_transport.py -q",
        ],
        "python": "/Volumes/mobile3/projects/project persona/.venv/bin/python",
        "working_directory": str(ROOT),
        "notes": "Use fresh output directories; no old output is overwritten. Hashes capture closeout code, not a preregistration.",
    }
    (BASE / "codex_closeout_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "tests_passed": len(cases),
                "hashed_files": len(files),
            }
        )
    )


if __name__ == "__main__":
    main()
