#!/usr/bin/env python3
"""Keep public tests runnable without silently rewriting local research tests."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "results/grain_gptpro_foundation_v3"
LOCAL_ARTIFACTS = {
    "test_grain_a02_near_root_evidence.py": "F05/near_root_identification_r1/results.json",
    "test_grain_r01_application_j.py": "P05/application_j_diagnostic_r1/results.json",
    "test_grain_r01_product_space.py": "P05/product_space_identification_r1/results.json",
    "test_grain_r08_conditional_ib.py": "F15/conditional_ib_r1/results.json",
    "test_grain_r08_controlled_psr_witness.py": "F15/controlled_psr_witness_r1/results.json",
    "test_grain_r08_equivalence_hierarchy.py": "F15/equivalence_hierarchy_r1/results.json",
    "test_grain_r08_pandora_history.py": "F14/pandora_history_comparison_r1/results.json",
    "test_grain_r08_pandora_venue_oom.py": "F14/pandora_venue_oom_r1/results.json",
    "test_grain_r08_text_behaviour_quotient.py": "F15/text_behaviour_quotient_r1/results.json",
    "test_grain_r15_hedge_channel_utilization.py": "F17/hedge_channel_utilization_r1/results.json",
    "test_grain_r15_hedge_commitment_reader.py": "F17/hedge_commitment_reader_r1/readings.jsonl",
    "test_grain_r15_speaker_identifiability.py": "F17/speaker_identifiability_r1/results.json",
    "test_grain_r25_wording_facet.py": "F12/designed_wording_facet_r1/results.json",
    "test_grain_r26_committed_design.py": "F05/source_full_r2/core_20260915/source_model.json",
    "test_grain_r26_reactive_head_to_head.py": "F06/reactive_head_to_head_r1/results.json",
    "test_grain_r26_recovery_design.py": "F05/source_full_r2/core_20260915/source_model.json",
    "test_grain_r34_source_halves.py": "F16/source_half_reconstruction_r1/results.json",
}
DEFERRED = pytest.StashKey[dict[str, str]]()


def pytest_addoption(parser):
    parser.addoption(
        "--require-local-artifacts",
        action="store_true",
        help="Require the registered non-distributed research artifacts instead of deferring modules",
    )


def pytest_ignore_collect(collection_path, config):
    if collection_path.parent != ROOT / "tests":
        return None
    rel = LOCAL_ARTIFACTS.get(collection_path.name)
    if rel is None or (ARTIFACT_ROOT / rel).is_file():
        return None
    if config.getoption("--require-local-artifacts"):
        raise pytest.UsageError(f"{collection_path.name}: missing local artifact {rel}")
    config.stash.setdefault(DEFERRED, {})[collection_path.name] = rel
    return True


def pytest_terminal_summary(terminalreporter, config):
    deferred = config.stash.get(DEFERRED, {})
    if deferred:
        terminalreporter.section(f"{len(deferred)} local-artifact modules not collected")
        for name, path in sorted(deferred.items()):
            terminalreporter.write_line(f"{name}: missing {path}")
