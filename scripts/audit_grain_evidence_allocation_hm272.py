#!/usr/bin/env python3
"""Post-start provenance annex; never rewrite failed or active science manifests."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_observed_family_hm170 as R


def main():
    """Audit failure provenance and all actual bank-support receipts without refitting."""
    primary = ROOT / "results/grain_math_response_v1/evidence_allocation_r1"
    recovery = ROOT / "results/grain_math_response_v1/evidence_allocation_logits_r1"
    diagnostic = primary / "support_diagnosis"
    R.check(primary)
    R.check(
        recovery,
        ("collected_seal.json", "analyzed_seal.json", "verification_seal.json"),
    )
    runtime = R.read(primary / "runtime.json")
    dm = R.read(diagnostic / "manifest.json")
    for path, expected in dm["code"].items():
        if R.digest(ROOT / path) != expected:
            raise ValueError("Diagnosis code changed")
    if (
        R.digest(primary / "manifest.json") != dm["parent_manifest"]
        or R.digest(primary / "runtime.json") != dm["parent_failed_runtime"]
    ):
        raise ValueError(
            "Failed primary was changed after the native support diagnosis"
        )
    for path, expected in R.read(diagnostic / "diagnosed_seal.json").items():
        if R.digest(diagnostic / path) != expected:
            raise ValueError("Native support diagnosis changed")
    if runtime["state"] != "FAILED" or runtime["done_batches"] != 0:
        raise ValueError("Primary failure must remain a failure, not a zero effect")
    hooks = R.read(recovery / "hook_receipts.json")
    counts = {
        k: sum(r[k] for h in hooks for r in h["layers"])
        for k in ("zero_native_first", "zero_native_second", "zero_native_combined")
    }
    if not all(r["finite_bank_logits"] for h in hooks for r in h["layers"]):
        raise ValueError("No conditional law recovery at an unavailable bank")
    quality = R.read(recovery / "collection_quality.json")
    diag = R.read(diagnostic / "diagnosis.json")
    seconds = (
        runtime["model_seconds"] + diag["model_seconds"] + quality["model_seconds"]
    )
    if seconds >= 1200:
        raise ValueError("Original combined resource contract exceeded")
    paths = [
        primary / "manifest.json",
        primary / "runtime.json",
        diagnostic / "manifest.json",
        diagnostic / "diagnosed_seal.json",
        diagnostic / "bank_masses.npz",
        diagnostic / "diagnosis.json",
    ]
    R.write(
        recovery / "recovery_provenance_annex.json",
        {
            "PASS": True,
            "annex_created_after_collection_start": True,
            "not_a_preregistered_input_seal": True,
            "prepare_aux_augmentation_exit": 1,
            "core_prepared_manifest_unchanged": True,
            "referenced_failed_primary_and_diagnosis": {
                str(p.relative_to(ROOT)): R.digest(p) for p in paths
            },
            "read_stage_occurrences_not_independent_events": counts,
            "combined_model_seconds": seconds,
            "all_registered_bank_logits_finite_and_masks_checked": True,
            "same_agent_not_blind": True,
            "audit_script_sha256": R.digest(__file__),
        },
    )
    R.seal(recovery, "provenance_annex_seal.json", ("recovery_provenance_annex.json",))
    print(
        {
            "PASS": True,
            "combined_model_seconds": seconds,
            "support_occurrences": counts,
            "annex_is_post_start": True,
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
