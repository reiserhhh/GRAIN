#!/usr/bin/env python3
"""One declared K/V-export compatibility repair under HM274's ORIGINAL budget."""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_observed_family_hm170 as R
from scripts import run_grain_prospective_evidence_hm218 as H
from scripts import run_grain_same_regime_reader_hm274 as E

PRIMARY = E.DEFAULT
DEFAULT = PRIMARY / "cache_export_recovery_r1"
SCIENCE = (
    "scripts/run_grain_same_regime_reader_hm274_recovery.py",
    "tests/test_grain_same_regime_cache_recovery.py",
    "docs/GRAIN_SAME_REGIME_CACHE_RECOVERY_HM274.md",
)


def cache_export_values(exports):
    """Preserve both exact K/V arrays while adapting a mapping to ordered values."""
    if set(exports) != {"keys", "values"}:
        raise ValueError("Both full K/V exports required")
    arrays = tuple(np.asarray(exports[k]) for k in ("keys", "values"))
    if arrays[0].shape != arrays[1].shape or any(
        a.ndim < 2 or a.dtype.kind != "f" or not np.isfinite(a).all() for a in arrays
    ):
        raise ValueError("Matched finite floating K/V arrays required")
    return arrays


def prepare(out):
    """Preserve the failed primary and freeze ONE equivalent mechanical repair."""
    if out.exists():
        raise FileExistsError("No replacement or second repair")
    original = R.check(PRIMARY)
    failed = R.read(PRIMARY / "runtime.json")
    if (
        failed["state"] != "FAILED"
        or failed["done_batches"] != 7
        or failed["error"] != "unsupported operand type(s) for -: 'str' and 'str'"
        or list((PRIMARY / "captures").iterdir())
        or (PRIMARY / "ACTIVE.lock").exists()
    ):
        raise ValueError("This companion only repairs the registered export-type fault")
    used = float(failed["model_seconds"])
    original_cap = original["config"]["max_model_seconds"]
    if not 0 < used < original_cap:
        raise ValueError("A remaining ORIGINAL finite cap is required")
    manifest = dict(original)
    manifest["config"] = {
        **original["config"],
        "max_model_seconds": original_cap - used,
    }
    manifest["code"] = {**original["code"], **{p: R.digest(ROOT / p) for p in SCIENCE}}
    manifest["inputs"] = dict(original["inputs"])
    names = tuple(R.read(PRIMARY / "prepared_seal.json"))
    for p in names + (
        "runtime.json",
        "native_guard_0.npz",
        "manifest.json",
        "prepared_seal.json",
    ):
        manifest["inputs"][str((PRIMARY / p).relative_to(ROOT))] = R.digest(PRIMARY / p)
    manifest["repair"] = {
        "failed_primary": str(PRIMARY.relative_to(ROOT)),
        "cause": "dictionary keys were iterated as if they were numerical arrays",
        "only_changed_operation": "map keys/values export to two numerical arrays",
        "primary_model_seconds": used,
        "original_total_cap_seconds": original_cap,
        "remaining_cap_seconds": original_cap - used,
        "first50_candidate_accounts_lost_before_archive_reobserved_once": True,
        "primary50_candidates_are_not_independent_replication": True,
        "no_model_target_material_kernel_parameter_or_score_change": True,
    }
    out.mkdir(parents=True)
    (out / "captures").mkdir()
    for p in names:
        shutil.copyfile(PRIMARY / p, out / p)
    R.write(out / "manifest.json", manifest)
    R.seal(out, "prepared_seal.json", names)
    H.status(
        out,
        {
            "state": "PREPARED",
            "done_batches": 0,
            "model_seconds": 0,
            "previous_failed_model_seconds": used,
        },
    )
    print(
        f"One mechanical repair prepared; ORIGINAL remaining cap {original_cap - used:.6f}s",
        flush=True,
    )


def main():
    """Operate the frozen equivalent collection, never the failed original files."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "stage", choices=("prepare", "collect", "analyze", "verify", "check")
    )
    p.add_argument("--out", type=Path, default=DEFAULT)
    args = p.parse_args()
    if args.stage == "prepare":
        prepare(args.out)
    elif args.stage == "collect":
        original = E.L.cache_arrays
        E.L.cache_arrays = lambda cache: cache_export_values(original(cache))
        try:
            E.collect(args.out)
        finally:
            E.L.cache_arrays = original
    elif args.stage == "check":
        R.check(
            args.out,
            ("collected_seal.json", "analyzed_seal.json", "verification_seal.json"),
        )
        print("HM274 equivalent cache-export repair seals PASS")
    else:
        getattr(E, args.stage)(args.out)


if __name__ == "__main__":
    main()
