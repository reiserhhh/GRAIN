#!/usr/bin/env python3
"""Repair only the offline addressed reader; never edit or rerun HM259 science."""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_continuation_routes_hm259 as F
from scripts import run_grain_late_condition_hm258 as P
from scripts import run_grain_observed_family_hm170 as R
from scripts import run_grain_source_meaning_bridge_hm233 as S

BASE = F.DEFAULT
DEFAULT = BASE / "analysis_recovery_r1"
SCIENCE = (
    "docs/GRAIN_CONTINUATION_ROUTES_ANALYSIS_RECOVERY_HM259.md",
    "scripts/recover_grain_continuation_routes_hm259.py",
    "tests/test_grain_continuation_routes_recovery.py",
)


def readings(base):
    """Keep the outer result container distinct from individual array lists."""
    cfg = R.read(base / "manifest.json")["config"]
    cases = R.read(base / "cases.json")
    old_cases, native = P.observations(ROOT / cfg["parent"])
    if len(cases) != len(old_cases):
        raise ValueError("All original author uses must remain")
    units = []
    for case, old_case, old in zip(cases, old_cases, native, strict=True):
        if (case["unit_id"], case["target_ids"]) != (
            old_case["unit_id"],
            old_case["target_ids"],
        ):
            raise ValueError("Actual source/target identity changed")
        family = {"native": old}
        for route in cfg["new_routes"]:
            by_key = {k: [] for k in ("chosen", "partition")}
            for mode in range(2):
                chunks = {k: [] for k in by_key}
                for start in range(0, 11, 4):
                    path = base / F.capture_name(case["unit_id"], mode, route, start)
                    with np.load(path, allow_pickle=False) as z:
                        if (
                            int(z["unit"]),
                            int(z["mode"]),
                            str(z["route"]),
                            int(z["start"]),
                        ) != (case["unit_id"], mode, route, start):
                            raise ValueError("Exact captured graph address changed")
                        for key, pieces in chunks.items():
                            block = z[key].astype(float)
                            if (
                                block.shape
                                != (min(4, 11 - start), len(case["target_ids"]))
                                or not np.isfinite(block).all()
                            ):
                                raise ValueError(
                                    "Missing/nonfinite/partial observation block"
                                )
                            pieces.append(block)
                for key, context_blocks in by_key.items():
                    context_blocks.append(np.concatenate(chunks[key]))
            family[route] = {key: np.stack(blocks) for key, blocks in by_key.items()}
        units.append(family)
    return cases, units


def prepare(out, base):
    """Create separate analysis provenance with unchanged source observations."""
    if out.exists():
        raise FileExistsError("Do not replace any historical recovery")
    manifest = R.check(base, ("collected_seal.json",))
    if (base / "analyzed_seal.json").exists() or (base / "actions.npz").exists():
        raise ValueError("Failure provenance expected no analyzed outputs")
    runtime = R.read(base / "runtime.json")
    if runtime["state"] != "COMPLETE" or runtime["done_batches"] != 864:
        raise ValueError("No partial neural collection may be rescued")
    out.mkdir(parents=True, exist_ok=False)
    for name in ("cases.json", "collection_quality.json", "input_quality.json"):
        R.write(out / name, R.read(base / name))
    R.archive(
        out / "source_observations.npz",
        **S.selected(
            base / "source_observations.npz", ("fields", "choice", "coordinates")
        ),
    )
    inputs = dict(manifest["inputs"])
    for name in (
        "manifest.json",
        "prepared_seal.json",
        "collected_seal.json",
        "cases.json",
        "source_observations.npz",
        "collection_quality.json",
        "runtime.json",
    ):
        path = base / name
        inputs[str(path.relative_to(ROOT))] = R.digest(path)
    for name in R.read(base / "collected_seal.json"):
        path = base / name
        inputs[str(path.relative_to(ROOT))] = R.digest(path)
    R.write(
        out / "manifest.json",
        {
            **manifest,
            "code": {**manifest["code"], **{p: R.digest(ROOT / p) for p in SCIENCE}},
            "inputs": inputs,
        },
    )
    R.write(
        out / "recovery.json",
        {
            "original_analysis_handle": 23162,
            "original_analysis_exit": 1,
            "original_error": "AttributeError: 'numpy.ndarray' object has no attribute 'items'",
            "cause": "Loop name values overwrote the outer observation container",
            "original_analysis_no_outputs": True,
            "original_collection_unchanged": True,
            "changed_only": "Offline reader and absolute capture path",
            "model_reruns_API_training": 0,
            "original_scope_complete": False,
        },
    )
    R.seal(
        out,
        "prepared_seal.json",
        (
            "cases.json",
            "source_observations.npz",
            "input_quality.json",
            "recovery.json",
        ),
    )
    R.seal(out, "collected_seal.json", ("collection_quality.json",))
    print("OFFLINE_RECOVERY_PREPARED", flush=True)


def main():
    """Delegate unchanged analysis algebra with scoped corrected file readers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "analyze", "verify", "check"))
    parser.add_argument("--out", type=Path, default=DEFAULT)
    parser.add_argument("--base", type=Path, default=BASE)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.out, args.base)
    elif args.stage == "check":
        R.check(
            args.out,
            ("collected_seal.json", "analyzed_seal.json", "verification_seal.json"),
        )
        print("ALL_SEALS_PASS")
    else:
        old_reader, old_name = F.observations, F.capture_name
        try:
            F.observations = lambda unused: readings(args.base)
            F.capture_name = lambda *parts: str(args.base / old_name(*parts))
            getattr(F, args.stage)(args.out)
        finally:
            F.observations, F.capture_name = old_reader, old_name


if __name__ == "__main__":
    main()
