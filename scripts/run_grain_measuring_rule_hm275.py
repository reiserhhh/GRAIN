#!/usr/bin/env python3
"""One fixed common-target/reference discriminator using exact worlds and old caches."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_measuring_rule as M
from scripts import run_grain_observed_family_hm170 as R

CONFIG = "configs/grain_measuring_rule_hm275.json"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_MEASURING_RULE_PLAN_HM275.md",
    "grain_research/construction/math_measuring_rule.py",
    "grain_research/construction/math_source_event_coupling.py",
    "grain_research/construction/math_relational_diagnostic_update.py",
    "scripts/score_grain_measuring_rule_cold.py",
    "scripts/run_grain_measuring_rule_hm275.py",
    "scripts/verify_grain_measuring_rule_hm275.py",
    "scripts/run_grain_observed_family_hm170.py",
    "tests/test_grain_measuring_rule.py",
)
DEFAULT = ROOT / "results/grain_math_response_v1/measuring_rule_r1"
OLD_FILES = (
    "manifest.json",
    "prepared_seal.json",
    "computed_seal.json",
    "verification_seal.json",
    "source_pair_bank.npz",
    "log_kernel_accounts.npz",
    "conditional_profiles.npz",
    "qualified_panel_pairs.npz",
    "example_pair.json",
)


def prepare(out):
    """Freeze one finite plan and old numerical inputs without reading new outcomes."""
    if out.exists():
        raise FileExistsError("Preserve old failures and frozen science")
    cfg = R.read(ROOT / CONFIG)
    parent = ROOT / cfg["source_parent"]
    R.check(parent, ("computed_seal.json", "verification_seal.json"))
    out.mkdir(parents=True)
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": {
                str((parent / p).relative_to(ROOT)): R.digest(parent / p)
                for p in OLD_FILES
            },
            "original_scope_complete": False,
            "prior_art": "finite Bayes/tower/proper-risk; not novel mathematical results",
        },
    )
    R.write(
        out / "input_contract.json",
        {
            "worlds": 3,
            "worlds_are_declared_census_not_people": True,
            "source_authors": 386,
            "historical_dependent_panel_pairs": 528,
            "source_label_target_not_independent_behavior": True,
            "native_reader_answers_not_fitted": True,
            "new_neural_API_private_training_calls": 0,
        },
    )
    R.seal(out, "prepared_seal.json", ("input_contract.json",))
    print("PREPARED HM275 fixed3worlds/all528oldpairs;no precision search", flush=True)


def compute(out):
    """Compile all reference types once, without selecting signs or targets."""
    cfg = R.check(out)["config"]
    parent = ROOT / cfg["source_parent"]
    R.write(
        out / "compute_started.json",
        {"state": "RUNNING", "cap": cfg["max_compute_seconds"]},
    )
    began = time.perf_counter()
    try:
        worlds, world_rows, identities = {}, [], []
        for i, rho in enumerate(cfg["world_rhos"]):
            w, rows, a = M.world_accounts(
                cfg["world_accuracy"], rho, cfg["world_prior"]
            )
            worlds |= {f"world{i}/{k}": v for k, v in w.items()}
            world_rows.extend(rows)
            identities.append(a)
        R.archive(out / "worlds.npz", **worlds)
        R.table(out / "world_readers.csv", world_rows)
        R.write(out / "world_identities.json", identities)
        with np.load(parent / "source_pair_bank.npz", allow_pickle=False) as z:
            codes, folds, relink = z["codes"], z["folds"], z["relink"]
        if len(codes) != cfg["source_authors"]:
            raise ValueError("Changed SOURCE scope")
        all_accounts, description = {}, {}
        with np.load(parent / "log_kernel_accounts.npz", allow_pickle=False) as stored:
            for space in ("E5", "GRAIN15"):
                logs = stored[space][: cfg["panel_pairs"]]
                if logs.shape != (528, 386, 2, 22):
                    raise ValueError("Changed complete historical pair scope")
                for bank, mask in (
                    ("bank0", folds == 0),
                    ("bank1", folds == 1),
                    ("pooled", np.ones(386, bool)),
                ):
                    key = f"{space}/{bank}"
                    v = M.source_rule(logs[:, mask], codes[mask], relink[mask])
                    all_accounts |= {
                        f"{key}/{field}": value for field, value in v.items()
                    }
                    description[key] = {
                        "pairs": 528,
                        "reference": v["source_reference"].tolist(),
                        "model_replacement_rms": np.sqrt(
                            np.mean(v["model_replacement"] ** 2, axis=0)
                        ).tolist(),
                        "evidence_after_left_rms": np.sqrt(
                            np.mean(v["evidence_after_left"] ** 2, axis=0)
                        ).tolist(),
                        "evidence_vs_model_rms": np.sqrt(
                            np.mean(
                                (v["evidence_after_left"] - v["model_replacement"])
                                ** 2,
                                axis=0,
                            )
                        ).tolist(),
                        "not_accuracy_correlations_significance_or_information_fractions": True,
                    }
                    if time.perf_counter() - began > cfg["max_compute_seconds"]:
                        raise TimeoutError("No budget expansion or retry")
        R.archive(out / "source_rule_accounts.npz", **all_accounts)
        command = [
            sys.executable,
            str(ROOT / "scripts/score_grain_measuring_rule_cold.py"),
            "--bundle",
            str(parent / "source_pair_bank.npz"),
            "--record",
            str(parent / "example_pair.json"),
        ]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        R.write(out / "cold_readback.json", json.loads(result.stdout))
        elapsed = time.perf_counter() - began
        if elapsed > cfg["max_compute_seconds"]:
            raise TimeoutError("Total compute including writes exceeded the frozen cap")
        correct = next(
            r
            for r in world_rows
            if r["rho"] == 0.5 and r["reader"] == "perfect_same_target"
        )
        wrong = next(
            r
            for r in world_rows
            if r["rho"] == 0.5 and r["reader"] == "complement_wrong_key"
        )
        R.write(
            out / "summary.json",
            {
                "seconds": elapsed,
                "world_rows": world_rows,
                "identities": identities,
                "H_sign_refuted_on_declared_world": correct["contact"] < 0
                and correct["joint_over_own_singleton_mean_risk_gain"] > 0
                and correct["regret_against_declared_Y"] == 0,
                "positive_contact_can_have_wrong_target": wrong["contact"] > 0
                and wrong["regret_against_declared_Y"] > 0,
                "SOURCE_rule": description,
                "negative_natural_contact_mechanism_not_identified_by_this_census": True,
                "original_scope_complete": False,
                "new_neural_API_private_training_or_application_label_use": 0,
            },
        )
        R.write(out / "runtime.json", {"state": "COMPLETE", "seconds": elapsed})
        R.seal(
            out,
            "computed_seal.json",
            (
                "worlds.npz",
                "world_readers.csv",
                "world_identities.json",
                "source_rule_accounts.npz",
                "cold_readback.json",
                "summary.json",
                "runtime.json",
            ),
        )
        print(
            f"COMPLETE HM275 oneCPUstage {elapsed:.6f}s/all6SOURCEcontracts", flush=True
        )
    except Exception as exc:
        R.write(
            out / "runtime.json",
            {
                "state": "FAILED_PRESERVED",
                "seconds": time.perf_counter() - began,
                "error": repr(exc),
            },
        )
        raise


def main():
    """Do not overwrite or automatically resume a failed scientific compute."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "compute", "check"))
    p.add_argument("--out", type=Path, default=DEFAULT)
    args = p.parse_args()
    if args.stage == "check":
        R.check(args.out, ("computed_seal.json", "verification_seal.json"))
        print("CHECKED HM275 allcode/input/outputseals")
    else:
        globals()[args.stage](args.out)


if __name__ == "__main__":
    main()
