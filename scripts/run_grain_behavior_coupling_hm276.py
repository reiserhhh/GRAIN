#!/usr/bin/env python3
"""One finite construction of paired criterion meaning, with no prediction grid."""

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
from grain_research.construction import math_behavior_coupling as M
from scripts import run_grain_observed_family_hm170 as R

CONFIG = "configs/grain_behavior_coupling_hm276.json"
DEFAULT = ROOT / "results/grain_math_response_v1/behavior_coupling_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_BEHAVIOR_COUPLING_PLAN_HM276.md",
    "grain_research/construction/math_behavior_coupling.py",
    "grain_research/construction/math_proxy_measurement.py",
    "scripts/score_grain_behavior_coupling_cold.py",
    "scripts/run_grain_behavior_coupling_hm276.py",
    "scripts/verify_grain_behavior_coupling_hm276.py",
    "scripts/run_grain_observed_family_hm170.py",
    "tests/test_grain_behavior_coupling.py",
)


def channels(config, gamma):
    """Evaluator-only generator, never imported by the cold scoring module."""
    bits = M.BITS
    matrices = []
    for name in ("z_accuracy", "w_accuracy"):
        a = config[name]
        matrices.append(
            np.prod(np.where(bits[:, None] == bits[None], a, 1 - a), axis=-1)
        )
    means = config["behavior_low"] + config["behavior_slope"] * bits
    cross = np.prod(means, axis=1) + gamma * np.prod(2 * bits - 1, axis=1)
    behavior = np.column_stack(
        (1 - means.sum(axis=1) + cross, means[:, 1] - cross, means[:, 0] - cross, cross)
    )
    if np.any(behavior <= 0):
        raise ValueError("Fixed behavior law must have positive complete support")
    return *matrices, behavior


def author_prior(kappa):
    """Generate distinct author joint laws with identical coordinate marginals."""
    return (1 + kappa * np.prod(2 * M.BITS - 1, axis=1)) / 4


def vector_pairs(alphabet):
    """Emit role-bound actual finite prototypes, not fabricated neural vectors."""
    return alphabet[M.BITS + [0, 2]]


def census_records(joint, alphabet, key):
    """Create observed exact-weight source records, excluding latent coordinates."""
    z, w, y = np.indices(joint.shape).reshape(3, -1)
    return {
        "z_vectors": vector_pairs(alphabet)[z],
        "w_vectors": vector_pairs(alphabet)[w],
        "outcomes": M.BITS[y],
        "weights": joint.flatten(),
        "units": np.array(
            [[f"{key}:{i}:{j}" for j in range(6)] for i in range(len(z))]
        ),
        "frame": M.FRAME,
    }


def record(law, alphabet, key):
    """Export only paired vectors, their observed law and binding metadata."""
    return {
        "id": key,
        "frame": list(M.FRAME),
        "vectors": vector_pairs(alphabet).tolist(),
        "weights": np.asarray(law).tolist(),
        "units": [[f"{key}:{i}:A", f"{key}:{i}:B"] for i in range(4)],
    }


def prepare(out):
    """Freeze one complete finite method before source/behavior construction."""
    if out.exists():
        raise FileExistsError("Preserve existing science and failures")
    cfg = R.read(ROOT / CONFIG)
    path = ROOT / cfg["alphabet"]
    with np.load(path, allow_pickle=False) as z:
        alphabet = z["menus"][:4].copy()
    M.pair_codes(vector_pairs(alphabet), alphabet)
    out.mkdir(parents=True)
    R.archive(out / "alphabet.npz", alphabet=alphabet)
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": {cfg["alphabet"]: R.digest(path)},
            "scope": "Exact weighted paired observation census; not humans or natural text",
            "original_scope_complete": False,
        },
    )
    R.seal(out, "prepared_seal.json", ("alphabet.npz",))
    print("PREPARED HM276 fixed joint-criterion construction", flush=True)


def compute(out):
    """Construct source bridges and score all fixed new laws exactly once."""
    cfg = R.check(out)["config"]
    R.write(
        out / "compute_started.json",
        {"state": "RUNNING", "cap": cfg["cpu_seconds_cap"]},
    )
    began = time.perf_counter()
    try:
        with np.load(out / "alphabet.npz", allow_pickle=False) as z:
            alphabet = z["alphabet"]
        rows, saved, paths = [], {}, []
        _, w, _ = channels(
            cfg, cfg["criterion_laws"][0]["conditional_joint_adjustment"]
        )
        reference = np.ones(4) @ w / 4
        for ci, item in enumerate(cfg["criterion_laws"]):
            z, w, behavior = channels(cfg, item["conditional_joint_adjustment"])
            for bi, prior in enumerate(cfg["source_priors"]):
                joint = np.einsum("u,uz,uw,uy->zwy", prior, z, w, behavior)
                observed = M.observed_joint(
                    **census_records(joint, alphabet, f"c{ci}b{bi}"), alphabet=alphabet
                )
                saved[f"c{ci}b{bi}/joint"] = observed
                bundle = M.compile_bundle(observed, alphabet, item["id"], reference)
                bp = f"bundle_c{ci}_b{bi}.json"
                R.write(out / bp, bundle)
                paths.append(bp)
                for ai, kappa in enumerate(cfg["author_joint_parameters"]):
                    law = author_prior(kappa) @ w
                    input_name = f"input_a{ai}.json"
                    if not (out / input_name).exists():
                        R.write(out / input_name, record(law, alphabet, f"a{ai}"))
                        paths.append(input_name)
                    reading = M.cold_profile(bundle, R.read(out / input_name))[
                        "reading"
                    ]
                    if reading["status"] != "ADMISSIBLE_JOINT_BEHAVIOR_LAW":
                        raise ValueError(
                            "Retain inadmissible reading, do not rescue this design"
                        )
                    truth = author_prior(kappa) @ behavior
                    means = truth @ M.BITS
                    reshaped = law.reshape(2, 2)
                    relinked = np.outer(
                        reshaped.sum(axis=1), reshaped.sum(axis=0)
                    ).flatten()
                    relink = M.cold_profile(
                        bundle, record(relinked, alphabet, f"r{ai}")
                    )["reading"]
                    rows.append(
                        {
                            "criterion": item["id"],
                            "source_bank": bi,
                            "author_law": ai,
                            "kappa_evaluator_only": kappa,
                            "mean_A": reading["marginal_means"][0],
                            "mean_B": reading["marginal_means"][1],
                            "coupling": reading["coupling"],
                            "held_behavior_coupling": float(truth[3] - np.prod(means)),
                            "all_joint_cells_error": float(
                                np.max(
                                    abs(
                                        np.array(reading["probabilities_00_01_10_11"])
                                        - truth
                                    )
                                )
                            ),
                            "mean_product_coupling": reading[
                                "product_of_author_means_coupling"
                            ],
                            "paired_single_weight_product_coupling": reading[
                                "paired_single_weight_product_coupling"
                            ],
                            "relinked_coupling": relink["coupling"],
                        }
                    )
                    saved[f"c{ci}b{bi}a{ai}/truth"] = truth
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/score_grain_behavior_coupling_cold.py"),
                "--bundle",
                str(out / "bundle_c0_b0.json"),
                "--record",
                str(out / "input_a2.json"),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        R.write(out / "cold_readback.json", json.loads(result.stdout))
        R.archive(out / "observed_accounts.npz", **saved)
        R.table(out / "behavior_coupling.csv", rows)
        elapsed = time.perf_counter() - began
        if elapsed > cfg["cpu_seconds_cap"]:
            raise TimeoutError("Original finite CPU budget exceeded")
        R.write(
            out / "summary.json",
            {
                "seconds": elapsed,
                "rows": rows,
                "source_weighted_observation_cells": 256,
                "fixed_author_laws": 3,
                "source_and_cold_vector_laws_identical_across_criterion_protocols": True,
                "marginal_behavior_means_identical_in_all_author_laws": True,
                "not_Pearson_AUC_accuracy_significance_personality_or_new_math": True,
                "new_neural_API_private_training": 0,
                "original_scope_complete": False,
            },
        )
        paths += [
            "cold_readback.json",
            "observed_accounts.npz",
            "behavior_coupling.csv",
            "summary.json",
        ]
        R.write(out / "runtime.json", {"state": "COMPUTED", "seconds": elapsed})
        paths.append("runtime.json")
        R.seal(out, "computed_seal.json", paths)
        print(
            f"COMPUTED HM276 {len(rows)} fixed joint-law readings {elapsed:.6f}s",
            flush=True,
        )
    except Exception as error:
        path = out / "runtime.json"
        if path.exists():
            path = out / "failure_receipt.json"
        R.write(path, {"state": "FAILED", "error": repr(error)})
        raise


def main():
    """Run one registered phase; check never changes scientific artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "compute", "check"))
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.out)
    elif args.phase == "compute":
        compute(args.out)
    else:
        stages = tuple(
            p
            for p in ("computed_seal.json", "verification_seal.json")
            if (args.out / p).exists()
        )
        R.check(args.out, stages)
        print("CHECKED HM276 frozen method/input/output identities")


if __name__ == "__main__":
    main()
