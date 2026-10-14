#!/usr/bin/env python3
"""Bounded exact-law paired-proxy experiment; not neural accuracy search."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_proxy_measurement as M

SPEC = ROOT / "configs/grain_proxy_measurement_hm252.json"
CODE = (
    "configs/grain_proxy_measurement_hm252.json",
    "docs/GRAIN_PROXY_MEASUREMENT_PLAN_HM252.md",
    "grain_research/construction/math_proxy_measurement.py",
    "scripts/run_grain_proxy_measurement_hm252.py",
    "scripts/score_grain_proxy_measurement_cold.py",
    "scripts/verify_grain_proxy_measurement_hm252.py",
    "tests/test_grain_proxy_measurement.py",
)


def read(path):
    """Read a named nonsecret artifact."""
    return json.loads(Path(path).read_text())


def sha(path):
    """Hash frozen inputs and science stages."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    """Create JSON exclusively, retaining failed or historical outputs."""
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def archive(path, **arrays):
    """Create a pickle-free numerical stage without overwrite."""
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def seal(out, name, paths):
    """Freeze one completed stage; later phases cannot rewrite it."""
    write(out / name, {p: sha(out / p) for p in paths})


def check(out):
    """Read-only check of code, historical alphabet and completed stages."""
    manifest = read(out / "manifest.json")
    for path, digest in (manifest["code"] | manifest["inputs"]).items():
        if sha(path) != digest:
            raise ValueError("Changed scientific dependency: " + path)
    for name in (
        "prepared_seal.json",
        "observed_seal.json",
        "analyzed_seal.json",
        "verification_seal.json",
    ):
        if (out / name).exists():
            for path, digest in read(out / name).items():
                if sha(out / path) != digest:
                    raise ValueError("Changed stage artifact: " + path)


def prepare(out):
    """Freeze a historical vector alphabet and the entire finite design."""
    if out.exists():
        raise FileExistsError("Existing science results must be retained")
    config = read(SPEC)
    source = ROOT / config["alphabet"]
    with np.load(source, allow_pickle=False) as data:
        alphabet = data["menus"][:4].copy()
    M.decode(alphabet, alphabet)
    out.mkdir(parents=True)
    archive(out / "alphabet.npz", alphabet=alphabet)
    write(
        out / "manifest.json",
        {
            "packet": "HM252",
            "code": {str(ROOT / p): sha(ROOT / p) for p in CODE},
            "inputs": {str(source): sha(source)},
            "scope": "Exact weighted simulator census, not people/text generalization",
            "no_neural_API_private_training": True,
        },
    )
    seal(out, "prepared_seal.json", ["alphabet.npz", "manifest.json"])


def simulator_joint(config, bank=0, coarse=False):
    """Generate observations in a declared world; not imported by cold scoring."""
    w = np.asarray(config["w_channels"], float).copy()
    z = np.asarray(config["z_channels"], float)
    behavior = np.asarray(config["behavior_channels"], float)
    prior = np.asarray(config["source_priors"][bank], float)
    if coarse:
        w[:, 2] = w[:, 1]
    joint = np.zeros((2, 4, 4, 2, 2))
    for c, mass in enumerate(config["source_condition_mass"]):
        for u in range(3):
            for zz in range(4):
                for ww in range(4):
                    for y0 in range(2):
                        for y1 in range(2):
                            py0 = behavior[c, u, 0] if y0 else 1 - behavior[c, u, 0]
                            py1 = behavior[c, u, 1] if y1 else 1 - behavior[c, u, 1]
                            joint[c, zz, ww, y0, y1] += (
                                mass
                                * prior[c, u]
                                * z[c, u, zz]
                                * w[c, u, ww]
                                * py0
                                * py1
                            )
    return joint


def census_records(joint, alphabet, bank):
    """Realize observable census cells as distinct weighted evidence units."""
    c, z, w, y0, y1 = np.indices(joint.shape).reshape(5, -1)
    units = np.array(
        [[f"{bank}:cell{i}:{role}" for role in ("z", "w", "y")] for i in range(len(c))]
    )
    return {
        "conditions": c,
        "z_vectors": alphabet[z],
        "w_vectors": alphabet[w],
        "outcomes": np.column_stack((y0, y1)),
        "weights": joint.flatten(),
        "units": units,
    }


def observe(out):
    """Compile source-only bridges and export independent cold inputs/truth."""
    check(out)
    started = time.monotonic()
    config = read(SPEC)
    with np.load(out / "alphabet.npz", allow_pickle=False) as data:
        alphabet = data["alphabet"]
    paths = []
    bundles, source_joint = [], []
    for bank in range(2):
        records = census_records(simulator_joint(config, bank), alphabet, bank)
        archive(out / f"source_records_{bank}.npz", **records)
        joint = M.observed_joint(**records, alphabet=alphabet)
        source_joint.append(joint)
        bundle = M.compile_bundle(
            joint, alphabet, config["targets"], config["rank_tolerance"]
        )
        bundles.append(bundle)
        write(out / f"bundle_{bank}.json", bundle)
        paths += [f"source_records_{bank}.npz", f"bundle_{bank}.json"]
    copied = census_records(source_joint[0], alphabet, "copy")
    copied["z_vectors"] = copied["w_vectors"].copy()
    copied["units"][:, 0] = copied["units"][:, 1]
    try:
        M.observed_joint(**copied, alphabet=alphabet)
    except ValueError as error:
        copied_rejection = str(error)
    else:
        raise AssertionError("Copied event passed the evidence ledger")
    copied_joint = np.zeros_like(source_joint[0])
    for w in range(4):
        copied_joint[:, w, w] = source_joint[0][:, :, w].sum(axis=1)
    bypass = M.compile_bundle(copied_joint, alphabet, config["targets"])
    coarse = M.compile_bundle(
        simulator_joint(config, coarse=True), alphabet, config["targets"]
    )
    alien = [M.functional_reading(p, [1, 0, 0, 0]) for p in bundles[0]["conditions"]]
    write(
        out / "negative_cases.json",
        {
            "copied_event_rejection": copied_rejection,
            "copied_if_ledger_bypassed": bypass,
            "coarsened_W": coarse,
            "alien_law": alien,
        },
    )
    paths.append("negative_cases.json")
    vectors, conditions, weights, identifiers, truths = [], [], [], [], []
    behavior = np.asarray(config["behavior_channels"], float)
    for channel in ("stable", "reversed_W"):
        w = np.asarray(config["w_channels"], float)
        if channel == "reversed_W":
            w = w[:, [2, 1, 0]]
        for profile in config["application_profiles"]:
            regimes = np.asarray(profile["regimes"], float)
            choice = np.asarray(profile["choice"], float)
            law = np.einsum("cu,cuw->cw", regimes, w)
            conditions.append(np.repeat([0, 1], 4))
            vectors.append(np.tile(alphabet, (2, 1)))
            weights.append((choice[:, None] * law).flatten())
            identifiers.append(channel + ":" + profile["id"])
            truths.append(np.einsum("c,cu,cuq->q", choice, regimes, behavior))
    archive(
        out / "cold_inputs.npz",
        vectors=vectors,
        conditions=conditions,
        weights=weights,
        ids=identifiers,
    )
    archive(
        out / "evaluation_only.npz", behavior_truth=truths, source_joint=source_joint
    )
    write(
        out / "observation_quality.json",
        {
            "source_census_cells": 256,
            "source_vector_observations": 512,
            "application_profiles": 12,
            "new_weighted_vector_observations": 96,
            "not_sample_size_or_new_humans": True,
            "source_rank": [[p["rank"] for p in b["conditions"]] for b in bundles],
            "bridge_status": [[p["status"] for p in b["conditions"]] for b in bundles],
            "seconds": time.monotonic() - started,
            "compiler_inputs": "observed joint census and alphabet only",
            "sampled_precision_search": False,
        },
    )
    paths += ["cold_inputs.npz", "evaluation_only.npz", "observation_quality.json"]
    if time.monotonic() - started > config["cpu_seconds_cap"]:
        raise TimeoutError("Frozen CPU cap exceeded; do not extend")
    seal(out, "observed_seal.json", paths)


def analyze(out):
    """Compare sealed cold readings to evaluation-only behavior means."""
    check(out)
    config = read(SPEC)
    cold = [read(out / f"cold_{bank}.json") for bank in range(2)]
    with np.load(out / "evaluation_only.npz", allow_pickle=False) as data:
        truth = data["behavior_truth"]
    rows = []
    for bank, result in enumerate(cold):
        for index, case in enumerate(result["profiles"]):
            for q, target in enumerate(case["reading"]["targets"]):
                value = target["absolute"]
                rows.append(
                    {
                        "source_bank": bank,
                        "profile": case["id"],
                        "target": target["target"],
                        "status": target["status"],
                        "bridge": value,
                        "ordinary": target["ordinary"],
                        "criterion_mean": float(truth[index, q]),
                        "bridge_error": value - float(truth[index, q])
                        if value is not None
                        else None,
                        "ordinary_error": target["ordinary"] - float(truth[index, q]),
                        "source_relative": target["source_relative"],
                    }
                )
    with (out / "behavior_readings.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    stable = [r for r in rows if r["profile"].startswith("stable:")]
    changed = [r for r in rows if r["profile"].startswith("reversed_W:")]
    negative = read(out / "negative_cases.json")
    copied_program = negative["copied_if_ledger_bypassed"]
    with np.load(out / "cold_inputs.npz", allow_pickle=False) as data:
        copied_profiles = [
            M.cold_profile(
                copied_program,
                data["conditions"][i],
                data["vectors"][i],
                data["weights"][i],
            )
            for i in range(6)
        ]
    copied_error = [
        x["targets"][q]["absolute"] - float(truth[i, q])
        for i, x in enumerate(copied_profiles)
        for q in range(2)
    ]
    b = np.array(
        [
            [
                p["reading"]["targets"][q]["absolute"]
                for p in c["profiles"][:6]
                for q in range(2)
            ]
            for c in cold
        ]
    )
    ordinary = np.array(
        [
            [
                p["reading"]["targets"][q]["ordinary"]
                for p in c["profiles"][:6]
                for q in range(2)
            ]
            for c in cold
        ]
    )
    summary = {
        "packet": "HM252",
        "scope": "Exact simulated measurement means, not r/AUC/human prediction accuracy",
        "stable_readings": len(stable),
        "stable_bridge_max_error": max(abs(r["bridge_error"]) for r in stable),
        "stable_ordinary_mean_abs_error": float(
            np.mean([abs(r["ordinary_error"]) for r in stable])
        ),
        "stable_ordinary_max_abs_error": max(abs(r["ordinary_error"]) for r in stable),
        "bridge_bank_difference_max": float(np.max(np.abs(b[0] - b[1]))),
        "ordinary_bank_difference_mean_abs": float(
            np.mean(np.abs(ordinary[0] - ordinary[1]))
        ),
        "ordinary_bank_difference_max": float(
            np.max(np.abs(ordinary[0] - ordinary[1]))
        ),
        "stable_score_range": [float(b.min()), float(b.max())],
        "choice_contrast_a2_minus_a0": (b[0, 4:6] - b[0, :2]).tolist(),
        "reversed_channel_max_error": max(abs(r["bridge_error"]) for r in changed),
        "reversed_channel_mean_abs_error": float(
            np.mean([abs(r["bridge_error"]) for r in changed])
        ),
        "copied_zero_equation_residual": max(
            max(p["residual"]) for p in copied_program["conditions"]
        ),
        "copied_mean_abs_behavior_error": float(np.mean(np.abs(copied_error))),
        "copied_event_rejected": True,
        "coarse_statuses": [p["status"] for p in negative["coarsened_W"]["conditions"]],
        "alien_statuses": [[x["status"] for x in c] for c in negative["alien_law"]],
        "source_references": [
            read(out / f"bundle_{i}.json")["reference"] for i in range(2)
        ],
        "primary_exact_construction_PASS": bool(
            np.max(np.abs(b[0] - b[1])) < config["equation_tolerance"]
            and all(
                abs(r["bridge_error"]) < config["equation_tolerance"] for r in stable
            )
        ),
        "original18_100_44_complete": False,
        "negative_transport_preserved": True,
    }
    write(out / "summary.json", summary)
    seal(
        out,
        "analyzed_seal.json",
        ["behavior_readings.csv", "summary.json", "cold_0.json", "cold_1.json"],
    )


def main():
    """Run exactly one named stage; no automatic science retries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "observe", "analyze", "check"])
    parser.add_argument("--out", type=Path, default=ROOT / read(SPEC)["output"])
    args = parser.parse_args()
    {"prepare": prepare, "observe": observe, "analyze": analyze, "check": check}[
        args.stage
    ](args.out.resolve())
    print(
        json.dumps(
            {"stage": args.stage, "out": str(args.out.resolve()), "status": "COMPLETE"}
        )
    )


if __name__ == "__main__":
    main()
