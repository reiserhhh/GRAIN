#!/usr/bin/env python3
"""HM69: exact author/observation accounts on frozen HM68 factorial readings."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_author_observation_accounts as M
from grain_research.vector_mechanisms.navigation import digest, read_json

SOURCE = ROOT / "results/grain_math_response_v1/ownership_composition_r1"
OUT = ROOT / "results/grain_math_response_v1/author_observation_accounts_r1"
TARGETS = (
    "extraversion",
    "neuroticism",
    "agreeableness",
    "conscientiousness",
    "openness",
    "anthropomorphism",
)
STAGES = (
    "score",
    "log_odds",
    "centered_logits",
    "body_endpoint",
    "score_link_control",
    "log_odds_link_control",
)
CODE = (
    "docs/GRAIN_AUTHOR_OBSERVATION_ACCOUNTS_PLAN_V1.md",
    "grain_research/construction/math_author_observation_accounts.py",
    "grain_research/construction/math_joint_law.py",
    "grain_research/vector_mechanisms/navigation.py",
    "scripts/run_grain_math_author_observation_accounts.py",
    "tests/test_grain_math_author_observation_accounts.py",
)
PAIRS = tuple(itertools.permutations(TARGETS, 2))
ORDERS = tuple(itertools.permutations(range(3)))


def write(path, value):
    """Save finite local metadata, rejecting NaN."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def csv_write(path, rows):
    """Save full tabular accounts; vectors remain in the accompanying archives."""
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_check():
    """Check actual frozen observations, not only an upstream status label."""
    for seal in ("03/output_seal.json", "01/raw_seal.json"):
        for name, sha in read_json(SOURCE / seal).items():
            if digest(SOURCE / name) != sha:
                raise ValueError("Frozen HM68 changed: " + name)
    if read_json(SOURCE / "verification.json")["actual_calls"] != 3456:
        raise ValueError("Complete HM68 acquisition required")


def prepare(out):
    """Freeze this finite plan, executable dependencies and prior readout hashes."""
    if out.exists():
        raise FileExistsError("Existing experiment must not be overwritten")
    source_check()
    names = (
        "00/manifest.json",
        "01/raw_seal.json",
        "03/output_seal.json",
        "03/stages_and_worlds.npz",
        "03/summary.json",
        "verification.json",
        "04_summary_readback/verification.json",
    )
    code = {}
    for name in CODE:
        dst = out / "code" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dst)
        code[name] = digest(dst)
    write(
        out / "manifest.json",
        {
            "packet": "HM69",
            "source": str(SOURCE.relative_to(ROOT)),
            "source_hashes": {name: digest(SOURCE / name) for name in names},
            "code_hashes": code,
            "pairs": PAIRS,
            "cases": M.CASES,
            "stages": STAGES,
            "orders": ORDERS,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "new_neural_calls_training_API_private_labels": [0, 0, 0, 0, 0],
            "existing_readings_new_numerical_profiles_not_people": True,
            "original_scope_complete": False,
        },
    )
    write(out / "checkpoint.json", {"status": "PREPARED_HM69", "active_processes": []})
    print("HM69 prepared: 30 pairs, 8 cases, 6 stages; zero new model calls")


def bound_check(out):
    """Refuse source or code drift after the actual freeze."""
    manifest = read_json(out / "manifest.json")
    for name, sha in manifest["source_hashes"].items():
        if digest(SOURCE / name) != sha:
            raise ValueError("Frozen input changed: " + name)
    for name, sha in manifest["code_hashes"].items():
        if digest(ROOT / name) != sha or digest(out / "code" / name) != sha:
            raise ValueError("Frozen executable changed: " + name)
    return manifest


def observations():
    """Load all declared captured vector stages without a model or label fit."""
    result = {}
    with np.load(SOURCE / "03/stages_and_worlds.npz", allow_pickle=False) as data:
        for stage in STAGES:
            result[stage] = np.asarray(
                [
                    [
                        data[f"f{family}_{j}_{k}_{stage}"].reshape(8, -1)
                        for family in (0, 1)
                    ]
                    for j, k in PAIRS
                ]
            )
    return result


def row_summary(result):
    """Report unit-preserving action magnitudes, not accuracy or mediation shares."""
    phi, seq, inter, delta = (
        result[k] for k in ("shapley", "sequences", "interactions", "delta")
    )
    return {
        "delta_norm": float(np.linalg.norm(delta)),
        "choice_norm": float(np.linalg.norm(phi[0])),
        "expression_norm": float(np.linalg.norm(phi[1])),
        "law_norm": float(np.linalg.norm(phi[2])),
        "order_range_norm": float(np.linalg.norm(np.ptp(seq, axis=0))),
        "joint_interaction_norm": float(np.linalg.norm(inter[[3, 5, 6, 7]])),
        "maximum_sum_error": float(np.max(abs(phi.sum(0) - delta))),
        "maximum_sequence_error": float(np.max(abs(seq.sum(1) - delta))),
    }


def aggregate(rows):
    """Aggregate complete tables separately for each case and observation unit."""
    summary = []
    names = tuple(k for k in rows[0] if k.endswith("_norm"))
    for case, stage in itertools.product(M.CASES, STAGES):
        group = [r for r in rows if r["case"] == case and r["stage"] == stage]
        summary.append(
            {
                "case": case,
                "stage": stage,
                "directed_pairs": len(group),
                **{
                    k + "_RMS": math.sqrt(
                        math.fsum(float(r[k]) ** 2 for r in group) / len(group)
                    )
                    for k in names
                },
                "maximum_sum_error": max(float(r["maximum_sum_error"]) for r in group),
                "maximum_sequence_error": max(
                    float(r["maximum_sequence_error"]) for r in group
                ),
            }
        )
    return summary


def run(out):
    """Execute the whole bounded accounting protocol and save every factor game."""
    bound_check(out)
    if read_json(out / "checkpoint.json")["status"] != "PREPARED_HM69":
        raise ValueError("Do not overwrite or retry a collected packet")
    start = time.monotonic()
    write(
        out / "checkpoint.json",
        {"status": "RUNNING_HM69", "active_processes": [os.getpid()]},
    )
    obs = observations()
    metadata, probabilities, components, families, ranks = [], [], [], [], []
    for pair_id, (j, k) in enumerate(PAIRS):
        for case in M.CASES:
            p0, p1, f0, f1, rank = M.case_profiles(case, *obs["score"][pair_id])
            metadata.append(
                {"pair_id": pair_id, "focal": j, "companion": k, "case": case}
            )
            probabilities.append([p0, p1])
            components.append([M.disintegrate(p0), M.disintegrate(p1)])
            families.append([f0, f1])
            ranks.append(-1 if rank is None else rank)
    probabilities, families = np.asarray(probabilities), np.asarray(families)
    components = np.asarray(components)
    data_dir = out / "01"
    data_dir.mkdir()
    write(data_dir / "rows.json", metadata)
    np.savez_compressed(
        data_dir / "profiles.npz",
        probabilities=probabilities,
        components=components,
        families=families,
        ranks=np.asarray(ranks),
    )
    rows, fibers = [], []
    for stage in STAGES:
        results = []
        for i, meta in enumerate(metadata):
            pair_id = meta["pair_id"]
            f0, f1 = families[i]
            p0, p1 = probabilities[i]
            r = M.accounts(p0, p1, obs[stage][pair_id, f0], obs[stage][pair_id, f1])
            results.append(r)
            rows.append({**meta, "stage": stage, **row_summary(r)})
            if stage == "score" and meta["case"].startswith("same_scores"):
                fibers.append(
                    {
                        **meta,
                        "family": int(f0),
                        "constraint_rank": int(ranks[i]),
                        "probability_min": float(probabilities[i].min()),
                        "probability_gap_norm": float(np.linalg.norm(p1 - p0)),
                        "composition_gap_norm": float(
                            np.linalg.norm(components[i, 1, 0] - components[i, 0, 0])
                        ),
                        "expression_gap_norm": float(
                            np.linalg.norm(components[i, 1, 1] - components[i, 0, 1])
                        ),
                        **{
                            s + "_gap_norm": float(
                                np.linalg.norm((p1 - p0) @ obs[s][pair_id, f0])
                            )
                            for s in STAGES
                        },
                    }
                )
        np.savez_compressed(
            data_dir / (stage + ".npz"),
            laws=obs[stage],
            **{key: np.asarray([r[key] for r in results]) for key in results[0]},
        )
        if time.monotonic() - start > 60:
            raise TimeoutError("Bounded numerical collector exceeded 60 seconds")
    csv_write(data_dir / "account_rows.csv", rows)
    csv_write(data_dir / "case_stage_summary.csv", aggregate(rows))
    csv_write(data_dir / "same_score_fibers.csv", fibers)
    write(
        data_dir / "summary.json",
        {
            "packet": "HM69",
            "complete_account_rows": len(rows),
            "profile_cases": len(metadata),
            "same_score_fibers": len(fibers),
            "CPU_wall_seconds": time.monotonic() - start,
            "new_calls_fits_API_private_labels": [0, 0, 0, 0, 0],
            "maximum_sum_error": max(r["maximum_sum_error"] for r in rows),
            "original_scope_complete": False,
        },
    )
    write(
        out / "output_seal.json",
        {str(p.relative_to(out)): digest(p) for p in sorted(data_dir.iterdir())},
    )
    write(out / "checkpoint.json", {"status": "COLLECTED_HM69", "active_processes": []})
    print(json.dumps(read_json(data_dir / "summary.json")))


def independent_game(p0, p1, f0, f1):
    """Use direct conditional sums and permutation marginals, not production helpers."""
    profiles = (p0.reshape(2, 4), p1.reshape(2, 4))
    pis = [t[0] + t[1] for t in profiles]
    mus = [(t[1] - t[0]) / pi for t, pi in zip(profiles, pis, strict=True)]
    game = []
    for mask in range(8):
        pi, mu, law = (
            pis[int(bool(mask & 1))],
            mus[int(bool(mask & 2))],
            (f0, f1)[int(bool(mask & 4))],
        )
        terms = [
            pi[c] * (1 + x * mu[c]) / 2 * law[(x == 1) * 4 + c]
            for x in (-1, 1)
            for c in range(4)
        ]
        game.append(
            np.asarray(
                [math.fsum(float(t[q]) for t in terms) for q in range(f0.shape[1])]
            )
        )
    game = np.asarray(game)
    inter = []
    for mask in range(8):
        subsets = [s for s in range(8) if s & mask == s]
        inter.append(
            sum(((-1) ** (mask.bit_count() - s.bit_count())) * game[s] for s in subsets)
        )
    seq = []
    for order in ORDERS:
        values, mask = np.zeros((3, f0.shape[1])), 0
        for j in order:
            after = mask | (1 << j)
            values[j] = game[after] - game[mask]
            mask = after
        seq.append(values)
    seq = np.asarray(seq)
    return {
        "game": game,
        "interactions": np.asarray(inter),
        "sequences": seq,
        "shapley": seq.mean(0),
        "delta": game[7] - game[0],
    }


def verify(out):
    """Independently recompute every vector game, decomposition, table and fiber."""
    bound_check(out)
    for name, sha in read_json(out / "output_seal.json").items():
        if digest(out / name) != sha:
            raise ValueError("Frozen HM69 output drift")
    obs, data_dir = observations(), out / "01"
    metadata = read_json(data_dir / "rows.json")
    with np.load(data_dir / "profiles.npz", allow_pickle=False) as archive:
        profiles, components, families, ranks = (
            archive[k] for k in ("probabilities", "components", "families", "ranks")
        )
    checks, max_error = 0, 0.0

    def compare(actual, expected):
        nonlocal checks, max_error
        a, b = np.asarray(actual, float), np.asarray(expected, float)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("Finite aligned actual evidence required")
        error = float(np.max(abs(a - b)))
        if error > 2e-10:
            raise ValueError(f"Independent numerical mismatch {error}")
        checks += a.size
        max_error = max(max_error, error)

    expected_metadata = [
        {"pair_id": i, "focal": j, "companion": k, "case": case}
        for i, (j, k) in enumerate(PAIRS)
        for case in M.CASES
    ]
    if metadata != expected_metadata or profiles.shape != (240, 2, 8):
        raise ValueError("Incomplete or reordered profile protocol")
    for i, p in enumerate(profiles):
        if p.min() <= 0:
            raise ValueError("Profile left the positive probability simplex")
        compare(p.sum(1), np.ones(2))
        for end in (0, 1):
            table = p[end].reshape(2, 4)
            pi = table.sum(0)
            mu = (table[1] - table[0]) / pi
            compare(components[i, end], np.asarray([pi, mu]))
        case = metadata[i]["case"]
        y, g = np.array(list(itertools.product((-1, 1), repeat=2))).T
        pi0 = np.full(4, 0.25)
        pi1 = np.exp(0.6 * y - 0.2 * g + 0.15 * y * g)
        pi1 /= pi1.sum()
        mu0 = 0.2 * y - 0.1 * g + 0.15 * y * g
        mu1 = -0.1 * y + 0.2 * g - 0.35 * y * g
        if case.startswith("choice"):
            expected = [(pi0, mu0), (pi1, mu0)]
            compare(families[i], [int(case[-1]), int(case[-1])])
        elif case.startswith("expression"):
            expected = [(pi0, -0.5 * y * g), (pi0, 0.5 * y * g)]
            compare(families[i], [int(case[-1]), int(case[-1])])
        elif case == "law_only":
            expected = [(pi0, mu0), (pi0, mu0)]
            compare(families[i], [0, 1])
        elif case == "mixed":
            expected = [(pi0, mu0), (pi1, mu1)]
            compare(families[i], [0, 1])
        else:
            compare((p[0] + p[1]) / 2, np.full(8, 1 / 8))
            compare(np.max(abs(p[1] - p[0])), 0.1)
            compare(families[i], [int(case[-1]), int(case[-1])])
            expected = None
        if expected is not None:
            for end, (pi, mu) in enumerate(expected):
                compare(components[i, end], np.asarray([pi, mu]))
                compare(profiles[i, end], np.r_[pi * (1 - mu) / 2, pi * (1 + mu) / 2])
    expected_rows = []
    for stage in STAGES:
        with np.load(data_dir / (stage + ".npz"), allow_pickle=False) as archive:
            compare(archive["laws"], obs[stage])
            for i, meta in enumerate(metadata):
                pair_id, (f0, f1) = meta["pair_id"], families[i]
                r = independent_game(
                    *profiles[i], obs[stage][pair_id, f0], obs[stage][pair_id, f1]
                )
                for key, value in r.items():
                    compare(archive[key][i], value)
                compare(r["shapley"].sum(0), r["delta"])
                compare(r["sequences"].sum(1), np.tile(r["delta"], (6, 1)))
                case = meta["case"]
                pure = (
                    0
                    if case.startswith("choice")
                    else 1
                    if case.startswith("expression")
                    else 2
                    if case == "law_only"
                    else None
                )
                if pure is not None:
                    compare(
                        np.delete(r["shapley"], pure, axis=0),
                        np.zeros((2, r["delta"].size)),
                    )
                expected_rows.append({**meta, "stage": stage, **row_summary(r)})
    for name, expected in (
        ("account_rows.csv", expected_rows),
        ("case_stage_summary.csv", aggregate(expected_rows)),
    ):
        with (data_dir / name).open() as stream:
            actual = list(csv.DictReader(stream))
        if len(actual) != len(expected):
            raise ValueError("Incomplete result table")
        for a, b in zip(actual, expected, strict=True):
            for key, value in b.items():
                if isinstance(value, str):
                    if a[key] != value:
                        raise ValueError("Result table identity drift")
                else:
                    compare(float(a[key]), value)
    with (data_dir / "same_score_fibers.csv").open() as stream:
        fibers = list(csv.DictReader(stream))
    if len(fibers) != 60:
        raise ValueError("All pair/family score fibers required")
    for row in fibers:
        pair_id, family = int(row["pair_id"]), int(row["family"])
        i = pair_id * 8 + 6 + family
        p0, p1 = profiles[i]
        compare((p1 - p0) @ obs["score"][pair_id, family], np.zeros(6))
        matrix = np.column_stack((np.ones(8), obs["score"][pair_id, family]))
        s = np.linalg.svd(matrix, compute_uv=False)
        rank = int(np.sum(s > np.finfo(float).eps * max(matrix.shape) * s[0]))
        compare(ranks[i], rank)
        compare(float(row["constraint_rank"]), rank)
        compare(float(row["probability_min"]), min(p0.min(), p1.min()))
        compare(float(row["probability_gap_norm"]), np.linalg.norm(p1 - p0))
        for index, name in enumerate(("composition", "expression")):
            compare(
                float(row[name + "_gap_norm"]),
                np.linalg.norm(components[i, 1, index] - components[i, 0, index]),
            )
        for stage in STAGES:
            compare(
                float(row[stage + "_gap_norm"]),
                np.linalg.norm((p1 - p0) @ obs[stage][pair_id, family]),
            )
    summary = read_json(data_dir / "summary.json")
    for key, value in (
        ("complete_account_rows", 1440),
        ("profile_cases", 240),
        ("same_score_fibers", 60),
    ):
        compare(summary[key], value)
    compare(
        summary["maximum_sum_error"], max(r["maximum_sum_error"] for r in expected_rows)
    )
    result = {
        "status": "VERIFIED_FINITE_HM69_NOT_GLOBAL",
        "actual_scalar_checks": checks,
        "maximum_alternate_error": max_error,
        "new_neural_calls": 0,
        "same_agent_alternate_arithmetic_not_independent_replication": True,
        "original_scope_complete": False,
    }
    write(out / "verification.json", result)
    write(out / "checkpoint.json", {"status": result["status"], "active_processes": []})
    print(json.dumps(result))


def main():
    """Explicit finite stages; never retry, resume old queues or start a model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "verify"))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    try:
        {"prepare": prepare, "run": run, "verify": verify}[args.stage](args.out)
    except Exception as exc:
        if args.stage == "run" and args.out.exists():
            write(
                args.out / "checkpoint.json",
                {
                    "status": "STOPPED_HM69_ERROR",
                    "active_processes": [],
                    "error": type(exc).__name__ + ": " + str(exc),
                    "preserve_outputs_no_automatic_retry": True,
                },
            )
        raise


if __name__ == "__main__":
    main()
