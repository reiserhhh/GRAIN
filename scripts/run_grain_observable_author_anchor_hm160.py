#!/usr/bin/env python3
"""Run a bounded source-only observable author construction, not a grid."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_observable_author_anchor as M

SPEC = ROOT / "configs/grain_observable_author_anchor_hm160.json"
CODE = (
    "configs/grain_observable_author_anchor_hm160.json",
    "docs/GRAIN_OBSERVABLE_AUTHOR_ANCHOR_PLAN_HM160.md",
    "grain_research/construction/math_observable_author_anchor.py",
    "scripts/run_grain_observable_author_anchor_hm160.py",
    "scripts/verify_grain_observable_author_anchor_hm160.py",
    "tests/test_grain_observable_author_anchor.py",
)


def read(path):
    """Read one named local JSON artifact."""
    return json.loads(Path(path).read_text())


def sha(path):
    """Hash frozen scientific dependencies and numerical artifacts."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    """Create an artifact exclusively; do not overwrite failed/history runs."""
    with Path(path).open("x") as stream:
        json.dump(
            value,
            stream,
            indent=2,
            allow_nan=False,
            default=lambda x: str(x) if isinstance(x, Fraction) else x,
        )
        stream.write("\n")


def archive(path, **arrays):
    """Create a numerical archive without overwrite."""
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def csv_table(path, rows):
    """Save every frozen condition/seed rather than selecting a winner."""
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare(out):
    """Freeze plan, implementation and a historical public vector alphabet."""
    if out.exists():
        raise FileExistsError("Existing results are retained")
    config = read(SPEC)
    source = ROOT / config["alphabet"]
    with np.load(source, allow_pickle=False) as data:
        alphabet = data["menus"][:4].copy()
    M.decode_vectors(alphabet, alphabet)
    out.mkdir(parents=True)
    archive(out / "alphabet.npz", alphabet=alphabet)
    write(
        out / "manifest.json",
        {
            "packet": "HM160",
            "code": {str(ROOT / p): sha(ROOT / p) for p in CODE},
            "inputs": {str(source): sha(source)},
            "alphabet_sha256": sha(out / "alphabet.npz"),
            "scope": "Exact dictionary observations, new simulated records, not natural psychological anchors",
        },
    )


def check(out):
    """Read-only seal check before run, audit or cold scoring."""
    manifest = read(out / "manifest.json")
    for path, expected in (manifest["code"] | manifest["inputs"]).items():
        if sha(path) != expected:
            raise ValueError("Changed frozen dependency: " + path)
    if sha(out / "alphabet.npz") != manifest["alphabet_sha256"]:
        raise ValueError("Changed alphabet")


def exact(out, alphabet):
    """Keep exact source tables and anchoring/gauge witnesses separate."""
    main = M.exact_table()
    bundle = M.compile_bundle(main, "randomized_condition")
    profiles = [M.cold_profile(bundle, v, alphabet) for v in alphabet]
    p = np.asarray(bundle["probabilities"], float)
    reference = np.asarray(bundle["reference"], float)
    gain = np.asarray(bundle["history_mass"]) @ ((p - reference) ** 2)
    null = M.compile_bundle(M.exact_table(null=True), "randomized_condition")
    alias = {}
    for world in (0, 1):
        alias[str(world)] = {}
        for regime in M.REGIMES:
            table = M.selection_alias_table(world, regime)
            program = M.compile_bundle(table, regime)
            alias[str(world)][regime] = {
                "table": table,
                "bundle": program,
                "profiles": [M.cold_profile(program, v, alphabet) for v in alphabet],
            }
    write(
        out / "exact.json",
        {
            "main_joint": main,
            "main_bundle": bundle,
            "main_profiles": profiles,
            "latent_gauge_witness": M.gauge_witness(),
            "exact_endpoint_brier_gain": gain.tolist(),
            "null_bundle": null,
            "selection_alias": alias,
            "interpretation": "Behavior-anchored function, not latent trait labels or psychological validity",
        },
    )


def draw(rng, n):
    """Draw declared author/history parameters for generation/evaluation only."""
    p = rng.choice([0.05, 0.45, 0.95], size=n, p=[0.5, 0.3, 0.2])
    u_c, u_r = rng.random((2, n))
    c = (u_c < p).astype(np.int64)
    r = (u_r < p).astype(np.int64)
    address = 2 * c + (c ^ r)
    transformed = p**2
    inverse = np.sqrt(transformed)
    readdress = 2 * (u_c < inverse).astype(np.int64) + (
        (u_c < inverse) ^ (u_r < inverse)
    ).astype(np.int64)
    if not np.array_equal(address, readdress):
        raise ValueError("Coordinate replay changed observed histories")
    return p, address, inverse


def run(out):
    """Construct source-only bundles and score independent new vector histories."""
    check(out)
    if (out / "exact.json").exists() or any(out.glob("seed_*")):
        raise FileExistsError("Do not restart a complete or partial science run")
    start = time.monotonic()
    spec = read(SPEC)
    with np.load(out / "alphabet.npz", allow_pickle=False) as data:
        alphabet = data["alphabet"]
    exact(out, alphabet)
    rows, profiles = [], []
    for seed in spec["seeds"]:
        directory = out / f"seed_{seed}"
        directory.mkdir()
        rng = np.random.default_rng(seed)
        n = spec["source_authors"]
        p, h, inverse = draw(rng, n)
        a = rng.integers(0, 2, size=n)
        rate = np.where(a == 0, p, p * p)
        uniform = rng.random(n)
        y = (uniform < rate).astype(np.int64)
        y_replayed = (uniform < np.where(a == 0, inverse, inverse * inverse)).astype(
            np.int64
        )
        if not np.array_equal(y, y_replayed):
            raise ValueError("Latent coordinate replay changed actual endpoints")
        endpoint = 2 * a + (a ^ y)
        ids = seed * 100000 + np.arange(n, dtype=np.int64)
        archive(
            directory / "source_observed.npz",
            author_id=ids,
            history=alphabet[h],
            endpoint=alphabet[endpoint],
        )
        archive(directory / "source_evaluation_only.npz", p=p, lambda_squared=p * p)
        # Constructor inputs stop here: only the observed source file fields.
        counts = M.observed_table(ids, alphabet[h], alphabet[endpoint], alphabet)
        program = M.compile_bundle(counts, spec["regime"])
        write(directory / "source_bundle.json", program)
        archive(directory / "source_counts.npz", counts=counts)

        app = spec["application_authors"]
        p_app, h_app, _ = draw(rng, app)
        u_app = rng.random((app, 2))
        observed_y = np.column_stack(
            (u_app[:, 0] < p_app, u_app[:, 1] < p_app * p_app)
        ).astype(np.int64)
        app_ids = seed * 100000 + n + np.arange(app, dtype=np.int64)
        endpoint_addresses = 2 * np.arange(2)[None, :] + (
            np.arange(2)[None, :] ^ observed_y
        )
        archive(
            directory / "application_input.npz",
            author_id=app_ids,
            history=alphabet[h_app],
        )
        archive(
            directory / "application_evaluation_only.npz",
            p=p_app,
            outcomes=observed_y,
            endpoints=alphabet[endpoint_addresses],
        )
        predictions, local_profiles = M.score_vectors(
            program, alphabet[h_app], alphabet
        )
        archive(directory / "cold_predictions.npz", probabilities=predictions)
        write(directory / "profiles.json", local_profiles)
        for h_index, profile in enumerate(local_profiles):
            for q, target in enumerate(profile["targets"]):
                profiles.append(
                    {
                        "seed": seed,
                        "history": h_index,
                        "condition": q,
                        "probability": target["probability"],
                        "reference": target["reference"],
                        "relative": target["relative"],
                    }
                )
        for q in range(2):
            loss = (predictions[:, q] - observed_y[:, q]) ** 2
            baseline = (program["reference"][q] - observed_y[:, q]) ** 2
            rows.append(
                {
                    "seed": seed,
                    "condition": q,
                    "application_authors": app,
                    "brier": float(loss.mean()),
                    "fixed_reference_brier": float(baseline.mean()),
                    "brier_gain": float((baseline - loss).mean()),
                    "meaning": "Registered future expression; not personality accuracy",
                }
            )
    csv_table(out / "profiles.csv", profiles)
    csv_table(out / "application_endpoint_checks.csv", rows)
    write(
        out / "summary.json",
        {
            "status": "FINITE_OBSERVABLE_AUTHOR_ANCHOR_COMPLETE_NOT_GLOBAL",
            "source_authors": len(spec["seeds"]) * spec["source_authors"],
            "new_application_authors": len(spec["seeds"]) * spec["application_authors"],
            "finite_vector_events": len(spec["seeds"])
            * (2 * spec["source_authors"] + 3 * spec["application_authors"]),
            "actual_neural_calls": 0,
            "latent_fields_used_by_constructor": False,
            "coordinate_record_replay_differences": 0,
            "model_selection": False,
            "endpoint_checks": rows,
            "wall_seconds": time.monotonic() - start,
            "scope": "Simulated observed source/applications, exact vector alphabet, no human trait claims",
            "outputs": {
                str(p.relative_to(out)): sha(p)
                for p in sorted(out.rglob("*"))
                if p.is_file() and p.name != "manifest.json"
            },
        },
    )


def main():
    """Expose prepare/run and an outcome-free cold vector scoring CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run", "score"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--history", type=Path)
    parser.add_argument("--score-out", type=Path)
    args = parser.parse_args()
    out = args.output or ROOT / read(SPEC)["output"]
    if args.action == "prepare":
        prepare(out)
    elif args.action == "run":
        run(out)
    else:
        if not all((args.bundle, args.history, args.score_out)):
            parser.error("score requires bundle/history/score-out")
        check(out)
        with np.load(out / "alphabet.npz", allow_pickle=False) as data:
            alphabet = data["alphabet"]
        history = read(args.history)
        if set(history) != {"history_vector"}:
            raise ValueError("Cold CLI history excludes outcomes and latent labels")
        write(
            args.score_out,
            M.cold_profile(read(args.bundle), history["history_vector"], alphabet),
        )


if __name__ == "__main__":
    main()
