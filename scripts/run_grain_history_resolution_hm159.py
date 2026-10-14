#!/usr/bin/env python3
"""Freeze and execute source-design author construction, with no reader fitting."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_history_resolution as M

SPEC = ROOT / "configs/grain_history_resolution_hm159.json"
CODE = (
    "configs/grain_history_resolution_hm159.json",
    "docs/GRAIN_HISTORY_RESOLUTION_PLAN_HM159.md",
    "grain_research/construction/math_history_resolution.py",
    "scripts/run_grain_history_resolution_hm159.py",
    "scripts/verify_grain_history_resolution_hm159.py",
    "tests/test_grain_history_resolution.py",
)


def read(path):
    """Read an explicitly named local JSON artifact."""
    return json.loads(Path(path).read_text())


def sha(path):
    """Hash an artifact without changing it."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    """Create a result exclusively, retaining old and failed attempts."""
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def archive(path, **arrays):
    """Save a new numerical archive without overwrite."""
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def table(path, rows):
    """Save all frozen cases, not a selected winner."""
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare(out):
    """Freeze four public vector prototypes and the prospective dependencies."""
    if out.exists():
        raise FileExistsError("Existing/failed results are retained")
    spec = read(SPEC)
    source = ROOT / spec["source_dictionary"]
    if sha(source) != read(ROOT / spec["source_manifest"])["dictionary_sha256"]:
        raise ValueError("Changed historical dictionary")
    with np.load(source, allow_pickle=False) as data:
        alphabet = data["menus"][:4].copy()
    if alphabet.shape != (4, 15) or len(np.unique(alphabet, axis=0)) != 4:
        raise ValueError("Four distinct 15-coordinate prototypes required")
    out.mkdir(parents=True)
    archive(out / "alphabet.npz", alphabet=alphabet)
    write(
        out / "manifest.json",
        {
            "packet": "HM159",
            "code": {str(ROOT / p): sha(ROOT / p) for p in CODE},
            "inputs": {str(source): sha(source)},
            "alphabet_sha256": sha(out / "alphabet.npz"),
            "scope": "Declared vector-event world, not natural-personality labels or native teacher targets",
        },
    )


def check(out):
    """Check source/code provenance before any numerical execution."""
    manifest = read(out / "manifest.json")
    for path, digest in (manifest["code"] | manifest["inputs"]).items():
        if sha(path) != digest:
            raise ValueError("Changed prospective dependency: " + path)
    if sha(out / "alphabet.npz") != manifest["alphabet_sha256"]:
        raise ValueError("Changed finite alphabet")


def exact(out):
    """Compute exact aliases, functional construction and finite witness bounds."""
    support, even, odd = M.matched_populations(4)
    worlds = {"even": even, "odd": odd}
    fingerprints = {}
    profiles, bounds = [], []
    for world in ("even", "odd", "beta_2_3"):
        fingerprints[world] = {}
        for n in (3, 4, 5, 6):
            f = (
                M.beta_fingerprint(2, 3, n)
                if world == "beta_2_3"
                else M.exact_fingerprint(support, worlds[world], n)
            )
            fingerprints[world][str(n)] = list(map(str, f))
            for k in range(5):
                result = M.cold_profile(
                    {"source_trials": n, "frequencies": list(map(float, f))},
                    {"history_trials": 4, "successes": k},
                )
                profiles.append(
                    {"world": world, "source_trials": n, "successes": k, **result}
                )
    alias_rows = []
    for design, n, erased in (
        ("fixed4", 4, False),
        ("fixed5", 5, False),
        ("joint3", 6, False),
        ("choice_only3", 3, False),
        ("choice_erased3", 3, True),
    ):
        left = M.exact_fingerprint(support, even, n, erased)
        right = M.exact_fingerprint(support, odd, n, erased)
        tv = sum(abs(a - b) for a, b in zip(left, right, strict=True)) / 2
        alias_rows.append(
            {
                "design": design,
                "effective_trials": n,
                "TV_exact": str(tv),
                "TV": float(tv),
                "equal": left == right,
            }
        )
    a = M.binomial_matrix(support, 4)
    f = np.array(M.exact_fingerprint(support, even, 4), float)
    for k in range(5):
        interval, witnesses = M.finite_ratio_bounds(
            a, f, a[k] * np.array(support, float), a[k]
        )
        bounds.append(
            {
                "successes": k,
                "lower": interval[0],
                "upper": interval[1],
                "width": interval[1] - interval[0],
            }
        )
        archive(out / f"finite_bound_{k}.npz", witnesses=np.stack(witnesses))
    tail_interval, tail_witnesses = M.finite_ratio_bounds(
        a, f, np.array([p > 0.5 for p in support], float), np.ones(len(support))
    )
    archive(out / "finite_tail_bounds.npz", witnesses=np.stack(tail_witnesses))
    write(
        out / "exact.json",
        {
            "support": list(map(str, support)),
            "even_mass": list(map(str, even)),
            "odd_mass": list(map(str, odd)),
            "fingerprints": fingerprints,
            "alias_designs": alias_rows,
            "finite_grid_tail_interval": tail_interval,
            "scope": "LP bounds only on six-atom witness support; polynomial compiler valid throughout [0,1]",
        },
    )
    table(out / "exact_profiles.csv", profiles)
    table(out / "source_design_aliases.csv", alias_rows)
    table(out / "finite_author_bounds.csv", bounds)


def decode_events(indices, alphabet):
    """Emit exact finite vectors and decode all new events in bounded chunks."""
    flat = indices.ravel()
    decoded = np.empty_like(flat)
    for start in range(0, len(flat), 10000):
        block = flat[start : start + 10000]
        decoded[start : start + len(block)] = M.decode_vectors(
            alphabet[block], alphabet
        )
    return decoded.reshape(indices.shape)


def run(out):
    """Generate new source events, then apply only observed-frequency bundles."""
    start = time.monotonic()
    check(out)
    spec = read(SPEC)
    exact(out)
    with np.load(out / "alphabet.npz", allow_pickle=False) as data:
        alphabet = data["alphabet"]
    support, even, odd = M.matched_populations(4)
    p_grid = np.array(support, float)
    rows, total_events, total_authors = [], 0, 0
    for world_index, world in enumerate(spec["worlds"]):
        for seed in spec["seeds"]:
            for count in spec["source_author_counts"]:
                if time.monotonic() - start > spec["max_CPU_wall_seconds"]:
                    raise TimeoutError("Finite CPU cap exhausted; no automatic renewal")
                directory = out / f"{world}_{seed}_{count}"
                directory.mkdir()
                rng = np.random.default_rng(
                    np.random.SeedSequence([seed, world_index, count])
                )
                if world == "beta_2_3":
                    latent = rng.beta(2, 3, count)
                else:
                    mass = np.array(even if world == "even" else odd, float)
                    latent = rng.choice(p_grid, count, p=mass)
                fixed_c = np.broadcast_to(np.arange(5, dtype=np.int8) % 2, (count, 5))
                fixed_r = (rng.random((count, 5)) < latent[:, None]).astype(np.int8)
                fixed_y = fixed_c ^ fixed_r
                free_c = (rng.random((count, 3)) < latent[:, None]).astype(np.int8)
                free_r = (rng.random((count, 3)) < latent[:, None]).astype(np.int8)
                free_y = free_c ^ free_r
                fixed = decode_events(2 * fixed_c + fixed_y, alphabet)
                free = decode_events(2 * free_c + free_y, alphabet)
                archive(directory / "observed_events.npz", fixed=fixed, free=free)
                archive(directory / "oracle_evaluation_only.npz", p=latent)
                fixed_response = (fixed // 2) ^ (fixed % 2)
                free_choice = free // 2
                free_response = free_choice ^ (free % 2)
                counts = {
                    "fixed4": (4, fixed_response[:, :4].sum(axis=1)),
                    "fixed5": (5, fixed_response.sum(axis=1)),
                    "joint3": (6, (free_choice + free_response).sum(axis=1)),
                    "choice_only3": (3, free_choice.sum(axis=1)),
                }
                archive(
                    directory / "choice_erased_counts.npz",
                    counts=(free % 2).sum(axis=1),
                )
                for design, (n, observed_count) in counts.items():
                    bundle = {
                        "source_trials": n,
                        "frequencies": (
                            np.bincount(observed_count, minlength=n + 1) / count
                        ).tolist(),
                    }
                    write(directory / f"{design}_bundle.json", bundle)
                    for k in range(5):
                        result = M.cold_profile(
                            bundle, {"history_trials": 4, "successes": k}
                        )
                        rows.append(
                            {
                                "world": world,
                                "seed": seed,
                                "source_authors": count,
                                "design": design,
                                "successes": k,
                                **result,
                            }
                        )
                total_authors += count
                total_events += 8 * count
    table(out / "sampled_profiles.csv", rows)
    duplicate = {
        "event_id": "original",
        "condition": 0,
        "response": 1,
        "qualification": "self_actual",
    }
    write(
        out / "duplicate_control.json",
        {
            "report_copies": 10,
            "distinct_events": M.count_distinct_events([duplicate] * 10),
            "assumed_extra_independence": False,
        },
    )
    outputs = {
        str(p.relative_to(out)): sha(p)
        for p in out.rglob("*")
        if p.is_file() and p.name != "manifest.json"
    }
    summary = {
        "packet": "HM159",
        "status": "FINITE_SOURCE_DESIGN_FUNCTIONAL_CONSTRUCTION_COMPLETE_NOT_GLOBAL",
        "CPU_seconds": time.monotonic() - start,
        "source_author_instances": total_authors,
        "new_vector_events": total_events,
        "source_runs": len(spec["worlds"])
        * len(spec["seeds"])
        * len(spec["source_author_counts"]),
        "sampled_profiles": len(rows),
        "inadmissible_sample_profiles": sum(
            r["status"] == "INADMISSIBLE_EMPIRICAL_MOMENTS" for r in rows
        ),
        "decoder_forwards": 0,
        "native_teacher_fit": False,
        "fixed_latent_class_count": False,
        "original_scope_complete": False,
        "outputs": outputs,
    }
    write(out / "summary.json", summary)
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "outputs"}, indent=2
        )
    )


def main():
    """Expose frozen prepare/run and independently usable cold scoring."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "score"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--source-bundle", type=Path)
    parser.add_argument("--history", type=Path)
    parser.add_argument("--score-out", type=Path)
    args = parser.parse_args()
    if args.mode == "score":
        if not all((args.source_bundle, args.history, args.score_out)):
            parser.error("Cold score requires source-bundle, history and score-out")
        write(
            args.score_out, M.cold_profile(read(args.source_bundle), read(args.history))
        )
        return
    out = args.out or ROOT / read(SPEC)["out"]
    with threadpool_limits(limits=1):
        (prepare if args.mode == "prepare" else run)(out)


if __name__ == "__main__":
    main()
