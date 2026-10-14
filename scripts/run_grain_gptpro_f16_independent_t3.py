#!/usr/bin/env python3
"""One-worker T3 source construction and hash-authenticated cold interface."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import os
import sys
import time
from pathlib import Path

# Set before importing NumPy/SciPy, including when the parent exported a larger value.
for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ[variable] = "1"
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "grain_research/gptpro_foundation_v3/f16_independent_t3.py"
SPEC = importlib.util.spec_from_file_location("independent_t3", MODULE_PATH)
t3 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(t3)
OUTPUT_ROOT = ROOT / "results/grain_gptpro_foundation_v3/F16/independent_t3_v1"
HANDBOOK = ROOT / "docs/GRAIN_F16_INDEPENDENT_DYNAMIC_HANDBOOK_V1.md"
BRIEF = ROOT / "configs/grain_f16_dynamic_briefs_v1.json"


def write_json(path, value):
    path = Path(path).resolve()
    if not path.is_relative_to(OUTPUT_ROOT):
        raise t3.ContractError("output must be inside independent_t3_v1")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = t3.strict_json_bytes(value)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    digest = hashlib.sha256(payload).hexdigest()
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar_temporary = sidecar.with_suffix(sidecar.suffix + ".tmp")
    sidecar_temporary.write_text(digest + "\n", encoding="ascii")
    os.replace(sidecar_temporary, sidecar)
    return digest


def read_json(path, audit):
    payload = path.read_bytes()
    audit.append(
        {"path": str(path.resolve()), "sha256": hashlib.sha256(payload).hexdigest()}
    )
    return json.loads(payload)


def source_manifest(seed, brief, audit):
    source_path = (
        ROOT
        / f"results/grain_gptpro_foundation_v3/F05/source_full_r2/core_{seed}/source.jsonl"
    )
    payload = source_path.read_bytes()
    source_digest = hashlib.sha256(payload).hexdigest()
    audit.append({"path": str(source_path), "sha256": source_digest})
    records = {}
    for line in payload.splitlines():
        row = json.loads(line)
        if row.get("role") != "source" or row.get("world") != brief["world"]:
            raise t3.ContractError(
                "non-source/wrong-world row in permitted source file"
            )
        rows = records.setdefault(row["author_id"], [])
        rows.append(row)
    for author, rows in records.items():
        # Never sort malformed input into apparent validity; validate the full order first.
        times = [row["t"] for row in rows]
        if any(type(value) is not int for value in times) or any(
            right <= left for left, right in itertools.pairwise(times)
        ):
            raise t3.ContractError("unordered/noninteger source times")
        if len({row["event_id"] for row in rows}) != len(rows):
            raise t3.ContractError("duplicate source event")
        records[author] = rows[: brief["reference_prefix"]]
    ensemble, fit_hashes = [], []
    for bootstrap in range(brief["source_bootstraps"]):
        path = (
            ROOT
            / f"results/grain_gptpro_foundation_v3/F13/dynamic_source_full_r1/core_{seed}/fit_{bootstrap:03d}.json"
        )
        value = read_json(path, audit)
        model = {
            key: value.get(key)
            for key in ("status", "seed", "world", "bootstrap", "source_ids")
        }
        if model["status"] == "FITTED":
            fit = value.get("fit", {})
            model["fit"] = {
                "prior": fit.get("prior"),
                "log_sigma": fit.get("log_sigma"),
            }
        ensemble.append(model)
        fit_hashes.append(audit[-1])
    provenance = {
        "source": {"path": str(source_path), "sha256": source_digest},
        "fits": fit_hashes,
    }
    return {
        "seed": seed,
        "world": brief["world"],
        "source_prefixes": records,
        "ensemble": ensemble,
        "provenance": provenance,
        "source_hash": t3.object_hash(provenance),
    }


def load_checkpoint(path, expected, audit):
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise t3.ContractError("checkpoint externally expected SHA256 mismatch")
    audit.append({"path": str(path.resolve()), "sha256": actual})
    return json.loads(payload)


def synthetic_fixture():
    import numpy as np

    brief = {
        "world": "core",
        "seeds": [20260915],
        "source_authors": 3,
        "reference_prefix": 8,
        "application_prefix": 8,
        "source_bootstraps": 2,
        "personal_draws": 200,
        "T3": {
            "target_id": "T3",
            "future_conditions": [1, -1, 1, -1],
            "unit": "future observation units",
        },
    }
    rng = np.random.default_rng(163)
    records = {}
    for index in range(3):
        author = f"synthetic_{index}"
        records[author] = [
            {
                "author_id": author,
                "role": "source",
                "world": "core",
                "episode_id": "synthetic",
                "event_id": f"{author}:{t}",
                "t": t,
                "z": (-1.0) ** t,
                "y": float(index - 1 + 0.3 * (-1) ** t + rng.normal(0, 0.3)),
            }
            for t in range(8)
        ]
    ensemble = [
        {
            "status": "FITTED",
            "seed": 20260915,
            "world": "core",
            "bootstrap": b,
            "source_ids": sorted(records),
            "fit": {
                "prior": {
                    "mu": [0, 0.3, 0.4],
                    "Sigma": [[1, 0, 0], [0, 0.5, 0], [0, 0, 0.3]],
                },
                "log_sigma": [-0.6 + b * 0.03, -0.8],
            },
        }
        for b in range(2)
    ]
    provenance = {"kind": "tiny synthetic numerical test, not production reference"}
    manifest = {
        "seed": 20260915,
        "world": "core",
        "source_prefixes": records,
        "ensemble": ensemble,
        "provenance": provenance,
        "source_hash": t3.object_hash(provenance),
    }
    return brief, manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-source")
    build.add_argument(
        "--seed", type=int, choices=[20260915, 20260916, 20260917], required=True
    )
    build.add_argument("--max-new-authors", type=int)
    build.add_argument("--resume-sha256")
    commands.add_parser("synthetic-warm")
    cold = commands.add_parser("cold")
    cold.add_argument("--bundle", type=Path, required=True)
    cold.add_argument("--expected-sha256", required=True)
    cold.add_argument("--expected-source-hash", required=True)
    cold.add_argument("--input", type=Path, required=True)
    cold.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    audit = [
        {
            "path": str(Path(__file__).resolve()),
            "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        {"path": str(MODULE_PATH), "sha256": t3.code_hash()},
    ]
    started = time.monotonic()
    if args.command == "cold":
        bundle = t3.load_bundle(
            args.bundle, args.expected_sha256, args.expected_source_hash
        )
        audit.append(
            {"path": str(args.bundle.resolve()), "sha256": args.expected_sha256}
        )
        value = read_json(args.input, audit)
        score = t3.cold_score(value, bundle)
        digest = write_json(args.output, score)
        write_json(args.output.with_suffix(".reads.json"), audit)
        print(
            json.dumps(
                {
                    "supported": score["supported"],
                    "raw_score": score["raw_score"],
                    "sha256": digest,
                    "output": str(args.output),
                }
            ),
            flush=True,
        )
        return 0 if score["supported"] else 2
    if args.command == "synthetic-warm":
        brief, manifest = synthetic_fixture()
        directory = OUTPUT_ROOT / "synthetic_smoke"
        bundle, checkpoint = t3.build_instrument(brief, manifest)
        digest = write_json(directory / "bundle.json", bundle)
        write_json(directory / "checkpoint.json", checkpoint)
        value = {
            "world": "core",
            "reader_hash": t3.READER_HASH,
            "prefix": manifest["source_prefixes"]["synthetic_0"],
        }
        write_json(directory / "input.json", value)
        score = t3.cold_score(value, bundle)
        write_json(directory / "warm_score.json", score)
        write_json(directory / "reads.json", audit)
        print(
            json.dumps(
                {
                    "kind": "SYNTHETIC_TEST_ONLY",
                    "bundle_sha256": digest,
                    "source_hash": bundle["source_hash"],
                    "raw_score": score["raw_score"],
                    "reference_mu": bundle["reference_mu"],
                    "reference_sigma": bundle["reference_sigma"],
                }
            ),
            flush=True,
        )
        return 0
    directory = OUTPUT_ROOT / f"core_{args.seed}"
    checkpoint_path = directory / "checkpoint.json"
    if checkpoint_path.exists() and not args.resume_sha256:
        raise t3.ContractError(
            "checkpoint exists; supply its expected SHA256 to resume, never silently overwrite"
        )
    audit.append(
        {
            "path": str(HANDBOOK),
            "sha256": hashlib.sha256(HANDBOOK.read_bytes()).hexdigest(),
        }
    )
    brief = read_json(BRIEF, audit)
    checkpoint = (
        load_checkpoint(checkpoint_path, args.resume_sha256, audit)
        if args.resume_sha256
        else None
    )
    try:
        manifest = source_manifest(args.seed, brief, audit)
    finally:
        write_json(directory / "read_log.json", audit)
    write_json(
        directory / "source_manifest_summary.json",
        {
            "seed": args.seed,
            "source_hash": manifest["source_hash"],
            "source_authors": len(manifest["source_prefixes"]),
            "source_prefixes_hash": t3.object_hash(manifest["source_prefixes"]),
            "fit_count": len(manifest["ensemble"]),
            "status_counts": {
                status: sum(model["status"] == status for model in manifest["ensemble"])
                for status in sorted(
                    {model["status"] for model in manifest["ensemble"]}
                )
            },
            "provenance": manifest["provenance"],
        },
    )

    def save_checkpoint(value):
        digest = write_json(checkpoint_path, value)
        print(
            json.dumps(
                {
                    "seed": args.seed,
                    "completed_authors": value["completed_authors"],
                    "required_authors": value["required_authors"],
                    "status": value["construction_status"],
                    "checkpoint_sha256": digest,
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                }
            ),
            flush=True,
        )

    bundle, checkpoint = t3.build_instrument(
        brief,
        manifest,
        checkpoint=checkpoint,
        checkpoint_callback=save_checkpoint,
        max_new_authors=args.max_new_authors,
    )
    name = (
        "bundle.json"
        if bundle["construction_status"] == "SOURCE_REFERENCE_FROZEN"
        else "draft_bundle.json"
    )
    digest = write_json(directory / name, bundle)
    write_json(
        directory / "run_summary.json",
        {
            "seed": args.seed,
            "status": bundle["construction_status"],
            "completed_authors": checkpoint["completed_authors"],
            "required_authors": bundle["source_authors"],
            "bundle_path": str(directory / name),
            "bundle_sha256": digest,
            "elapsed_seconds": time.monotonic() - started,
            "numerical_workers": 1,
            "reference_mu": bundle["reference_mu"],
            "reference_sigma": bundle["reference_sigma"],
            "scorer_code_hash": bundle["scorer_code_hash"],
            "source_hash": bundle["source_hash"],
        },
    )
    return 2 if bundle["construction_status"] == "UNSUPPORTED_SOURCE_REFERENCE" else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (t3.ContractError, OSError, KeyError, json.JSONDecodeError) as error:
        print(f"T3_CONSTRUCTION_ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
