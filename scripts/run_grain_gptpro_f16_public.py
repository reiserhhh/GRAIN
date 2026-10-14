#!/usr/bin/env python3
"""Compile all six F16 routes and execute source-frozen public contract checks."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from grain_research.gptpro_foundation_v3 import f16_public as p

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/grain_gptpro_foundation_v3"
ROUTES = (
    ("numeric_T1", "t1_source_full_r1", "T1"),
    ("numeric_T2", "t2_cold_full_r1", "T2"),
    ("numeric_T3", "t3_application_full_r1", "T3"),
    ("text_T1", "t1_text_full_r1", "T1"),
    ("text_T2", "dynamic_text_full_r1", "T2"),
    ("text_T3", "dynamic_text_full_r1", "T3"),
)


def save(path, value):
    """Append strict checkpoints without replacing a previous experiment."""
    with Path(path).open("x") as stream:
        stream.write(p.json_bytes(value).decode() + "\n")


def lines(path):
    """Read an explicitly declared JSONL archive."""
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def difference(a, b):
    """Compare complete JSON objects including nulls, identities and finite values."""
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            raise ValueError("different object fields")
        return max((difference(a[k], b[k]) for k in a), default=0.0)
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            raise ValueError("different object lengths")
        return max((difference(x, y) for x, y in zip(a, b, strict=True)), default=0.0)
    numeric = (int, float, np.integer, np.floating)
    if (
        isinstance(a, numeric)
        and isinstance(b, numeric)
        and not isinstance(a, (bool, np.bool_))
        and not isinstance(b, (bool, np.bool_))
    ):
        if not np.isfinite([a, b]).all():
            raise ValueError("nonfinite comparison")
        return abs(a - b)
    if a != b:
        raise ValueError(f"non-numeric mismatch: {a!r} / {b!r}")
    return 0.0


def inputs_and_reference(folder, seed, bundle, smoke):
    """Select original declared authors without reading truth or source fitting."""
    if bundle["engine"].startswith("text"):
        raw = lines(folder / "application_inputs.jsonl")
        old = lines(folder / "application_cold_scores.jsonl")
        indexes = [0, len(raw) - 1] if smoke else list(range(len(raw)))
        if len(raw) != 200 or len(old) != 200:
            raise ValueError("missing original text application population")
        return [raw[i] for i in indexes], [old[i] for i in indexes], indexes
    manifest = json.loads((folder / "application_manifest.json").read_text())
    target = bundle["target_id"]
    path = BASE / (
        f"F04/source_full_r1/L_{seed}/application.jsonl"
        if target == "T1"
        else f"F05/source_full_r2/core_{seed}/application.jsonl"
    )
    expected = manifest.get(
        "application_sha256", manifest.get("input_hash", manifest.get("sha256"))
    )
    if p.file_hash(path) != expected:
        raise ValueError("original numeric application hash mismatch")
    groups = {}
    for row in lines(path):
        if row["role"] != "application":
            raise ValueError("wrong application role")
        groups.setdefault(row["author_id"], []).append(row)
    ids = sorted(groups)
    if ids != manifest["author_order"] or len(ids) != 200:
        raise ValueError("archived author identity/order mismatch")
    old = lines(folder / "cold_scores.jsonl")
    if target == "T3":
        old = [
            {
                "score": x["raw_score"],
                "variance": x["variance_components"]["total"],
                "reference_score": x["source_standardized_score"],
            }
            for x in old
        ]
    indexes = [0, len(ids) - 1]
    result = []
    for index in indexes:
        rows = groups[ids[index]][: 32 if target == "T1" else 48]
        if target == "T1":
            result.append(
                {
                    "world_contract": bundle["world_contract"],
                    "reader_hash": bundle["base_bundle"]["reader_hash"],
                    "records": rows,
                }
            )
        else:
            result.append(
                {
                    "world": "core",
                    "reader_hash": bundle["base_bundle"]["reader_hash"],
                    "target_time": 47,
                    "records": rows,
                }
            )
    return result, [old[i] for i in indexes], indexes


def renamed(request, name, extreme=False, bundle=None):
    """Change only observed author identity, optionally add a fixed stress value."""
    x = copy.deepcopy(request)
    data = x["data"]
    if "history" in data:
        data["author_id"] = name
        high = max(
            bundle["base_bundle"]["materials"],
            key=lambda k: bundle["base_bundle"]["materials"][k]["pole_readout"],
        )
        for op in data["history"]:
            if "observation" in op:
                op["observation"]["author_id"] = name
                if extreme:
                    op["observation"]["material_id"] = high
    else:
        for row in data["records"]:
            row["author_id"] = name
            if extreme:
                row["y"] += 10.0
    return x


def controls(requests, bundle):
    """Test actual single/batch/outsider invariance plus explicit absence."""
    first = p.cold_score(requests[0], bundle)
    clone = renamed(requests[0], "renamed_same_observations", bundle=bundle)
    extreme = renamed(requests[0], "extreme_outsider", True, bundle)
    batch = p.cold_scores([requests[0], requests[1], clone, extreme], bundle)
    err = max(difference(first, batch[0]), difference(first, batch[2]))
    if err > 1e-10:
        raise ValueError("single/batch/renamed author scores differ")
    missing = copy.deepcopy(requests[0])
    empty = copy.deepcopy(requests[0])
    key = "history" if "history" in missing["data"] else "records"
    missing["data"][key] = missing["data"][key][:-4]
    empty["data"][key] = []
    absence = p.cold_scores([missing, empty], bundle)
    if any(o["target_time"] != bundle["time_contract"]["target_time"] for o in absence):
        raise ValueError("requested clock moved after removing evidence")
    return {
        "inputs": [requests[0], requests[1], clone, extreme, missing, empty],
        "single": first,
        "batch": batch,
        "absence": absence,
        "max_difference": err,
    }


def rejection_checks(request, bundle):
    """Persist real exceptions, not literal success flags."""
    tests = []
    for field, value in (
        ("target_id", "T0"),
        ("unit", "new_unit"),
        ("reader_hash", "0" * 64),
        ("world_contract", "F09"),
        ("time_contract", {"unit": "days"}),
        ("condition_contract", {}),
        ("object_class", "reader_judgment"),
    ):
        wrong = copy.deepcopy(request)
        wrong["contract"][field] = value
        tests.append(("request_" + field, wrong, bundle))
    for field, value in (
        ("unit", "new_unit"),
        ("reader_hash", "0" * 64),
        ("source_hash", "0" * 64),
        ("reference_mu", 999),
        ("object_class", "reader_judgment"),
    ):
        wrong = copy.deepcopy(bundle)
        wrong[field] = value
        tests.append(("bundle_" + field, request, wrong))
    truth = copy.deepcopy(request)
    truth["data"]["truth"] = 0.0
    tests.append(("forbidden_truth", truth, bundle))
    out = []
    for name, raw, pack in tests:
        try:
            p.cold_score(raw, pack)
        except (ValueError, KeyError) as exc:
            out.append(
                {"case": name, "exception": type(exc).__name__, "message": str(exc)}
            )
        else:
            raise ValueError(f"invalid case accepted: {name}")
    return out


def main():
    """Freeze all source bundles and then run public loading in new processes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    seeds = [20260915] if args.smoke else [20260915, 20260916, 20260917]
    code = [
        Path(__file__),
        ROOT / "grain_research/gptpro_foundation_v3/f16_public.py",
        ROOT / "tests/test_grain_gptpro_f16_public.py",
        ROOT / "docs/GRAIN_F16_PUBLIC_CONTRACT_PROTOCOL_V1.md",
    ]
    save(
        args.output / "protocol_before_run.json",
        {
            "seeds": seeds,
            "smoke": args.smoke,
            "source_hashes": {
                f"{folder}/{seed}": p.file_hash(
                    BASE / f"F16/{folder}/{seed}/bundle.json"
                )
                for seed in seeds
                for _, folder, _ in ROUTES
            },
            "code_hashes": {str(x): p.file_hash(x) for x in code},
            "text_application": "all200 perseed/target;smoke firstlast",
            "numeric_application": "firstlast original200 IDs;full200source/personintegration",
            "scope": "public adapter regression;not new target validation or completed transport",
        },
    )
    snap = args.output / "source_snapshot"
    snap.mkdir()
    for path in code:
        (snap / path.name).write_bytes(path.read_bytes())
    ready = []
    for seed in seeds:
        for route, folder, target in ROUTES:
            source = BASE / f"F16/{folder}/{seed}/bundle.json"
            d = p.describe(json.loads(source.read_text()), target)
            brief = {
                k: d[k]
                for k in (
                    "target_id",
                    "object_class",
                    "unit",
                    "world_contract",
                    "observed_role",
                )
            }
            bundle = p.build_instrument(
                brief,
                {
                    "path": str(source),
                    "sha256": p.file_hash(source),
                    "role": "source_fitted",
                },
            )
            out = args.output / f"{route}_{seed}"
            out.mkdir()
            save(out / "brief.json", brief)
            save(out / "bundle.json", bundle)
            ready.append((seed, route, source.parent, out, bundle))
    summary = []
    for seed, route, folder, out, bundle in ready:
        raw, reference, indices = inputs_and_reference(folder, seed, bundle, args.smoke)
        requests = [{"contract": p.request_contract(bundle), "data": r} for r in raw]
        with (out / "inputs.jsonl").open("x") as stream:
            for r in requests:
                stream.write(p.json_bytes(r).decode() + "\n")
        warm = p.cold_scores(requests, bundle)
        save(out / "warm.json", warm)
        env = {
            **os.environ,
            "PYTHONPATH": str(ROOT),
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
        }
        with (out / "cold.log").open("x") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "grain_research.gptpro_foundation_v3.f16_public",
                    "--bundle",
                    str((out / "bundle.json").resolve()),
                    "--sha256",
                    p.file_hash(out / "bundle.json"),
                    "--input",
                    str((out / "inputs.jsonl").resolve()),
                    "--output",
                    str((out / "cold.jsonl").resolve()),
                ],
                cwd=out,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        cold = lines(out / "cold.jsonl")
        err = difference(warm, cold)
        if bundle["engine"].startswith("text") or bundle["engine"] == "numeric_t1":
            olderr = difference([o["engine_output"] for o in cold], reference)
        else:
            olderr = max(
                difference(
                    {k: o[k] for k in ("score", "variance", "reference_score")},
                    {k: r[k] for k in ("score", "variance", "reference_score")},
                )
                for o, r in zip(cold, reference, strict=True)
            )
        if max(err, olderr) > 1e-10:
            raise ValueError("cold/archive mismatch")
        check = controls(requests, bundle)
        save(out / "invariance.json", check)
        rejection = rejection_checks(requests[0], bundle)
        save(out / "rejections.json", rejection)
        row = {
            "seed": seed,
            "route": route,
            "objects": len(cold),
            "indices": indices,
            "cold_warm_max": err,
            "archive_max": olderr,
            "batch_max": check["max_difference"],
            "negative_cases": len(rejection),
        }
        save(out / "summary.json", row)
        summary.append(row)
        print(
            json.dumps(
                {
                    **{k: v for k, v in row.items() if k != "indices"},
                    "seconds": time.monotonic() - started,
                }
            ),
            flush=True,
        )
    save(
        args.output / "summary.json",
        {
            "status": "PUBLIC_CONTRACT_EXECUTED_TRANSPORT_REMAINS",
            "cells": summary,
            "objects": sum(x["objects"] for x in summary),
            "seconds": time.monotonic() - started,
            "scope": "same-source executablecontract;not new psychological/transport accuracy;no sourcefit/neuralcalls",
        },
    )
    save(
        args.output / "output_manifest.json",
        {
            str(x.relative_to(args.output)): p.file_hash(x)
            for x in sorted(args.output.rglob("*"))
            if x.is_file()
        },
    )


if __name__ == "__main__":
    main()
