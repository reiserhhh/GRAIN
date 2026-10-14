#!/usr/bin/env python3
"""Preserved-run adapter for a known terminal HM119 export-variable defect."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_expression_response as M
from grain_research.construction import math_history_routing as R
from grain_research.construction import math_recursive_collector as C
from scripts import run_grain_author_evidence_merge_hm102 as H
from scripts import run_grain_expression_response_hm119 as ORIGINAL


def append_batch(storage, keys, values, readings):
    """Append distinct output streams without shadowing a history-input buffer."""
    k, v, r = (np.asarray(x) for x in (keys, values, readings))
    if k.shape != v.shape or len(k) != len(r) or k.ndim != 5:
        raise ValueError("Complete matching constructed batch required")
    for name, array in (("k", k), ("v", v), ("readings", r)):
        storage[name].extend(array)


def cold(input_path, output):
    """Repeat only deterministic arithmetic within the original remaining cap."""
    import torch

    spec = H.load(input_path)
    for path, digest in spec["hashes"].items():
        if C.stream_sha(path) != digest:
            raise ValueError("Changed construction dependency")
    if output.exists() or "transformers" in sys.modules:
        raise ValueError("Existing output or forbidden model-object runtime")
    torch.set_num_threads(2)
    if spec["device"] == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Declared platform missing")
    begun = time.perf_counter()
    weights = C.load_weights(spec["weights"], spec["blocks"], spec["device"])
    with np.load(spec["coordinates"], allow_pickle=False) as z:
        coords = {name: z[name] for name in z.files}
    storage = {name: [] for name in ("k", "v", "readings")}
    compute = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(spec["contexts"]), spec["batch_size"]):
            if time.perf_counter() - compute >= spec["max_seconds"]:
                raise TimeoutError("Remaining original cap exhausted; no renewal")
            paths = spec["contexts"][start : start + spec["batch_size"]]
            history = {name: [] for name in ("k", "v")}
            for path in paths:
                with np.load(path, allow_pickle=False) as z:
                    for name, buffer in history.items():
                        buffer.append(z[name])
            k, v = M.compile_histories(
                {name: np.stack(buffer) for name, buffer in history.items()},
                coords["seeds"],
                coords["cos"],
                coords["sin"],
                np.asarray(spec["valid"])[start : start + len(paths)],
                weights,
                spec["config"],
                spec["device"],
            )
            read = np.array(
                [
                    [
                        R.read_current(kk, vv, spec["endpoint"], spec["reader"])
                        for kk, vv in zip(krow, vrow, strict=True)
                    ]
                    for krow, vrow in zip(k, v, strict=True)
                ]
            )
            append_batch(storage, k, v, read)
            print("HM119 export adapter", start + len(paths), flush=True)
    arrays = {name: np.stack(buffer) for name, buffer in storage.items()}
    ORIGINAL.save(output, **arrays)
    H.write(
        output.with_suffix(".json"),
        {
            "successful_current_only_executions": len(arrays["k"]) * 2,
            "known_failed_execution_count_before_repeat": 576,
            "history_model_calls_repeated": 0,
            "native_answer_calls": 0,
            "fits": 0,
            "transformers_loaded": "transformers" in sys.modules,
            "arithmetic_and_scientific_rules_changed": False,
            "original_cap": spec["original_cap"],
            "first_attempt_wall_upper_seconds": spec[
                "first_attempt_wall_upper_seconds"
            ],
            "remaining_execution_cap": spec["max_seconds"],
            "load_seconds": compute - begun,
            "execution_seconds": time.perf_counter() - compute,
            "input_sha256": C.stream_sha(input_path),
            "output_sha256": C.stream_sha(output),
        },
    )


def complete(out):
    """Authenticate the specific terminal defect, preserve it and seal its adapter."""
    ORIGINAL.check(out)
    failure = H.load(out / "02/process.json")
    empty = out / "02/constructed.npz"
    if (
        failure["returncode"] != 1
        or "all input arrays must have the same shape" not in failure["stderr"]
        or "HM119 cold 288" not in failure["stdout"]
        or empty.stat().st_size != 0
    ):
        raise ValueError("Not the known terminal export defect; do not retry")
    spec = H.load(out / "02/input.json")
    used = (
        max(0.0, empty.stat().st_mtime - (out / "02/input.json").stat().st_mtime) + 1.0
    )
    remaining = spec["max_seconds"] - used
    if remaining <= 0:
        raise TimeoutError("No original budget remains")
    folder = out / "02_corrected"
    folder.mkdir()
    inputs = {
        **spec,
        "original_cap": spec["max_seconds"],
        "first_attempt_wall_upper_seconds": used,
        "max_seconds": remaining,
    }
    inputs["hashes"] = {
        **spec["hashes"],
        str(Path(__file__).resolve()): H.sha(Path(__file__)),
        str(
            (ROOT / "tests/test_grain_expression_response_adapter.py").resolve()
        ): H.sha(ROOT / "tests/test_grain_expression_response_adapter.py"),
        **{
            str((out / name).resolve()): H.sha(out / name)
            for name in ("02/input.json", "02/process.json", "02/constructed.npz")
        },
    }
    H.write(folder / "input.json", inputs)
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "cold",
            "--output",
            str(out),
            "--input",
            str(folder / "input.json"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    H.write(
        folder / "process.json",
        {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        },
    )
    if result.returncode:
        raise RuntimeError("Adapter failed; preserve history, no retry")
    H.write(folder / "construction_seal.json", H.seals(out, folder))
    print(
        "HM119 deterministic export correction completed; original failed files retained",
        flush=True,
    )


def corrected_load(out):
    """Address only the new sealed output, without altering any frozen old code."""
    folder = out / "02_corrected"
    spec = H.load(folder / "input.json")
    for path, digest in spec["hashes"].items():
        if C.stream_sha(path) != digest:
            raise ValueError("Changed adapter/input/code dependency")
    H.verify_hashes(out, H.load(folder / "construction_seal.json"))
    rows = H.load(out / "00/rows.json")
    with np.load(folder / "constructed.npz", allow_pickle=False) as z:
        raw = {name: z[name] for name in z.files}
    n = np.empty((3, 2, 6, 2, 2, 2, 2, 2, 6))
    seen = set()
    for row, value in zip(rows, raw["readings"], strict=True):
        index = tuple(row[name] for name in ("variant", "wording", "family")) + tuple(
            (row[name] + 1) // 2 for name in ("condition", "polarity", "owner")
        )
        if index in seen:
            raise ValueError("Duplicate condition address")
        seen.add(index)
        n[index] = value
    if len(seen) != 288:
        raise ValueError("Incomplete constructed support")
    return rows, raw, n


def main():
    """Explicit correction/analysis actions; never restore the failed execution."""
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("cold", "compile", "analyze", "verify"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input", type=Path)
    args = parser.parse_args()
    if args.action == "cold":
        if args.input is None:
            parser.error("cold requires --input")
        cold(args.input, args.output / "02_corrected/constructed.npz")
    elif args.action == "compile":
        complete(args.output)
    else:
        ORIGINAL.load_constructed = corrected_load
        {"analyze": ORIGINAL.analyze, "verify": ORIGINAL.verify}[args.action](
            args.output
        )


if __name__ == "__main__":
    main()
