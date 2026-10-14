#!/usr/bin/env python3
"""Execute a sealed numeric report-edit experiment without a neural observer."""

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
from grain_research.construction import math_evidence_update_integrability as M

DEFAULT = ROOT / "results/grain_math_response_v1/evidence_update_integrability_r1"
CONFIG = "configs/grain_evidence_update_integrability_hm163.json"
CODE = (
    CONFIG,
    "grain_research/construction/math_evidence_update_integrability.py",
    "scripts/run_grain_evidence_update_integrability_hm163.py",
    "scripts/verify_grain_evidence_update_integrability_hm163.py",
    "tests/test_grain_evidence_update_integrability.py",
    "docs/GRAIN_EVIDENCE_UPDATE_INTEGRABILITY_PLAN_HM163.md",
    "grain_research/construction/math_source_update_support.py",
    "grain_research/construction/math_target_carrier_closure.py",
)


def read(path):
    """Read a specified public protocol or numeric-result metadata file."""
    return json.loads(Path(path).read_text())


def write(path, value):
    """Write finite JSON, refusing implicit nonfinite numeric repair."""
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path):
    """Hash an input without loading the full memory archive into Python bytes."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(out):
    """Authenticate existing evidence and seal the new plan before loop analysis."""
    config = read(ROOT / CONFIG)
    parent = ROOT / config["parent"]
    old = read(parent / "manifest.json")
    source = old["source_operations"]
    if digest(ROOT / source) != old["hashes"][source]:
        raise ValueError("Registered source memory changed")
    if digest(parent / "rows.json") != old["inputs"]["rows.json"]:
        raise ValueError("Registered report semantics changed")
    rows = read(parent / "rows.json")
    if len(rows) != config["rows"]:
        raise ValueError("No support expansion or deletion")
    out.mkdir(parents=True, exist_ok=False)
    write(out / "rows.json", rows)
    inputs = (
        source,
        config["parent"] + "/manifest.json",
        config["parent"] + "/rows.json",
    )
    write(
        out / "manifest.json",
        {
            **config,
            "source_operations": source,
            "hashes": {p: digest(ROOT / p) for p in (*CODE, *inputs)},
            "row_hash": digest(out / "rows.json"),
            "source_endpoint_type": "PUBLIC_FICTIONAL_REPORT_ATTRIBUTION_NOT_FUTURE_BEHAVIOR",
            "behavioral_anchor_bridge_complete": False,
        },
    )
    write(out / "status.json", {"status": "PREPARED", "active_job": None})
    print(json.dumps({"prepared": str(out), "new_neural_calls": 0}))


def authenticate(out):
    """Verify the same source material and code that were sealed prospectively."""
    manifest = read(out / "manifest.json")
    for name, checksum in manifest["hashes"].items():
        if digest(ROOT / name) != checksum:
            raise ValueError("Sealed dependency changed: " + name)
    if digest(out / "rows.json") != manifest["row_hash"]:
        raise ValueError("Sealed report semantics changed")
    return manifest


def norm(value):
    """Measure carrier displacement, not a psychological effect or accuracy."""
    return float(np.linalg.norm(np.asarray(value).ravel()))


def csv_write(path, rows):
    """Save complete finite conditions with explicit column names."""
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(out):
    """Compare semantic-equivalent edit paths once, without model selection."""
    start = time.monotonic()
    manifest = authenticate(out)
    if read(out / "status.json")["status"] != "PREPARED":
        raise RuntimeError("Do not overwrite an already started finite analysis")
    write(
        out / "status.json", {"status": "RUNNING", "active_job": "finite_cpu_analysis"}
    )
    cases, paths, slot_stats = [], [], []
    done = 0
    try:
        rows = read(out / "rows.json")
        with np.load(
            ROOT / manifest["source_operations"], allow_pickle=False
        ) as source:
            for condition in (0, 1):
                for order in (0, 1):
                    selected = [
                        r
                        for r in rows
                        if r["condition"] == condition and r["order"] == order
                    ]
                    if sorted((r["e"], r["n"]) for r in selected) != [
                        (-1, -1),
                        (-1, 1),
                        (1, -1),
                        (1, 1),
                    ]:
                        raise ValueError("Complete report-attribution support required")
                    for channel in manifest["channels"]:
                        cube = source[f"{condition}_{order}_{channel}"].astype(
                            np.float64
                        )
                        coefficients = M.compile_cube(cube)
                        for row in selected:
                            state = (row["e"], row["n"])
                            base = cube[(state[0] + 1) // 2, (state[1] + 1) // 2]
                            for policy in manifest["policies"]:
                                outputs = {}
                                for word in manifest["words"]:
                                    delta, end = M.walk(
                                        coefficients, row, state, word, policy
                                    )
                                    outputs[word] = delta
                                    paths.append(
                                        {
                                            "row": row["id"],
                                            "channel": channel,
                                            "policy": policy,
                                            "word": word or "identity",
                                            "start_e": state[0],
                                            "start_n": state[1],
                                            "end_e": end[0],
                                            "end_n": end[1],
                                            "semantic_identity": end == state,
                                            "displacement_norm": norm(delta),
                                        }
                                    )
                                loop = outputs["enen"]
                                predicted = M.predicted_loop(
                                    coefficients, row, state, policy
                                )
                                partner = cube[(1 - state[0]) // 2, (1 - state[1]) // 2]
                                i = int(np.argmax(np.abs(loop)))
                                index = np.unravel_index(i, loop.shape)
                                natural = norm(partner - base)
                                cases.append(
                                    {
                                        "row": row["id"],
                                        "condition": condition,
                                        "order": order,
                                        "e": state[0],
                                        "n": state[1],
                                        "channel": channel,
                                        "policy": policy,
                                        "single_involution_max": max(
                                            float(np.max(np.abs(outputs[w])))
                                            for w in ("ee", "nn")
                                        ),
                                        "loop_norm": norm(loop),
                                        "loop_max": float(np.max(np.abs(loop))),
                                        "loop_relative_to_whole_two_edit": norm(loop)
                                        / natural
                                        if natural
                                        else None,
                                        "signed_formula_max_error": float(
                                            np.max(np.abs(loop - predicted))
                                        ),
                                        "route_loop_max_error": float(
                                            np.max(
                                                np.abs(
                                                    outputs["en"] - outputs["ne"] - loop
                                                )
                                            )
                                        ),
                                        "reverse_loop_max_error": float(
                                            np.max(np.abs(outputs["nene"] + loop))
                                        ),
                                        "natural_two_edit_discrepancy_norm": norm(
                                            base + outputs["en"] - partner
                                        ),
                                        "natural_two_edit_relative_discrepancy": norm(
                                            base + outputs["en"] - partner
                                        )
                                        / natural
                                        if natural
                                        else None,
                                        "witness_layer": int(index[0]),
                                        "witness_head": int(index[1]),
                                        "witness_slot": int(index[2]),
                                        "witness_channel": int(index[3]),
                                        "witness_signed_value": float(loop[index]),
                                    }
                                )
                                slot_stats.append(
                                    np.sqrt(np.mean(loop**2, axis=(0, 1, 3)))
                                )
                                done += 1
                    if time.monotonic() - start > manifest["seconds_cap"]:
                        raise TimeoutError(
                            "Finite CPU budget reached; no automatic extension"
                        )
        csv_write(out / "cases.csv", cases)
        csv_write(out / "paths.csv", paths)
        np.savez_compressed(out / "loop_slot_rms.npz", values=np.stack(slot_stats))
        summary = {
            "packet": manifest["packet"],
            "cases": len(cases),
            "paths": len(paths),
            "policies": {},
            "new_neural_calls": 0,
            "source_observed_future_endpoints": 0,
            "original_scope_complete": False,
            "seconds": time.monotonic() - start,
        }
        for policy in manifest["policies"]:
            group = [r for r in cases if r["policy"] == policy]
            summary["policies"][policy] = {
                "cases": len(group),
                "nonzero_loops": sum(
                    r["loop_max"] > manifest["algebra_tolerance"] for r in group
                ),
                "loop_relative_mean": float(
                    np.mean([r["loop_relative_to_whole_two_edit"] for r in group])
                ),
                "loop_max": max(r["loop_max"] for r in group),
                "single_involution_max": max(r["single_involution_max"] for r in group),
                "signed_formula_max_error": max(
                    r["signed_formula_max_error"] for r in group
                ),
                "natural_two_edit_relative_mean": float(
                    np.mean([r["natural_two_edit_relative_discrepancy"] for r in group])
                ),
            }
        write(out / "summary.json", summary)
        write(
            out / "output_seal.json",
            {
                p: digest(out / p)
                for p in ("summary.json", "cases.csv", "paths.csv", "loop_slot_rms.npz")
            },
        )
        write(
            out / "status.json",
            {
                "status": "COMPLETE",
                "active_job": None,
                "cases": done,
                "seconds": time.monotonic() - start,
            },
        )
        print(json.dumps(summary, indent=2))
    except BaseException as error:
        write(
            out / "status.json",
            {
                "status": "FAILED_RETAINED",
                "active_job": None,
                "cases": done,
                "error": repr(error),
            },
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "analyze"))
    parser.add_argument("--out", type=Path, default=DEFAULT)
    arguments = parser.parse_args()
    (prepare if arguments.mode == "prepare" else analyze)(arguments.out)
