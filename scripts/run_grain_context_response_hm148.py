#!/usr/bin/env python3
"""Cross source/application locations and changes; no new reader fit."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_context_response as R
from scripts import run_grain_semantic_anchor_hm147 as H

PARENT = H.DEFAULT
DEFAULT = ROOT / "results/grain_math_response_v1/context_response_r1"
FIELDS = ("source_midpoints", "source_displacements", "midpoints", "displacements")
CODE = (
    "grain_research/construction/math_context_response.py",
    "scripts/run_grain_context_response_hm148.py",
    "tests/test_grain_context_response.py",
    "docs/GRAIN_CONTEXT_RESPONSE_PLAN_HM148.md",
)


def parent_check():
    """Authenticate the frozen parent constructor, numeric inputs and outputs."""
    H.check(PARENT)
    for filename in ("source_seal.json", "output_seal.json"):
        for name, value in H.load(PARENT / filename).items():
            if H.digest(PARENT / name) != value:
                raise ValueError("Parent scientific artifact changed: " + name)


def prepared_pairs():
    """Assemble every controlled high/low pair in unchanged source units."""
    program = H.archive(
        PARENT / "program.npz", ("origin", "scale", "coefficients", "norms")
    )
    pairs = {}
    for role, metaname, arrayname in (
        ("source", "source_metadata.json", "source_input.npz"),
        ("application", "application_metadata.json", "cold_input.npz"),
    ):
        rows = H.load(PARENT / metaname)
        raw = H.archive(PARENT / arrayname, ("views",))["views"]
        z = H.M.standardize(raw, program["origin"], program["scale"])
        for i, row in enumerate(rows):
            if row["axis"] not in H.TARGETS:
                continue
            key = (row["axis"], row["family"], row["seed"], row["situation"])
            pair = pairs.setdefault(key, {"role": role, "metadata": row, "poles": {}})
            if row["pole"] in pair["poles"]:
                raise ValueError("Duplicate controlled pole")
            pair["poles"][row["pole"]] = (row["text_id"], z[i])
    if len(pairs) != 108:
        raise ValueError("All108 high/low generation pairs required")
    for pair in pairs.values():
        if set(pair["poles"]) != {-1, 1}:
            raise ValueError("Incomplete generation pair")
        high, low = pair["poles"][1][1], pair["poles"][-1][1]
        pair["m"], pair["d"] = (high + low) / 2, high - low
    return program, pairs


def prepare(out):
    """Freeze the complete crossed design before numerical response analysis."""
    parent_check()
    _, pairs = prepared_pairs()
    arrays, rows = {key: [] for key in FIELDS}, []
    for appkey, app in sorted(pairs.items()):
        if app["role"] != "application":
            continue
        source_keys = [
            key
            for key, p in pairs.items()
            if p["role"] == "source" and key[:3] == appkey[:3]
        ]
        if len(source_keys) != 3:
            raise ValueError("Three source task pairs required per application pair")
        for sourcekey in sorted(source_keys):
            source = pairs[sourcekey]
            for key, value in zip(
                FIELDS, (source["m"], source["d"], app["m"], app["d"])
            ):
                arrays[key].append(value)
            rows.append(
                {
                    "manipulated_axis": appkey[0],
                    "family": appkey[1],
                    "seed": appkey[2],
                    "application_task": appkey[3],
                    "source_task": sourcekey[3],
                    "role": app["metadata"]["role"],
                    "application_high": app["poles"][1][0],
                    "application_low": app["poles"][-1][0],
                    "source_high": source["poles"][1][0],
                    "source_low": source["poles"][-1][0],
                }
            )
    if len(rows) != 162:
        raise ValueError("Exactly162 complete crossings required")
    out.mkdir(parents=True, exist_ok=False)
    H.write_npz(out / "input.npz", **{k: np.asarray(v) for k, v in arrays.items()})
    H.write_json(out / "metadata.json", rows)
    H.write_json(
        out / "manifest.json",
        {
            "packet": "HM148",
            "comparisons": len(rows),
            "independent_application_pairs": 54,
            "code": {name: H.digest(ROOT / name) for name in CODE},
            "inputs": {
                name: H.digest(out / name) for name in ("input.npz", "metadata.json")
            },
            "parent": {
                name: H.digest(PARENT / name)
                for name in (
                    "program.npz",
                    "manifest.json",
                    "source_seal.json",
                    "output_seal.json",
                )
            },
            "scope": "Numeric crossed endpoints, not new valid texts, independent people or psychological causes",
        },
    )
    print(json.dumps({"prepared": str(out), "comparisons": len(rows)}))


def check(out):
    """Authenticate immutable new inputs/code and the unchanged parent program."""
    parent_check()
    manifest = H.load(out / "manifest.json")
    for root, key in ((ROOT, "code"), (out, "inputs"), (PARENT, "parent")):
        for name, value in manifest[key].items():
            if H.digest(root / name) != value:
                raise ValueError("Frozen input/code changed: " + name)


def score(program_path, input_path, output_path):
    """Cold numeric execution with no design metadata, text or target labels."""
    program = H.archive(program_path, ("origin", "scale", "coefficients", "norms"))
    inputs = H.archive(input_path, FIELDS)
    values, parts, location, change, interaction = R.crossed_account(
        **inputs, coefficients=program["coefficients"]
    )
    H.write_npz(
        output_path,
        values=values,
        odd_parts=parts,
        location=location,
        change=change,
        interaction=interaction,
    )
    print(
        json.dumps(
            {
                "scored": len(values),
                "metadata_received": False,
                "output": str(output_path),
            }
        )
    )


def analyze(out):
    """Join saved outputs to the fixed crossed design without selecting a reader."""
    check(out)
    start = time.monotonic()
    rows = H.load(out / "metadata.json")
    result = H.archive(
        out / "scores.npz", ("values", "odd_parts", "location", "change", "interaction")
    )
    if result["values"].shape != (162, 3, 2, 4):
        raise ValueError("All arms/targets/corners required")
    first_masks, third_masks = (1, 2, 4, 8), (7, 11, 13, 14)
    table = []
    for i, metadata in enumerate(rows):
        for arm, armname in enumerate(H.ARMS):
            for target, name in enumerate(H.TARGETS):
                values = result["values"][i, arm, target]
                parts = result["odd_parts"][i, arm, target]
                table.append(
                    metadata
                    | {"arm": armname, "target": name}
                    | {
                        f"r{a}{b}": float(values[2 * a + b])
                        for a in range(2)
                        for b in range(2)
                    }
                    | {
                        key: float(result[key][i, arm, target])
                        for key in ("location", "change", "interaction")
                    }
                    | {
                        f"linear{a}{b}": float(parts[2 * a + b, first_masks].sum())
                        for a in range(2)
                        for b in range(2)
                    }
                    | {
                        f"cubic{a}{b}": float(parts[2 * a + b, third_masks].sum())
                        for a in range(2)
                        for b in range(2)
                    }
                )
    summaries = []
    value_keys = [
        key for key in table[0] if key.startswith(("r0", "r1", "linear", "cubic"))
    ] + ["location", "change", "interaction"]
    for role in ("development", "evaluation", "all_application"):
        for axis in H.TARGETS:
            for target in H.TARGETS:
                for arm in H.ARMS:
                    selected = [
                        r
                        for r in table
                        if (role == "all_application" or r["role"] == role)
                        and r["manipulated_axis"] == axis
                        and r["target"] == target
                        and r["arm"] == arm
                    ]
                    summaries.append(
                        {
                            "role": role,
                            "manipulated_axis": axis,
                            "target": target,
                            "arm": arm,
                            "crossings": len(selected),
                            "application_pairs": len(
                                {r["application_high"] for r in selected}
                            ),
                        }
                        | {
                            key: float(np.mean([r[key] for r in selected]))
                            for key in value_keys
                        }
                    )
    H.write_csv(out / "crossed_effects.csv", table)
    H.write_csv(out / "effects_by_role.csv", summaries)
    H.write_json(
        out / "summary.json",
        {
            "status": "EXECUTED_LOCATION_BY_CHANGE_DIAGNOSTIC_NOT_PERSONALITY_PREDICTOR",
            "role_effects": summaries,
            "comparisons": len(rows),
            "effect_rows": len(table),
            "seconds": time.monotonic() - start,
            "new_model_API_training_private_calls": 0,
            "scope": "Exact polynomial actions on artificial numeric crossed vectors; all observed text endpoints and three source anchors retained,not a denoising score",
        },
    )
    H.write_json(
        out / "output_seal.json",
        {
            name: H.digest(out / name)
            for name in (
                "scores.npz",
                "crossed_effects.csv",
                "effects_by_role.csv",
                "summary.json",
            )
        },
    )
    print(json.dumps({"effect_rows": len(table), "role_cells": len(summaries)}))


def verify(out):
    """Replay all response corners with the independent source product kernel."""
    check(out)
    for name, value in H.load(out / "output_seal.json").items():
        if H.digest(out / name) != value:
            raise ValueError("Frozen result changed: " + name)
    program, pairs = prepared_pairs()
    meta = H.load(out / "metadata.json")
    inp = H.archive(out / "input.npz", FIELDS)
    got = H.archive(
        out / "scores.npz", ("values", "odd_parts", "location", "change", "interaction")
    )
    _, _, z, relinked, weights, _, _ = H.source_objects(PARENT)
    norms = program["norms"]

    def kernel_scores(x):
        """Independently contract source weights with full/single/relinked kernels."""
        full = H.M.product_kernel(x, z) @ weights / norms
        single = (1 + sum(x[:, v] @ z[:, v].T for v in range(4))) @ weights / norms
        changed = H.M.product_kernel(x, relinked) @ weights / norms
        return np.stack((full, single, changed), axis=1)

    corners = (
        (inp["source_midpoints"], inp["source_displacements"]),
        (inp["source_midpoints"], inp["displacements"]),
        (inp["midpoints"], inp["source_displacements"]),
        (inp["midpoints"], inp["displacements"]),
    )
    expected = np.stack(
        [kernel_scores(m + d / 2) - kernel_scores(m - d / 2) for m, d in corners],
        axis=-1,
    )
    errors = {
        "all_corners_kernel": float(abs(expected - got["values"]).max()),
        "odd_sum": float(abs(got["odd_parts"].sum(-1) - got["values"]).max()),
        "allocation_sum": float(
            abs(
                got["location"]
                + got["change"]
                - (got["values"][..., 3] - got["values"][..., 0])
            ).max()
        ),
    }
    amplitude_errors, gauge_errors, input_errors = [], [], []
    shift = np.arange(16, dtype=float).reshape(4, 4) / 17 - 0.4
    moved = R.translated_coefficients(program["coefficients"], shift)
    for corner, (m, d) in enumerate(corners):
        doubled = kernel_scores(m + d) - kernel_scores(m - d)
        cubic = (doubled - 2 * expected[..., corner]) / 6
        linear = expected[..., corner] - cubic
        corner_parts = got["odd_parts"][..., corner, :]
        amplitude_errors.append(
            float(abs(cubic - corner_parts[..., [7, 11, 13, 14]].sum(-1)).max())
        )
        amplitude_errors.append(
            float(abs(linear - corner_parts[..., [1, 2, 4, 8]].sum(-1)).max())
        )
        gauge_errors.append(
            float(abs(R.response(m - shift, d, moved) - expected[..., corner]).max())
        )
    for i, row in enumerate(meta):
        for prefix, task, fields in (
            ("source", row["source_task"], FIELDS[:2]),
            ("application", row["application_task"], FIELDS[2:]),
        ):
            pair = pairs[(row["manipulated_axis"], row["family"], row["seed"], task)]
            if pair["role"] != prefix:
                raise ValueError("Crossing roles changed")
            input_errors.extend(
                [
                    float(abs(inp[fields[0]][i] - pair["m"]).max()),
                    float(abs(inp[fields[1]][i] - pair["d"]).max()),
                ]
            )
    errors |= {
        "odd_via_double_amplitude": max(amplitude_errors),
        "coordinate_translation": max(gauge_errors),
        "original_pair_reconstruction": max(input_errors),
    }
    with (out / "crossed_effects.csv").open() as stream:
        table = list(csv.DictReader(stream))
    if len(table) != 972:
        raise ValueError("Complete crossed effect table required")
    table_errors = []
    for i, metadata in enumerate(meta):
        for arm, armname in enumerate(H.ARMS):
            for target, name in enumerate(H.TARGETS):
                row = table[6 * i + 2 * arm + target]
                for key, value in metadata.items():
                    if row[key] != str(value):
                        raise ValueError("Crossed metadata mismatch: " + key)
                if row["arm"] != armname or row["target"] != name:
                    raise ValueError("Crossed reader mismatch")
                for corner in range(4):
                    suffix = f"{corner // 2}{corner % 2}"
                    table_errors.append(
                        abs(float(row["r" + suffix]) - expected[i, arm, target, corner])
                    )
                    parts = got["odd_parts"][i, arm, target, corner]
                    for prefix, masks in (
                        ("linear", [1, 2, 4, 8]),
                        ("cubic", [7, 11, 13, 14]),
                    ):
                        table_errors.append(
                            abs(float(row[prefix + suffix]) - parts[masks].sum())
                        )
                r00, r01, r10, r11 = expected[i, arm, target]
                allocated = {
                    "location": ((r10 - r00) + (r11 - r01)) / 2,
                    "change": ((r01 - r00) + (r11 - r10)) / 2,
                    "interaction": r11 - r10 - r01 + r00,
                }
                for key, value in allocated.items():
                    table_errors.extend(
                        (
                            abs(float(row[key]) - value),
                            abs(got[key][i, arm, target] - value),
                        )
                    )
    with (out / "effects_by_role.csv").open() as stream:
        summaries = list(csv.DictReader(stream))
    if len(summaries) != 36 or H.load(out / "summary.json")["effect_rows"] != 972:
        raise ValueError("Complete role summaries required")
    for summary in summaries:
        selected = [
            row
            for row in table
            if (summary["role"] == "all_application" or row["role"] == summary["role"])
            and all(
                row[key] == summary[key]
                for key in ("manipulated_axis", "target", "arm")
            )
        ]
        if int(summary["crossings"]) != len(selected) or int(
            summary["application_pairs"]
        ) != len({row["application_high"] for row in selected}):
            raise ValueError("Source crossings are not independent application pairs")
        for key in summary:
            if key.startswith(("r0", "r1", "linear", "cubic")) or key in (
                "location",
                "change",
                "interaction",
            ):
                table_errors.append(
                    abs(
                        float(summary[key])
                        - sum(float(row[key]) for row in selected) / len(selected)
                    )
                )
    errors["complete_effect_and_role_tables"] = max(table_errors)
    if max(errors.values()) > 1e-9:
        raise ValueError(
            "Alternate arithmetic or complete-table check failed: " + str(errors)
        )
    print(
        json.dumps(
            {
                "verified": True,
                "checks": errors,
                "corner_scalar_readings": expected.size,
                "effect_rows": len(table),
                "source_program_unchanged": True,
                "scope": "Same-agent alternate arithmetic;not independentblind replication or valid-text intervention",
            }
        )
    )


def main():
    """Expose separate prepare/cold-score/analyze/readonly-verify stages."""
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("action", choices=("prepare", "score", "analyze", "verify"))
    parser.add_argument("--out", type=Path, default=DEFAULT)
    parser.add_argument("--program", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "score":
        if not all((args.program, args.input, args.output)):
            parser.error("Cold scoring requires explicit numeric paths")
        score(args.program, args.input, args.output)
    else:
        {"prepare": prepare, "analyze": analyze, "verify": verify}[args.action](
            args.out
        )


if __name__ == "__main__":
    main()
