#!/usr/bin/env python3
"""Replay report-edit sums via direct corners, not the primary edge/walk code."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_grain_evidence_update_integrability_hm163 import (
    DEFAULT,
    authenticate,
    digest,
    read,
    write,
)


def project(value, slots):
    """Independently retain specified original time rows."""
    result = np.zeros_like(value)
    for slot in slots:
        result[:, :, slot, :] = value[:, :, slot, :]
    return result


def edges(cube, row, state, policy, axis):
    """Use source corner differences or a separately reconstructed potential."""
    e, n = state
    i, j = (e + 1) // 2, (n + 1) // 2
    kind = policy.split("_")[1]
    es = (
        row["scopes"]["neuroticism"]
        if kind == "other"
        else list(range(min(row["scopes"]["extraversion"]), cube.shape[4]))
    )
    ns = (
        row["scopes"]["extraversion"]
        if kind == "other"
        else list(range(min(row["scopes"]["neuroticism"]), cube.shape[4]))
    )
    if policy.startswith("shared_"):
        sectors = (
            sum(cube[a, b] for a in (0, 1) for b in (0, 1)) / 4,
            (-cube[0, 0] - cube[0, 1] + cube[1, 0] + cube[1, 1]) / 4,
            (-cube[0, 0] + cube[0, 1] - cube[1, 0] + cube[1, 1]) / 4,
            (cube[0, 0] - cube[0, 1] - cube[1, 0] + cube[1, 1]) / 4,
        )
        common = project(sectors[3], sorted(set(es) | set(ns)))
        return (
            -2 * e * project(sectors[1], es) - 2 * e * n * common
            if axis == "e"
            else -2 * n * project(sectors[2], ns) - 2 * e * n * common
        )
    if policy.startswith("relation_"):
        delta = -e * n * (cube[0, 0] - cube[0, 1] - cube[1, 0] + cube[1, 1]) / 2
    else:
        delta = (
            cube[1 - i, j] - cube[i, j] if axis == "e" else cube[i, 1 - j] - cube[i, j]
        )
    return project(delta, es if axis == "e" else ns)


def main():
    """Authenticate and compare every fixed path and saved loop statistic."""
    out = DEFAULT
    manifest = authenticate(out)
    for name, checksum in read(out / "output_seal.json").items():
        if digest(out / name) != checksum:
            raise ValueError("Saved primary result changed: " + name)
    rows = {r["id"]: r for r in read(out / "rows.json")}
    with (out / "paths.csv").open() as stream:
        paths = list(csv.DictReader(stream))
    with (out / "cases.csv").open() as stream:
        cases = list(csv.DictReader(stream))
    errors, comparisons, loops = [], 0, {}
    with np.load(ROOT / manifest["source_operations"], allow_pickle=False) as source:
        cubes = {k: source[k].astype(np.float64) for k in source.files}
        for entry in paths:
            row = rows[int(entry["row"])]
            cube = cubes[f"{row['condition']}_{row['order']}_{entry['channel']}"]
            state = (int(entry["start_e"]), int(entry["start_n"]))
            result = np.zeros_like(cube[0, 0])
            word = "" if entry["word"] == "identity" else entry["word"]
            for axis in word:
                result += edges(cube, row, state, entry["policy"], axis)
                state = (-state[0], state[1]) if axis == "e" else (state[0], -state[1])
            if state != (int(entry["end_e"]), int(entry["end_n"])):
                raise ValueError("Semantic endpoint mismatch")
            errors.append(
                abs(
                    float(np.linalg.norm(result.ravel()))
                    - float(entry["displacement_norm"])
                )
            )
            if word == "enen":
                loops[int(entry["row"]), entry["channel"], entry["policy"]] = result
            comparisons += 1
        slot_values = np.load(out / "loop_slot_rms.npz", allow_pickle=False)["values"]
        for index, case in enumerate(cases):
            loop = loops[int(case["row"]), case["channel"], case["policy"]]
            errors.append(abs(float(np.max(np.abs(loop))) - float(case["loop_max"])))
            errors.append(
                float(
                    np.max(
                        np.abs(
                            np.sqrt(np.mean(loop**2, axis=(0, 1, 3)))
                            - slot_values[index]
                        )
                    )
                )
            )
            comparisons += 2
    maximum = max(errors)
    if maximum > manifest["algebra_tolerance"]:
        raise ValueError("Alternate direct-corner replay failed: " + str(maximum))
    audit = {
        "status": "PASS",
        "comparisons": comparisons,
        "max_error": maximum,
        "scope": "same_agent_alternate_arithmetic_not_blind",
        "new_neural_calls": 0,
    }
    write(out / "alternate_audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
